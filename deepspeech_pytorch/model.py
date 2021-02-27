import math
from typing import List, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.nn import CTCLoss

from deepspeech_pytorch.configs.train_config import SpectConfig, BiDirectionalConfig, OptimConfig, AdamConfig, \
    SGDConfig, UniDirectionalConfig, FullyConvolutionalConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.validation import CharErrorRate, WordErrorRate


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_
        
class GlobalAvgPool2d(nn.Module):
    def __init__(self,
                 device='cpu'):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))

class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x


class Lookahead(nn.Module):
    # Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    # input shape - sequence, batch, feature - TxNxH
    # output shape - same as input
    def __init__(self, n_features, context):
        super(Lookahead, self).__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = (0, self.context - 1)
        self.conv = nn.Conv1d(
            self.n_features,
            self.n_features,
            kernel_size=self.context,
            stride=1,
            groups=self.n_features,
            padding=0,
            bias=False
        )

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'


class DeepSpeech(pl.LightningModule):
    def __init__(self,
                 labels: List,
                 model_cfg: Union[UniDirectionalConfig, BiDirectionalConfig],
                 precision: int,
                 optim_cfg: Union[AdamConfig, SGDConfig],
                 spect_cfg: SpectConfig
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.model_cfg = model_cfg
        self.precision = precision
        self.optim_cfg = optim_cfg
        self.spect_cfg = spect_cfg
        self.bidirectional = True if OmegaConf.get_type(model_cfg) is BiDirectionalConfig else False
        self.fcn = True if OmegaConf.get_type(model_cfg) is FullyConvolutionalConfig else False

        
        self.labels = labels
        num_classes = len(self.labels)
        print('self.spect_cfg.window_size', self.spect_cfg.window_size)
        self.spec_features = int(math.floor((self.spect_cfg.sample_rate * self.spect_cfg.window_size) / 2) + 1)
        
        rnn_input_size = int(math.floor(self.spec_features + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32
        
        if not self.fcn:
            self.conv = MaskConv(nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True)
            ))
            
            # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
            
            self.rnns = nn.Sequential(
                BatchRNN(
                    input_size=rnn_input_size,
                    hidden_size=self.model_cfg.hidden_size,
                    rnn_type=self.model_cfg.rnn_type.value,
                    bidirectional=self.bidirectional,
                    batch_norm=False
                ),
                *(
                    BatchRNN(
                        input_size=self.model_cfg.hidden_size,
                        hidden_size=self.model_cfg.hidden_size,
                        rnn_type=self.model_cfg.rnn_type.value,
                        bidirectional=self.bidirectional
                    ) for x in range(self.model_cfg.hidden_layers - 1)
                )
            )

            fully_connected = nn.Sequential(
                nn.BatchNorm1d(self.model_cfg.hidden_size),
                nn.Linear(self.model_cfg.hidden_size, num_classes, bias=False)
            )
            self.fc = nn.Sequential(
                SequenceWise(fully_connected),
            )
            
        else:
            self.conv = MaskConv(nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True)
            ))
            
#             stride_red = (3,1)
#             stride_red2 = (2,1)
#             pad_red = (1,1)
            
#         #self.conv = MaskConv(nn.Sequential(
            
#             self.conv1_1 = nn.Conv2d(1, 64, 3, padding=100)
#             self.relu1_1 = nn.ReLU(inplace=True)
#             self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
#             self.relu1_2 = nn.ReLU(inplace=True)
#             self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

#             # conv2
#             self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
#             self.relu2_1 = nn.ReLU(inplace=True)
#             self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
#             self.relu2_2 = nn.ReLU(inplace=True)
#             self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

#             # conv3
#             self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
#             self.relu3_1 = nn.ReLU(inplace=True)
#             self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
#             self.relu3_2 = nn.ReLU(inplace=True)
#             self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
#             self.relu3_3 = nn.ReLU(inplace=True)
#             self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

#             # conv4
#             self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
#             self.relu4_1 = nn.ReLU(inplace=True)
#             self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
#             self.relu4_2 = nn.ReLU(inplace=True)
#             self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
#             self.relu4_3 = nn.ReLU(inplace=True)
#             self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

#             # conv5
#             self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
#             self.relu5_1 = nn.ReLU(inplace=True)
#             self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
#             self.relu5_2 = nn.ReLU(inplace=True)
#             self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
#             self.relu5_3 = nn.ReLU(inplace=True)
#             self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

#             # fc6
#             self.fc6 = nn.Conv2d(512, 2048, 7)
#             self.relu6 = nn.ReLU(inplace=True)
#             self.drop6 = nn.Dropout2d()

#             # fc7
#             self.fc7 = nn.Conv2d(2048, 2048, 1)
#             self.relu7 = nn.ReLU(inplace=True)
#             self.drop7 = nn.Dropout2d()

#             self.score_fr = nn.Conv2d(2048, num_classes, 1)
#             self.score_pool4 = nn.Conv2d(512, num_classes, 1)

#             self.upscore2 = nn.ConvTranspose2d(
#                 num_classes, num_classes, 4, stride=2, bias=False)

#             self.upscore16 = nn.ConvTranspose2d(
#                 num_classes, num_classes, 32, stride=(1,16), bias=False)

#             #self.avpool = nn.AvgPool2d(kernel_size=1, stride=1)
#             #self.globlavgpool = nn.AdaptiveAvgPool2d((1,1))
#             #self.flatten = nn.Flatten()
            
            c1d_kernel =3
            #c1d_kernel =1 
            channel_start_out = 2048
            conv1d_input = rnn_input_size #int(self.spec_features/16)*num_classes
            print('conv1d_input:', conv1d_input)
            self.conv1D =  nn.Sequential(
                nn.Conv1d(
                    conv1d_input,
                    channel_start_out,
                    kernel_size=3,
                    stride=1,
                    groups=1,
                    padding=1,
                    bias=False
                ),
                nn.ReLU(inplace=True),
                nn.AvgPool1d(1),
                nn.Conv1d(
                    channel_start_out,
                    int(channel_start_out*2),
                    kernel_size=c1d_kernel,
                    stride=1,
                    groups=1, #self.n_features,
                    padding=1,
                    bias=False
                ),
                nn.ReLU(inplace=True),
                nn.AvgPool1d(1),
                nn.Conv1d(
                    int(channel_start_out*2),
                    int(channel_start_out*4),
                    kernel_size=c1d_kernel,
                    stride=1,
                    groups=1, #self.n_features,
                    padding=1,
                    bias=False
                ),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    int(channel_start_out*4),
                    int(channel_start_out*4),
                    kernel_size=c1d_kernel,
                    stride=1,
                    groups=1, #self.n_features,
                    padding=1,
                    bias=False
                ),
                nn.AvgPool1d(1),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    int(channel_start_out*4),
                    int(channel_start_out*2),
                    kernel_size=c1d_kernel,
                    stride=1,
                    groups=1, #self.n_features,
                    padding=1,
                    bias=False
                ),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    int(channel_start_out*2),
                    channel_start_out,
                    kernel_size=c1d_kernel,
                    stride=1,
                    groups=1, #self.n_features,
                    padding=1,
                    bias=False
                ),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    channel_start_out,
                    int(channel_start_out/2),
                    kernel_size=c1d_kernel,
                    stride=1,
                    groups=1, #self.n_features,
                    padding=1,
                    bias=False
                ),
                nn.AvgPool1d(1),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    int(channel_start_out/2),
                    int(channel_start_out/4),
                    kernel_size=c1d_kernel,
                    stride=1,
                    groups=1, #self.n_features,
                    padding=1,
                    bias=False
                ),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    512,
                    128,
                    kernel_size=c1d_kernel,
                    stride=1,
                    groups=1, #self.n_features,
                    padding=1,
                    bias=False
                ),
                nn.AvgPool1d(1),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    128,
                    num_classes,
                    kernel_size=c1d_kernel,
                    stride=1,

                    groups=1, #self.n_features,
                    padding=1,
                    bias=False
                ))

            
        
        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            Lookahead(num_classes if self.fcn else self.model_cfg.hidden_size, context=self.model_cfg.lookahead_context),
            nn.Hardtanh(0, 20, inplace=True)
        ) if not self.bidirectional else None


        self.inference_softmax = InferenceBatchSoftmax()
        self.criterion = CTCLoss(blank=self.labels.index('_'), reduction='sum', zero_infinity=True)
        self.evaluation_decoder = GreedyDecoder(self.labels)  # Decoder used for validation
        self.wer = WordErrorRate(
            decoder=self.evaluation_decoder,
            target_decoder=self.evaluation_decoder
        )
        self.cer = CharErrorRate(
            decoder=self.evaluation_decoder,
            target_decoder=self.evaluation_decoder
        )

    def forward(self, x, lengths):
        
        #print(1, x.size())
        
        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths)
        #print('output_lengths', output_lengths)
        
        x, _ = self.conv(x, output_lengths)
        sizes = x.size()

        #print('3a', x.size())
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        #print('3b', x.size())
          # TxNxH
        #print(3, x.size())
        
        if not self.fcn:
            x = x.transpose(1, 2).transpose(0, 1).contiguous()
            
            for rnn in self.rnns:
                x = rnn(x, output_lengths)
                
            if not self.bidirectional:  # no need for lookahead layer in bidirectional
                x = self.lookahead(x)
            #print(4, x.size())
            x = self.fc(x)
            #print('5a', x.size())
            x = x.transpose(0, 1)
        else:
            x = self.conv1D(x)
            
            x = x.transpose(1, 2).contiguous()
            x = self.lookahead(x).transpose(0, 1)
            x = x.transpose(0, 1)
        
#         else:
#             xsize1 = x.size()
#             #print('x',x.size())
#             h = x
#             h = self.relu1_1(self.conv1_1(h))
#             h = self.relu1_2(self.conv1_2(h))
#             h = self.pool1(h)

#             h = self.relu2_1(self.conv2_1(h))
#             h = self.relu2_2(self.conv2_2(h))
#             h = self.pool2(h)

#             h = self.relu3_1(self.conv3_1(h))
#             h = self.relu3_2(self.conv3_2(h))
#             h = self.relu3_3(self.conv3_3(h))
#             h = self.pool3(h)

#             h = self.relu4_1(self.conv4_1(h))
#             h = self.relu4_2(self.conv4_2(h))
#             h = self.relu4_3(self.conv4_3(h))
#             h = self.pool4(h)
#             pool4 = h  # 1/16

#             h = self.relu5_1(self.conv5_1(h))
#             h = self.relu5_2(self.conv5_2(h))
#             h = self.relu5_3(self.conv5_3(h))
#             h = self.pool5(h)

#             h = self.relu6(self.fc6(h))
#             h = self.drop6(h)

#             h = self.relu7(self.fc7(h))
#             h = self.drop7(h)

#             h = self.score_fr(h)
#             h = self.upscore2(h)
#             upscore2 = h  # 1/16

#             h = self.score_pool4(pool4)
#             h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
#             score_pool4c = h  # 1/16

#             h = upscore2 + score_pool4c
            
#             h = self.upscore16(h)
            
#             #print('h1a', h.size(), xsize1[3], 27+xsize1[3])
            
#             h = h[:, :, 27:27 + int(self.spec_features/16), 27:27 + xsize1[3]].contiguous()
            
#             x = h
#             sizes = h.size()
            
#             #print('3a', x.size())
#             #x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
#             #x=torch.mean(x.transpose(1, 2),dim=1)
#             x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
            
# #             print('3a1', x.size())
# #             x = self.globlavgpool(x)
# #             print('3a2', x.size())
# #             x = self.flatten(x)
#             #print('3a3', x.size())
#             x = self.conv1D(x)
            
#             #print('3a4', x.size())
#             #print('3b', x.size())
#             #x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
#             #print(3, x.size())
#             x = x.transpose(1, 2).contiguous()  # TxNxH
            
#             #x = self.lookahead(x)
#             #print(4, x.size())
#             #x = x.transpose(0, 1)
            
#             #print(5, x.size())
            
            
        #print('bifs', x.size())
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        
        return x, output_lengths

    def training_step(self, batch, batch_idx):
        inputs, targets, input_percentages, target_sizes = batch
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        out, output_sizes = self(inputs, input_sizes)
        #print('last -1', out.size())
        out = out.transpose(0, 1)  # TxNxH
        out = out.log_softmax(-1)
        
        #print('last', out.size())
        loss = self.criterion(out, targets, output_sizes, target_sizes)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, input_percentages, target_sizes = batch
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(self.device)
        with autocast(enabled=self.precision == 16):
            out, output_sizes = self(inputs, input_sizes)
        decoded_output, _ = self.evaluation_decoder.decode(out, output_sizes)
        self.wer(
            preds=out,
            preds_sizes=output_sizes,
            targets=targets,
            target_sizes=target_sizes
        )
        self.cer(
            preds=out,
            preds_sizes=output_sizes,
            targets=targets,
            target_sizes=target_sizes
        )
        self.log('wer', self.wer.compute(), prog_bar=True, on_epoch=True)
        self.log('cer', self.cer.compute(), prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        if OmegaConf.get_type(self.optim_cfg) is SGDConfig:
            optimizer = torch.optim.SGD(
                params=self.parameters(),
                lr=self.optim_cfg.learning_rate,
                momentum=self.optim_cfg.momentum,
                nesterov=True,
                weight_decay=self.optim_cfg.weight_decay
            )
        elif OmegaConf.get_type(self.optim_cfg) is AdamConfig:
            optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=self.optim_cfg.learning_rate,
                betas=self.optim_cfg.betas,
                eps=self.optim_cfg.eps,
                weight_decay=self.optim_cfg.weight_decay
            )
        else:
            raise ValueError("Optimizer has not been specified correctly.")

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=self.optim_cfg.learning_anneal
        )
        return [optimizer], [scheduler]

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
#         if self.fcn:
#             return input_length
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) // m.stride[1] + 1)
        return seq_len.int()
