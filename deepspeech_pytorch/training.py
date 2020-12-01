import json

import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from deepspeech_pytorch.checkpoint import CheckpointHandler, GCSCheckpointHandler
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig, CheckpointConfig
from deepspeech_pytorch.loader.data_module import DeepSpeechDataModule
from deepspeech_pytorch.model import DeepSpeech

import time

import torch
from pytorch_lightning import Callback


class CUDACallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        max_memory = torch.tensor(max_memory, dtype=torch.int, device=trainer.root_gpu)
        epoch_time = torch.tensor(epoch_time, dtype=torch.int, device=trainer.root_gpu)

        torch.distributed.all_reduce(max_memory, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(epoch_time, op=torch.distributed.ReduceOp.SUM)

        world_size = torch.distributed.get_world_size()

        print(f"Average Epoch time: {epoch_time.item() / float(world_size):.2f} seconds")
        print(f"Average Peak memory {max_memory.item() / float(world_size):.2f}MiB")


def train(cfg: DeepSpeechConfig):
    # Set seeds for determinism
    seed_everything(cfg.training.seed)

    with open(to_absolute_path(cfg.data.labels_path)) as label_file:
        labels = json.load(label_file)

    if OmegaConf.get_type(cfg.checkpointing) is CheckpointConfig:
        checkpoint_callback = CheckpointHandler(
            cfg=cfg.checkpointing
        )
    else:
        checkpoint_callback = GCSCheckpointHandler(
            cfg=cfg.checkpointing
        )
    if cfg.checkpointing.load_auto_checkpoint:
        latest_checkpoint = checkpoint_callback.find_latest_checkpoint()
        if latest_checkpoint:
            cfg.checkpointing.continue_from = latest_checkpoint

    data_loader = DeepSpeechDataModule(
        labels=labels,
        data_cfg=cfg.data,
        normalize=True,
        multigpu=cfg.training.multigpu
    )

    model = DeepSpeech(
        labels=labels,
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
        precision=cfg.training.precision,
        spect_cfg=cfg.data.spect
    )

    trainer = pl.Trainer(
        max_epochs=1,
        gpus=cfg.training.gpus,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=cfg.checkpointing.continue_from if cfg.checkpointing.continue_from else None,
        precision=cfg.training.precision.value,
        gradient_clip_val=cfg.optim.max_norm,
        replace_sampler_ddp=False,
        accelerator='ddp',
        callbacks=[CUDACallback()]
    )
    trainer.fit(model, data_loader)
