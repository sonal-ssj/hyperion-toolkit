"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import os
from collections import OrderedDict as ODict

import logging

import torch
import torch.nn as nn

from ..utils import MetricAcc, TorchDDP
from .xvector_trainer import XVectorTrainer


class XVectorTrainerFromWavTwoStream(XVectorTrainer):
    """Trainer to train x-vector style models but using two-stream data.

    Attributes:
      model: x-Vector model object.
      feat_extractor: feature extractor nn.Module
      optim: pytorch optimizer object or options dict
      epochs: max. number of epochs
      exp_path: experiment output path
      cur_epoch: current epoch
      grad_acc_steps: gradient accumulation steps to simulate larger batch size.
      device: cpu/gpu device
      metrics: extra metrics to compute besides cxe.
      lrsched: learning rate scheduler object or options dict.
      loggers: LoggerList object, loggers write training progress to std. output and file.
      ddp: if True use distributed data parallel training
      ddp_type: type of distributed data parallel in  (ddp, oss_ddp, oss_shared_ddp)
      loss: if None, it uses cross-entropy
      train_mode: training mode in ['train', 'ft-full', 'ft-last-layer']
      use_amp: uses mixed precision training.
      log_interval: number of optim. steps between log outputs
      use_tensorboard: use tensorboard logger
      use_wandb: use wandb logger
      wandb: wandb dictionary of options
      grad_clip: norm to clip gradients, if 0 there is no clipping
      grad_clip_norm: norm type to clip gradients
      swa_start: epoch to start doing swa
      swa_lr: SWA learning rate
      swa_anneal_epochs: SWA learning rate anneal epochs
      cpu_offload: CPU offload of gradients when using fully sharded ddp
    """

    def __init__(
        self,
        model,
        feat_extractor,
        optim={},
        epochs=100,
        exp_path="./train",
        cur_epoch=0,
        grad_acc_steps=1,
        device=None,
        metrics=None,
        lrsched=None,
        loggers=None,
        ddp=False,
        ddp_type="ddp",
        loss=None,
        train_mode="train",
        use_amp=False,
        log_interval=10,
        use_tensorboard=False,
        use_wandb=False,
        wandb={},
        grad_clip=0,
        grad_clip_norm=2,
        swa_start=0,
        swa_lr=1e-3,
        swa_anneal_epochs=10,
        cpu_offload=False,
    ):

        super().__init__(
            model,
            optim,
            epochs,
            exp_path,
            cur_epoch=cur_epoch,
            grad_acc_steps=grad_acc_steps,
            device=device,
            metrics=metrics,
            lrsched=lrsched,
            loggers=loggers,
            ddp=ddp,
            ddp_type=ddp_type,
            loss=loss,
            train_mode=train_mode,
            use_amp=use_amp,
            log_interval=log_interval,
            use_tensorboard=use_tensorboard,
            use_wandb=use_wandb,
            wandb=wandb,
            grad_clip=grad_clip,
            grad_clip_norm=grad_clip_norm,
            swa_start=swa_start,
            swa_lr=swa_lr,
            swa_anneal_epochs=swa_anneal_epochs,
            cpu_offload=cpu_offload,
        )

        self.feat_extractor = feat_extractor
        if device is not None:
            self.feat_extractor.to(device)

        # if ddp:
        #     self.feat_extractor = TorchDDP(self.feat_extractor)

    def train_epoch(self, data_loader):
        """Training epoch loop

        Args:
          data_loader: pytorch data loader returning features and class labels.
        """

        self.model.update_loss_margin(self.cur_epoch)

        metric_acc = MetricAcc(device=self.device)
        batch_metrics = ODict()
        self.set_train_mode()

        # logging.info('next(iter(data_loader))={}'.format(next(iter(data_loader))))

        for batch, (data, target) in enumerate(data_loader):
            self.loggers.on_batch_begin(batch)
            if batch % self.grad_acc_steps == 0:
                self.optimizer.zero_grad()

            # logging.info('data={}'.format(data))
            # logging.info('data.shape={}'.format(data.shape))


            data, target = data.to(self.device), target.to(self.device)
            
            batch_size = data.shape[0]
            with torch.no_grad():
                dataA, dataB = data[:,0], data[:,1]
                #feats = self.feat_extractor(data)
                featsA = self.feat_extractor(dataA)
                featsB = self.feat_extractor(dataB)
                if self.model.in_channels==1:
                    feats = torch.cat((featsA, featsB), axis=1)
                elif self.model.in_channels==2:
                    feats = torch.cat((featsA.unsqueeze(1), featsB.unsqueeze(1)), axis=1)
                else:
                    logging.error('self.model.in_channels must be 1 or 2, others not implemented')

                # try:
                #     logging.info('self.model={}'.format(self.model))
                # except:
                #     logging.info('Logging info failed')
                # try:
                #     logging.info('self.model.get_config()["in_channels"]={}'.format(self.model.get_config()["in_channels"]))    
                # except:
                #     logging.info('Logging info failed')
                # try:
                #     logging.info('self.model.in_channels={}'.format(self.model.in_channels))
                # except:
                #     logging.info('Logging info failed')
                # try:
                #     logging.info('self.model["in_channels"]={}'.format(self.model["in_channels"]))
                # except:
                #     logging.info('Logging info failed')

            with self.amp_autocast():
                output = self.model(feats, target)
                loss = self.loss(output, target).mean() / self.grad_acc_steps

            if self.use_amp:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch + 1) % self.grad_acc_steps == 0:
                if self.lr_scheduler is not None and not self.in_swa:
                    self.lr_scheduler.on_opt_step()
                self.update_model()

            batch_metrics["loss"] = loss.item() * self.grad_acc_steps
            for k, metric in self.metrics.items():
                batch_metrics[k] = metric(output, target)

            metric_acc.update(batch_metrics, batch_size)
            logs = metric_acc.metrics
            logs["lr"] = self._get_lr()
            self.loggers.on_batch_end(logs=logs, batch_size=batch_size)

        logs = metric_acc.metrics
        logs = ODict(("train_" + k, v) for k, v in logs.items())
        logs["lr"] = self._get_lr()
        return logs

    def validation_epoch(self, data_loader, swa_update_bn=False):
        """Validation epoch loop

        Args:
          data_loader: PyTorch data loader return input/output pairs
        """
        metric_acc = MetricAcc(device=self.device)
        batch_metrics = ODict()
        with torch.no_grad():
            if swa_update_bn:
                log_tag = "train_"
                self.set_train_mode()
            else:
                log_tag = "val_"
                self.model.eval()

            for batch, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.shape[0]

                # feats = self.feat_extractor(data)
                dataA, dataB = data[:,0], data[:,1]
                featsA = self.feat_extractor(dataA)
                featsB = self.feat_extractor(dataB)
                if self.model.in_channels==1:
                    feats = torch.cat((featsA, featsB), axis=1)
                elif self.model.in_channels==2:
                    feats = torch.cat((featsA.unsqueeze(1), featsB.unsqueeze(1)), axis=1)
                else:
                    logging.error('self.model.in_channels must be 1 or 2, others not implemented')
                    
                with self.amp_autocast():
                    output = self.model(feats, **self.amp_args)
                    loss = self.loss(output, target)

                batch_metrics["loss"] = loss.mean().item()
                for k, metric in self.metrics.items():
                    batch_metrics[k] = metric(output, target)

                metric_acc.update(batch_metrics, batch_size)

        logs = metric_acc.metrics
        logs = ODict((log_tag + k, v) for k, v in logs.items())
        return logs
