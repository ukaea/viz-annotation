"""
@authors: 
    Niraj Bhujel, SciML-STFC-UKRI (niraj.bhujel@stfc.ac.uk)
"""
import os
import json
import time
import torch
import pickle
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from copy import copy
from tqdm import tqdm
from omegaconf import DictConfig
from collections import defaultdict, Counter
from multiprocessing import cpu_count
import sklearn.metrics as skmetrics

from utils.misc import *
from utils.metrics import ClassificationMetricLogger
from utils.checkpoint import ModelCheckpoint
from utils.earlystopping import EarlyStopping
from utils.schedulers import cosine_scheduler
from dataset.datasets import ELMDataset, split_data, collate_fn
from model.network import Network
from model.losses import BCELoss


import torch
from torch.nn import functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DS

class Trainer:

    def __init__(
        self,
        cfg: DictConfig,
        rank: int = 0,
        world_size: int = 1,
        trace_func=print,
        exp_name='exp',
        **kwargs,
    ):

        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.trace_func = trace_func
        self.exp_name = exp_name
        self.exp_dir = f"{cfg.exp.log_dir}/{exp_name}"
        self.debug = cfg.train.debug

        # set seed
        set_random_seed(cfg.rng.seed)

        # seed generators
        self.rng_gen = np.random.default_rng(cfg.rng.seed)
        self.torch_gen = torch.manual_seed(cfg.rng.torch_seed)

        if not self.debug:
            self.ckpt_dir = create_new_dir(f"{self.exp_dir}/checkpoints")

            self.tb_logger = SummaryWriter(log_dir=f"{self.exp_dir}/summary")

        if cfg.train.ddp:
            self.device = torch.device("cuda:{}".format(rank))
        else:
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{cfg.train.gpu}")
            else:
                print("GPUs not available. Setting device to cpu!!!")
                self.device = torch.device("cpu")

        # Network
        self.network = Network(cfg, device=self.device)
        self.network = self.network.to(self.device)
        self.network_ddp = self.network
        self.num_params = sum(p.numel() for p in self.network.parameters())
        if not cfg.train.debug:
            self.tb_logger.add_scalar("parameters", self.num_params)

        if cfg.train.ddp:
            if self.has_batchnorms():
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.network
                )

            self.network = DDP(self.network, device_ids=[self.rank])
            self.network_ddp = self.network.module


        # Loss 
        self.cls_loss_fn = torch.nn.CrossEntropyLoss()
        if cfg.net.detection:
            self.elm_loss_fn = torch.nn.CrossEntropyLoss()
        

        # Metric Logger
        self.cls_metric_logger = ClassificationMetricLogger(name="elm_types", 
                                                            num_class=cfg.data.n_classes, 
                                                            class_labels=cfg.data.class_types, 
                                                            trace_func=self.trace_func,
                                                           )
        if cfg.net.detection:
            self.det_metric_logger = ClassificationMetricLogger(name="elm_det",
                                                                num_class=2, 
                                                                trace_func=self.trace_func,
                                                            )
        

    def has_batchnorms(self):
        bn_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
        )
        for _, module in self.network.named_modules():
            if isinstance(module, bn_types):
                return True
        return False
    
    def create_dataloader(self, data_files, phase='train', sampler=None, generator=None, shuffle=True, drop_last=False, collate_fn=None):
        
        cfg = self.cfg

        dataset = ELMDataset(cfg.data, data_files, mode='train')

        if (phase=='train')  & cfg.train.ddp:
            shuffle = False
            sampler = DS(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=shuffle,
            generator=generator,
            sampler=sampler,
            pin_memory=True if torch.cuda.is_available() else False,  # pin_memory is slightly faster but cpu pricy
            num_workers=min(cfg.train.num_workers, cpu_count()),
            persistent_workers=True if cfg.train.num_workers > 0 else False,
            drop_last=drop_last,
            collate_fn=collate_fn
        )

        self.trace_func(
            f"{phase} samples: {len(dataset)}, "
            f"{phase} batch size: {dataloader.batch_size}, "
            f"{phase} batches: {len(dataloader)}\n"
        )

        self.trace_func(f"Verifying {phase} dataloader ... ")
        for batch in dataloader:
            # print("data:", batch[0].shape, "labels:", batch[0].shape)
            self.trace_func({k: v.shape for k, v in batch.items() if isinstance(v, torch.Tensor)})
            break

        return dataloader

    def train(self, train_sets=[], val_sets=[], test_sets=[]):

        cfg = self.cfg

        if (not len(val_sets)>0) and cfg.train.monitor in ['val/loss', 'val/acc']:
            raise Exception("cannot monitor {cfg.monitor} when validation samples is {len(val_sets)}")

        ## Prepare datasets
        self.trace_func("Preparing train/test dataset ... ")
        if not len(train_sets)>0:
            train_sets, val_sets = split_data(cfg.data.data_dir,
                                             cfg.data.label_dir,
                                             cfg.data.train_split,
                                             cfg.data.n_folds, 
                                             cfg.data.curr_fold,
                                             cfg.rng.seed
                                            )

        self.trace_func("Creating dataloaders ...")
        self.train_loader = self.create_dataloader(
            train_sets,
            phase='train',
            shuffle=True,
            collate_fn=collate_fn
        )

        if not len(self.train_loader) > 0:
            raise ValueError("Train dataset is empty")

        self.val_loader = self.create_dataloader(
            val_sets,
            phase='val',
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        self.test_loader = self.create_dataloader(
            test_sets,
            phase='test',
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        #  Optimizer
        net_params = [{'params': self.network.parameters()}]
        # net_params.append({'params': self.network.class_head.parameters(), 'lr': 2*cfg.optim.lr})
        self.optimizer = torch.optim.AdamW(net_params, lr=cfg.optim.lr)

        # Lr Schedulers 
        if cfg.optim.lr_scheduler == "reduceonplateau":
            self.lr_scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=cfg.optim.lr_mode,
                factor=cfg.optim.lr_reduce_factor,
                patience=cfg.optim.lr_patience,
                threshold=1e-6,
                threshold_mode='abs',
                verbose=True,
            )

        # Weight scheduler
        if cfg.optim.weight_decay > 0:
            self.wt_scheduler = cosine_scheduler(
                cfg.optim.weight_decay,
                cfg.optim.weight_decay_end,
                cfg.train.epochs,
                len(self.train_loader),
            )

        # Early Stopping
        self.early_stopping = EarlyStopping(
            mode=cfg.train.early_stop_mode,
            patience=cfg.train.early_stop_patience,
            warmup_steps=cfg.train.early_stop_warmup,
            metric_name=cfg.train.early_stop_metric,
            trace_func=self.trace_func,
        )

        # Model Checkpointing
        self.model_checkpointer = ModelCheckpoint(
            self.network,
            self.ckpt_dir,
            ckpt_name=cfg.exp.ckpt_name, # "best_acc",
            monitor=cfg.train.monitor,
            mode=cfg.train.monitor_mode,
            trace_func=self.trace_func,
            debug=self.debug,
        )

        self.global_step = 0
        
        self.train_hist = defaultdict(list)

        if cfg.train.pretrained is not None:
            self.trace_func(f"Loading pretrained network from {cfg.train.pretrained}")
            try:
                self.network_ddp.load_state_dict(
                    torch.load(cfg.train.pretrained), strict=False
                )
                print("Model loaded sucessfully!")
            except Exception as e:
                print(e)
                pass

        self.trace_func(
            f"Training Started (device: {self.device}, " f"rank: {self.rank})"
        )

        start = time.time()
        for epoch in range(cfg.train.epochs):

            if self.rank == 0:
                self.trace_func(f"Epoch {epoch}/{cfg.train.epochs}")

            epoch_start = time.time()

            if cfg.train.ddp and (not cfg.train.single_batch):
                self.train_loader.sampler.set_epoch(epoch)

            epoch_hist = self.train_epoch() # new epoch_history for each epoch

            if self.rank == 0:
                eval_condition = ((epoch + 1) % cfg.train.eval_interval == 0) & (epoch > cfg.train.warmup_epochs-1)
                if eval_condition :
                    self.trace_func("Evaluating")
                    if len(self.val_loader)>0:
                        epoch_hist = self.evaluate_epoch(
                            hist=epoch_hist, 
                            phase="val", 
                            dataloader=self.val_loader, 
                            epoch=epoch, 
                            log_outputs=cfg.train.log_outputs, 
                            log_metrics=cfg.train.log_metrics,
                            )

                    if len(self.test_loader)>0:
                        epoch_hist = self.evaluate_epoch(
                            hist=epoch_hist, 
                            phase='test', 
                            dataloader=self.test_loader, 
                            epoch=epoch,
                            log_outputs=cfg.train.log_outputs, 
                            log_metrics=cfg.train.log_metrics,
                            )
                    
                # Check for nan values
                for k, v in epoch_hist.items():
                    if contains_nan(v):
                        self.logger.exception(f"{k} has nan values")
                        raise Exception(f"{k} has nan values!! {v}")
                        
                # average metrics over batch iterations
                for k, v in epoch_hist.items():
                    self.train_hist[k].append(np.mean(v))
                    if  k==cfg.train.monitor:
                        self.train_hist[f"{k}_best"].append(get_best_val(self.train_hist[k], cfg.train.monitor_mode)[1])
                        
                self.train_hist["etc"].append(time.time() - epoch_start)
                self.train_hist["lr"].append(float(f"{self.optimizer.param_groups[0]['lr']:.6f}"))
                self.train_hist["fold"].append(cfg.data.curr_fold)
                self.train_hist["epoch"].append(epoch)
                
                log_text = ", ".join(
                    [f"{k}:{str(v[-1]):.9s}" for k, v in sorted(self.train_hist.items())]
                )
                if torch.cuda.is_available():
                    free_mem, total_mem = torch.cuda.mem_get_info(self.device)
                    log_text += (
                        f", GPU_used/total:{(total_mem-free_mem)/1024**2:.0f}"
                        f"/{total_mem/1024**2:.0f} Mib"
                    )
                
                # self.trace_func(f"epoch: {epoch}, " f"step:{self.global_step}, " f"{log_text}")

                # Update schedulers/checkpointer/early stopping after evaluation only 
                # TODO: Update even if eval_condition is not met 
                if eval_condition:
                    self.model_checkpointer(epoch_hist[self.model_checkpointer.monitor][-1])
                    self.early_stopping(epoch_hist[self.early_stopping.metric_name][-1])
                    self.lr_scheduler.step(epoch_hist[cfg.optim.lr_monitor][-1])

                if not self.debug:
                    for k, v in self.train_hist.items():
                        self.tb_logger.add_scalar(k, v[-1], epoch)
                    
                    if eval_condition:
                        # Try to syncronize with the model checkpoint, may incorrectly save the latest evaluated metric
                        if self.model_checkpointer.best_model:
                            self.best_epoch = epoch
                            self.save_states(epoch)
                            self.best_cls_results = copy(self.cls_metric_logger.results)
                            if cfg.net.detection:
                                self.best_det_results = copy(self.det_metric_logger.results)
                        
                    with open(self.exp_dir + "/train_hist.txt", "w") as f:
                        json.dump({k: str(v) for k, v in self.train_hist.items()}, f)
            
                if self.early_stopping.early_stop:
                    self.trace_func(f"Early stopping at epoch {epoch}")
                    break

        self.trace_func(f"DONE in {time.time()-start}!")
        self.trace_func(f"Best ELM Classfication Results (epoch {self.best_epoch})", self.best_cls_results)
        if cfg.net.detection:
            self.trace_func(f"Best ELM Detection Results (epoch {self.best_epoch})", self.best_det_results)

        self.tb_logger.flush()
        self.tb_logger.close

                            
    def train_epoch(self, **kwargs):
        
        self.network.train()

        # Container to store epoch results
        hist = defaultdict(list)

        pbar = tqdm(total=len(self.train_loader), position=0, leave=True)
        
        for batch in self.train_loader:

            # set weight decay
            if self.cfg.optim.weight_decay > 0:
                # only the first group is regularized
                self.optimizer.param_groups[0]["weight_decay"] = self.wt_scheduler[
                    self.global_step
                ]
                
            hist, loss, _, _ = self.step(batch, hist, phase='train')

            pbar.update(1)
            postfix = ""
            for k, v in hist.items():
                if "train" in k:
                    # if len(v)>0:
                    postfix += f", {k.split('/')[1]}: {v[-1]:8.3f}"
            pbar.set_postfix_str(postfix)

            if loss.isnan():
                self.trace_func("loss is nan !!")
                raise Exception

            # compute gradients
            loss.backward()

            # clip gradients before step
            if self.cfg.optim.clip_grad_norm>0:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.optim.clip_grad_norm)

            # self.log_gradient_norms()
            
            # update all model parameters
            self.optimizer.step() 

            # zero gradients 
            self.optimizer.zero_grad(set_to_none=True)

            if self.cfg.train.reset_head_steps is not None:
                if self.global_step == self.cfg.train.reset_head_steps:
                    print("RESETTING HEADS")
                    self.head.reset_parameters()

            self.global_step += 1

            if self.cfg.train.ddp:
                torch.cuda.synchronize()
            # break

        pbar.close()

        return hist

    def step(self, batch, hist, phase, **kwargs):

        if phase=='train':
            self.network.train()
        else:
            self.network.eval()

        batch = batch.to(self.device)
        
        cls_preds, elm_preds = self.network(batch.dalpha)

        total_loss= 0
        elm_cls_loss = self.cls_loss_fn(cls_preds, batch.cls_labels)
        hist[f"{phase}/elm_cls_loss"].append(elm_cls_loss.detach().item())
        total_loss += elm_cls_loss
        
        if self.cfg.net.detection:
            elm_det_loss = self.elm_loss_fn(elm_preds, batch.elm_labels)
            hist[f"{phase}/elm_det_loss"].append(elm_det_loss.detach().item())
            total_loss += elm_det_loss

        hist[f"{phase}/loss"].append(total_loss.detach().item())

        if self.global_step%self.cfg.train.log_interval==0 & self.global_step>1:
            self.log_outputs(batch, (cls_preds, elm_preds), self.global_step, phase)
            
        return hist, total_loss, cls_preds, elm_preds

    def plot_history(self, ms=2, figsize=(10, 6)):
        
        train_epochs = np.arange(len(self.train_hist['epoch']))  # 450 epochs
        test_epochs = np.linspace(0, len(self.train_hist['epoch']) - 1, len(self.train_hist['test/loss']))  # 90 test points
        
        # Plot Loss Components
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
        ax.plot(train_epochs, self.train_hist['train/elm_cls_loss'], label=' Train Classification Loss')
        ax.plot(test_epochs, self.train_hist['test/elm_cls_loss'], label='Test Classification Loss')
        if 'train/elm_det_loss' in self.train_hist:
            ax.plot(train_epochs, self.train_hist['train/elm_det_loss'], label='Train Detection Loss')
        if 'test/elm_det_loss' in self.train_hist:
            ax.plot(test_epochs, self.train_hist['test/elm_det_loss'], label='Test Detection Loss')
            
        ax.set_title("Train/test Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid()
        
        plt.show()
            
    def plot_preds(self, inputs, cls_preds, elm_preds, n_images=5, figsize=(8, 6)):
        
        '''
        Visualize class predictions
        Args:
            inputs (DictStruct): Current batch inputs 
            cls_preds (array): ELM type prediction 
            elm_preds (array): ELM label prediction. 
        '''
        
        bs = inputs.dalpha.shape[0] # bs may be different from actual batch size
        
        n_rows = min(n_images, bs)
        
        print(f"{n_rows=}")
        
        plt.close('all')
        fig, axes = plt.subplots(n_rows, 1, figsize=figsize, layout='constrained')

        for i in range(n_rows):
            shot = inputs.files[i].split('.')[0]
            gt_class = inputs.cls_labels[i].cpu().numpy()
            pred_class = cls_preds[i].argmax().cpu().numpy()
            dtime = inputs.dtime[i].reshape(-1).cpu().numpy()
            dalpha = inputs.dalpha[i].reshape(-1).cpu().numpy()
            text_label = f"Shot_{shot}, Type:{gt_class}, Pred:{pred_class}"
            axes[i].plot(dtime, dalpha, label=text_label, color="blue", zorder=1)
        
            if elm_preds is not None:
                axes[i].plot(dtime, inputs.elm_labels[i].cpu().numpy(), label="GT ELM", color="green", alpha=0.6, zorder=5)
                axes[i].plot(dtime, elm_preds[i].argmax(0).cpu().numpy(), label="Pred ELM", color="red", linestyle='--', zorder=10)
                
            axes[i].legend(fontsize=8)
            axes[i].grid()
        
        plt.xlabel("Time Steps")
        plt.show()

        return fig, ax
        
    def log_gradient_norms(self, ):
        """Logs gradient norms for all parameters to TensorBoard.
        """
        if self.debug:
            return
            
        for name, param in self.network.named_parameters():
            if "weight" in name:
                if param.grad is not None: 
                    grad_norm = torch.norm(param.grad).item()
                    self.tb_logger.add_scalar(f"grad_norms/{name}", grad_norm, self.global_step)
                
    def log_hparams(self, final_epoch):
        if self.debug:
            return
            
        metric_dict = {
            f"hparams/{k}": val_list[-1]
            for k, val_list in self.train_hist.items()
            if (("train" in k) or ("val" in k))
        }
        hparams_dict = dictconf_to_dict(self.cfg)
        hparams_dict["hparams/parameters"] = self.num_params
        hparams_dict["hparams/final_epoch"] = final_epoch
        self.tb_logger.add_hparams(
            hparams_dict, metric_dict=metric_dict, run_name="hparams"
        )

    def log_metrics(self, metric_logger, epoch, phase):
        if self.debug:
            return
        # try:            
        for k, v in metric_logger.results.items():
            self.tb_logger.add_scalar(f"{metric_logger.name}_metrics/{phase}/{k}", v, epoch)
                            
        metric_logger.plot_confusion_matrix()
        # self.add_plot_to_tb(f"confusion_matrix/phase", epoch)
        self.tb_logger.add_figure(f"{metric_logger.name}/confusion_matrix/{phase}", figure=plt.gcf(), global_step=epoch, close=True)

        if metric_logger.num_class==2:
            metric_logger.plot_precision_recall()
            # self.add_plot_to_tb(f"precision_recall/{phase}", epoch)
            self.tb_logger.add_figure(f"{metric_logger.name}/precision_recall/{phase}", figure=plt.gcf(), global_step=epoch, close=True)

            metric_logger.plot_roc_curve()
            # self.add_plot_to_tb(f"roc_curve/{phase}", epoch)
            self.tb_logger.add_figure(f"{metric_logger.name}/roc_curve/{phase}", figure=plt.gcf(), global_step=epoch, close=True)

            self.tb_logger.add_scalar(f"{metric_logger.name}/roc_auc/{phase}", torch.tensor(metric_logger.roc_auc_score), epoch)
            
        # except Exception as e:
        #     print(e)
        
    def evaluate_epoch(self, dataloader, phase="val", hist=None, epoch=0, return_preds=False, log_outputs=False, log_metrics=False):

        self.network.eval()
        
        self.cls_metric_logger.reset()
        if self.cfg.net.detection:
            self.det_metric_logger.reset()
        
        if hist is None:
            hist = defaultdict(list)
        
        pbar = tqdm(total=len(dataloader), position=0)
        with torch.no_grad():
            for batch in dataloader:
                pbar.update(1)

                hist, val_loss, cls_preds, elm_preds = self.step(batch, 
                                                                 hist, 
                                                                 phase=phase, 
                                                                 return_preds=return_preds,
                                                                )
                
                cls_acc = self.cls_metric_logger(cls_preds, 
                                                 batch.cls_labels, 
                                                 files=batch.files,
                                                )
                # TODO! Taking mean of acc might be different from overal acc 
                # due to different number of samples in last batch
                hist[f"{phase}/cls_acc"].extend(cls_acc.tolist())
                postfix = f"{phase}_loss: {np.mean(hist[f'{phase}/loss']):.3f}, \
                            {phase}_cls_acc: {np.mean(hist[f'{phase}/cls_acc']):.3f}"

                if self.cfg.net.detection:
                    det_acc = self.det_metric_logger(F.softmax(elm_preds.permute(0, 2, 1).flatten(0, 1), dim=-1), 
                                                    batch.elm_labels.flatten(), 
                                                    files=batch.files,
                                                    )
                    hist[f"{phase}/det_acc"].extend(det_acc.tolist())
                    postfix += f", {phase}_det_acc: {np.mean(hist[f'{phase}/det_acc']):.3f}"
                
                pbar.set_postfix_str(postfix)

                if return_preds:
                    hist[f"{phase}/inputs"].append(batch)
                    hist[f"{phase}/cls_preds"].append(cls_preds)
                    hist[f"{phase}/elm_preds"].append(elm_preds)

        if log_outputs:
            print("Logging predictions ...")
            fig, ax = self.plot_preds(batch, cls_preds, elm_preds, epoch, phase)
            self.tb_logger.add_figure(f"preds/{phase}", fig, epoch, close=True)
            
        self.cls_metric_logger.compute()
        if self.cfg.net.detection:
            self.det_metric_logger.compute()

        if log_metrics:
            self.log_metrics(self.cls_metric_logger, epoch, phase)
            if self.cfg.net.detection:
                self.log_metrics(self.det_metric_logger, epoch, phase)

            
        return hist

    def evaluate(self, data_sets, ckpt_name='best_accuracy', return_preds=False):

        dataloader = self.create_dataloader(
            test_sets,
            phase='test',
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        self.trace_func("Loading Checkpoint ... ")
            
        self.model_checkpointer.load_checkpoint()
        
        h = self.evaluate_epoch(dataloader, return_preds=return_preds)
            
        return h
        
    def save_states(self, epoch=0, ckpt_dir=None, ckpt_name="model_states"):
        state = {
            "last_epoch": epoch,
            "global_step": self.global_step,
            "model_state": self.network_ddp.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.lr_scheduler.state_dict(),
        }
        torch.save(state, os.path.join(ckpt_dir or self.ckpt_dir, ckpt_name + ".pth"))

    def load_states(self, ckpt_dir=None, ckpt_name="model_states"):

        checkpoint = torch.load(
            os.path.join(ckpt_dir or self.ckpt_dir, ckpt_name + ".pth"), map_location=self.device
        )

        self.global_step = checkpoint["global_step"]
        self.network_ddp.load_state_dict(checkpoint["model_state"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state"])

        return checkpoint["last_epoch"]
