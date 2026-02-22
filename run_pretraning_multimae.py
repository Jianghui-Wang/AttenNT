# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv and MAE code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# --------------------------------------------------------
import argparse
import datetime
import json
import math
import os
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import yaml
from einops import rearrange
from torch.utils.data import DataLoader, Subset, Dataset

import utils
from utils import task_balancing
import utils.data_constants as data_constants
from multimae import multimae
from multimae.criterion import (MaskedCrossEntropyLoss, MaskedL1Loss,
                                MaskedMSELoss)
from multimae.input_adapters import PatchedInputAdapter, SemSegInputAdapter
from multimae.output_adapters import SpatialOutputAdapter
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import create_model
from utils.data_constants import COCO_SEMSEG_NUM_CLASSES
from utils.datasets import build_multimae_pretraining_dataset
from utils.optim_factory import create_optimizer
from utils.task_balancing import (NoWeightingStrategy,
                                  UncertaintyWeightingStrategy,
                                  SoftmaxWeightingStrategy)

class AtteNTComplex:
    """    
    This algorithm performs selective retraining based on sample losses:
    1. Forward pass on all data to compute losses
    2. Select subset based on losses using specified strategy
    3. Retrain selected samples for interval epochs
    4. Update and repeat
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        tasks_loss_fn: Dict[str, nn.Module],
        loss_balancer: nn.Module,
        device: torch.device = None,
        ratio_schedule: Literal['cosine', 'incremental'] = 'cosine',
        interval_schedule: Literal['fixed', 'incremental'] = 'fixed',
        selection_strategy: Literal['random', 'hard', 'soft'] = 'hard',
        initial_ratio: float = 0.2,
        final_ratio: float = 0.8,
        initial_interval: int = 5,
        max_interval: int = 10,
        total_cycles: int = 10,
        gumbel_temperature: float = 1.0,
        soft_normalization: Literal['sigmoid', 'softmax'] = 'softmax',
        num_encoded_tokens: int = 196,
        in_domains: List[str] = [],
        loss_on_unmasked: bool = True,
        alphas: float = 1.0,
        sample_tasks_uniformly: bool = False,
        standardize_depth: bool = True,
        extra_norm_pix_loss: bool = False,
        fp32_output_adapters: List[str] = [],
        loss_scaler = None,
        max_norm: float = None,
        max_skip_norm: float = None,
        log_writer = None,
        lr_schedule_values = None,
        wd_schedule_values = None,
        start_epoch: int = 0,
        save_ckpt_freq: int = 20,
        output_dir: str = '',
        args = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.tasks_loss_fn = tasks_loss_fn
        self.loss_balancer = loss_balancer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.ratio_schedule = ratio_schedule
        self.interval_schedule = interval_schedule
        self.selection_strategy = selection_strategy
        
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.initial_interval = initial_interval
        self.max_interval = max_interval
        self.total_cycles = total_cycles
        self.gumbel_temperature = gumbel_temperature
        self.soft_normalization = soft_normalization
        
        self.num_encoded_tokens = num_encoded_tokens
        self.in_domains = in_domains
        self.loss_on_unmasked = loss_on_unmasked
        self.alphas = alphas
        self.sample_tasks_uniformly = sample_tasks_uniformly
        self.standardize_depth = standardize_depth
        self.extra_norm_pix_loss = extra_norm_pix_loss
        self.fp32_output_adapters = fp32_output_adapters
        self.loss_scaler = loss_scaler
        self.max_norm = max_norm
        self.max_skip_norm = max_skip_norm
        self.log_writer = log_writer
        self.lr_schedule_values = lr_schedule_values
        self.wd_schedule_values = wd_schedule_values
        self.start_epoch = start_epoch
        self.save_ckpt_freq = save_ckpt_freq
        self.output_dir = output_dir
        self.args = args
        
        self.dataset = train_loader.dataset
        self.batch_size = train_loader.batch_size
        self.num_samples = len(self.dataset)
    
        self.model.to(self.device)
        
    def get_ratio(self, cycle: int) -> float:
        """Calculate ratio for current cycle based on schedule"""
        progress = cycle / (self.total_cycles - 1) if self.total_cycles > 1 else 1.0
        
        if self.ratio_schedule == 'cosine':
            cosine_val = (1 - math.cos(progress * math.pi)) / 2
            ratio = self.initial_ratio + (self.final_ratio - self.initial_ratio) * cosine_val
        else:  # incremental (linear)
            ratio = self.initial_ratio + (self.final_ratio - self.initial_ratio) * progress
            
        return min(max(ratio, self.initial_ratio), self.final_ratio)
    
    def get_interval(self, cycle: int) -> int:
        """Calculate interval for current cycle based on schedule"""
        if self.interval_schedule == 'fixed':
            return self.initial_interval
        else:  # incremental
            progress = cycle / (self.total_cycles - 1) if self.total_cycles > 1 else 1.0
            interval = self.initial_interval + int((self.max_interval - self.initial_interval) * progress)
            return interval
    
    @torch.no_grad()
    def compute_all_losses(self) -> Tuple[torch.Tensor, List[int]]:
        """Compute losses for all samples in the dataset"""
        self.model.eval()
        all_losses = []
        all_indices = []
        
        indexed_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.train_loader.num_workers if hasattr(self.train_loader, 'num_workers') else 0
        )
        
        for batch_idx, (x, _) in enumerate(indexed_loader):
            tasks_dict = {
                task: tensor.to(self.device, non_blocking=True)
                for task, tensor in x.items()
            }
            
            if self.standardize_depth and 'depth' in tasks_dict:
                trunc_depth = torch.sort(rearrange(tasks_dict['depth'], 'b c h w -> b (c h w)'), dim=1)[0]
                trunc_depth = trunc_depth[:,int(0.1 * trunc_depth.shape[1]): int(0.9 * trunc_depth.shape[1])]
                tasks_dict['depth'] = (tasks_dict['depth'] - trunc_depth.mean(dim=1)[:,None,None,None]) / torch.sqrt(trunc_depth.var(dim=1)[:,None,None,None] + 1e-6)
            
            input_dict = {
                task: tensor
                for task, tensor in tasks_dict.items()
                if task in self.in_domains
            }
            
            with torch.cuda.amp.autocast():
                preds, masks = self.model(
                    input_dict,
                    num_encoded_tokens=self.num_encoded_tokens,
                    alphas=self.alphas,
                    sample_tasks_uniformly=self.sample_tasks_uniformly,
                    fp32_output_adapters=self.fp32_output_adapters
                )
                
                if self.extra_norm_pix_loss:
                    tasks_dict['norm_rgb'] = tasks_dict['rgb']
                    masks['norm_rgb'] = masks.get('rgb', None)
                
                batch_losses = []
                for task in preds:
                    target = tasks_dict[task]
                    if self.loss_on_unmasked:
                        task_loss = self.tasks_loss_fn[task](preds[task].float(), target)
                    else:
                        task_loss = self.tasks_loss_fn[task](preds[task].float(), target, mask=masks.get(task, None))
                    
                    if task_loss.dim() > 1:
                        task_loss = task_loss.mean(dim=list(range(1, task_loss.dim())))
                    batch_losses.append(task_loss)
                
                total_batch_loss = sum(batch_losses)
                all_losses.append(total_batch_loss.cpu())
            
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + len(input_dict[self.in_domains[0]]), self.num_samples)
            all_indices.extend(range(start_idx, end_idx))
        
        all_losses = torch.cat(all_losses)
        return all_losses, all_indices
    
    def gumbel_top_k(self, scores: torch.Tensor, k: int, temperature: float = 1.0) -> torch.Tensor:
        """Gumbel-Top-K selection algorithm"""
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
        perturbed_scores = (torch.log(scores + 1e-10) + gumbel_noise) / temperature
        _, indices = torch.topk(perturbed_scores, k)
        return indices
    
    def select_samples(self, losses: torch.Tensor, ratio: float) -> List[int]:
        """Select samples based on losses and selection strategy"""
        num_select = int(self.num_samples * ratio)
        num_select = max(1, min(num_select, self.num_samples))
        
        if self.selection_strategy == 'random':
            indices = torch.randperm(self.num_samples)[:num_select].tolist()
        elif self.selection_strategy == 'hard':
            _, indices = torch.topk(losses, num_select)
            indices = indices.tolist()
        else:  
            if self.soft_normalization == 'sigmoid':
                probs = torch.sigmoid(losses)
            else: 
                probs = F.softmax(losses, dim=0)
            indices = self.gumbel_top_k(probs, num_select, self.gumbel_temperature)
            indices = indices.tolist()
        
        return indices
    
    def train_on_subset(self, indices: List[int], num_epochs: int, cycle: int):
        subset_dataset = Subset(self.dataset, indices)
        subset_loader = DataLoader(
            subset_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.train_loader.num_workers if hasattr(self.train_loader, 'num_workers') else 0
        )
        
        for interval_epoch in range(num_epochs):
            if self.selection_strategy == 'soft' and interval_epoch > 0:
                losses, _ = self.compute_all_losses()
                indices = self.select_samples(losses, len(indices) / self.num_samples)
                subset_dataset = Subset(self.dataset, indices)
                subset_loader = DataLoader(
                    subset_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.train_loader.num_workers if hasattr(self.train_loader, 'num_workers') else 0
                )
        
            train_stats = self._train_one_epoch_subset(
                subset_loader, 
                f"Cycle {cycle+1} - Subset Epoch {interval_epoch+1}/{num_epochs}"
            )
            
            if self.log_writer is not None:
                self.log_writer.update({**{k: v for k, v in train_stats.items()}})
    
    def _train_one_epoch_subset(self, data_loader: DataLoader, header: str):
        """Train one epoch on a subset"""
        self.model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        print_freq = 10
        
        for step, (x, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            # Update learning rate and weight decay
            global_step = self.current_step
            if self.lr_schedule_values is not None or self.wd_schedule_values is not None:
                for param_group in self.optimizer.param_groups:
                    if self.lr_schedule_values is not None and global_step < len(self.lr_schedule_values):
                        param_group["lr"] = self.lr_schedule_values[global_step] * param_group.get("lr_scale", 1.0)
                    if self.wd_schedule_values is not None and param_group["weight_decay"] > 0 and global_step < len(self.wd_schedule_values):
                        param_group["weight_decay"] = self.wd_schedule_values[global_step]
            
            tasks_dict = {
                task: tensor.to(self.device, non_blocking=True)
                for task, tensor in x.items()
            }
            
            # Truncated depth standardization
            if self.standardize_depth and 'depth' in tasks_dict:
                trunc_depth = torch.sort(rearrange(tasks_dict['depth'], 'b c h w -> b (c h w)'), dim=1)[0]
                trunc_depth = trunc_depth[:,int(0.1 * trunc_depth.shape[1]): int(0.9 * trunc_depth.shape[1])]
                tasks_dict['depth'] = (tasks_dict['depth'] - trunc_depth.mean(dim=1)[:,None,None,None]) / torch.sqrt(trunc_depth.var(dim=1)[:,None,None,None] + 1e-6)
            
            input_dict = {
                task: tensor
                for task, tensor in tasks_dict.items()
                if task in self.in_domains
            }
            
            with torch.cuda.amp.autocast():
                preds, masks = self.model(
                    input_dict,
                    num_encoded_tokens=self.num_encoded_tokens,
                    alphas=self.alphas,
                    sample_tasks_uniformly=self.sample_tasks_uniformly,
                    fp32_output_adapters=self.fp32_output_adapters
                )
                
                if self.extra_norm_pix_loss:
                    tasks_dict['norm_rgb'] = tasks_dict['rgb']
                    masks['norm_rgb'] = masks.get('rgb', None)
                
                task_losses = {}
                for task in preds:
                    target = tasks_dict[task]
                    if self.loss_on_unmasked:
                        task_losses[task] = self.tasks_loss_fn[task](preds[task].float(), target)
                    else:
                        task_losses[task] = self.tasks_loss_fn[task](preds[task].float(), target, mask=masks.get(task, None))
                
                weighted_task_losses = self.loss_balancer(task_losses)
                loss = sum(weighted_task_losses.values())
            
            loss_value = sum(task_losses.values()).item()
            
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            
            self.optimizer.zero_grad()
            is_second_order = hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order
            grad_norm = self.loss_scaler(loss, self.optimizer, clip_grad=self.max_norm, skip_grad=self.max_skip_norm,
                                        parameters=self.model.parameters(), create_graph=is_second_order)
            
            torch.cuda.synchronize()
            
            metric_logger.update(loss=loss_value)
            for task, l in task_losses.items():
                metric_logger.update(**{f'{task}_loss': l.item()})
            
            self.current_step += 1
        
        metric_logger.synchronize_between_processes()
        return {'[Subset] ' + k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    def run(self, epochs: int):
        """Execute the AtteNT Complex algorithm"""
        print(f"Starting AtteNT Complex Training")
        print(f"Device: {self.device}")
        print(f"Ratio Schedule: {self.ratio_schedule}")
        print(f"Interval Schedule: {self.interval_schedule}")
        print(f"Selection Strategy: {self.selection_strategy}")
        print("-" * 50)
        
        self.current_step = 0
        num_steps_per_epoch = len(self.train_loader)
        
        print("Initial full dataset training...")
        epoch = self.start_epoch
        train_stats = train_one_epoch(
            model=self.model,
            data_loader=self.train_loader,
            tasks_loss_fn=self.tasks_loss_fn,
            loss_balancer=self.loss_balancer,
            optimizer=self.optimizer,
            device=self.device,
            epoch=epoch,
            loss_scaler=self.loss_scaler,
            max_norm=self.max_norm,
            max_skip_norm=self.max_skip_norm,
            log_writer=self.log_writer,
            start_steps=epoch * num_steps_per_epoch,
            lr_schedule_values=self.lr_schedule_values,
            wd_schedule_values=self.wd_schedule_values,
            num_encoded_tokens=self.num_encoded_tokens,
            in_domains=self.in_domains,
            loss_on_unmasked=self.loss_on_unmasked,
            alphas=self.alphas,
            sample_tasks_uniformly=self.sample_tasks_uniformly,
            standardize_depth=self.standardize_depth,
            extra_norm_pix_loss=self.extra_norm_pix_loss,
            fp32_output_adapters=self.fp32_output_adapters
        )
        
        epoch += 1
        self.current_step = epoch * num_steps_per_epoch
        
        epochs_per_cycle = (epochs - epoch) // self.total_cycles
        
        for cycle in range(self.total_cycles):
            print(f"\nCycle {cycle + 1}/{self.total_cycles}")
            
            ratio = self.get_ratio(cycle)
            interval = self.get_interval(cycle)
            print(f"Ratio: {ratio:.3f}, Interval: {interval}")
            
            print("Computing losses for all samples...")
            losses, indices = self.compute_all_losses()
            
            selected_indices = self.select_samples(losses, ratio)
            print(f"Selected {len(selected_indices)} samples out of {self.num_samples}")
            
            print(f"Training on selected subset for {interval} epochs...")
            self.train_on_subset(selected_indices, interval, cycle)
            
            remaining_epochs = epochs_per_cycle - interval
            for _ in range(remaining_epochs):
                print(f"Full dataset training - Epoch {epoch}/{epochs}")
                train_stats = train_one_epoch(
                    model=self.model,
                    data_loader=self.train_loader,
                    tasks_loss_fn=self.tasks_loss_fn,
                    loss_balancer=self.loss_balancer,
                    optimizer=self.optimizer,
                    device=self.device,
                    epoch=epoch,
                    loss_scaler=self.loss_scaler,
                    max_norm=self.max_norm,
                    max_skip_norm=self.max_skip_norm,
                    log_writer=self.log_writer,
                    start_steps=self.current_step,
                    lr_schedule_values=self.lr_schedule_values,
                    wd_schedule_values=self.wd_schedule_values,
                    num_encoded_tokens=self.num_encoded_tokens,
                    in_domains=self.in_domains,
                    loss_on_unmasked=self.loss_on_unmasked,
                    alphas=self.alphas,
                    sample_tasks_uniformly=self.sample_tasks_uniformly,
                    standardize_depth=self.standardize_depth,
                    extra_norm_pix_loss=self.extra_norm_pix_loss,
                    fp32_output_adapters=self.fp32_output_adapters
                )
                
                if self.output_dir and ((epoch + 1) % self.save_ckpt_freq == 0 or epoch + 1 == epochs):
                    utils.save_model(
                        args=self.args, model=self.model, model_without_ddp=self.model.module if hasattr(self.model, 'module') else self.model,
                        optimizer=self.optimizer, loss_scaler=self.loss_scaler,
                        loss_balancer=self.loss_balancer.module if hasattr(self.loss_balancer, 'module') else self.loss_balancer,
                        epoch=epoch
                    )
                
                epoch += 1
                self.current_step = epoch * num_steps_per_epoch
        
        print("\nAtteNT Complex Training Completed!")

DOMAIN_CONF = {
    'rgb': {
        'channels': 3,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=3),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=3),
        'loss': MaskedMSELoss,
    },
    'depth': {
        'channels': 1,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=1),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=1),
        'loss': MaskedL1Loss,
    },
    'semseg': {
        'num_classes': 133,
        'stride_level': 4,
        'input_adapter': partial(SemSegInputAdapter,
                                num_classes=COCO_SEMSEG_NUM_CLASSES,
                                dim_class_emb=64, interpolate_class_emb=False),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=COCO_SEMSEG_NUM_CLASSES),
        'loss': partial(MaskedCrossEntropyLoss, label_smoothing=0.0),
    },
}


def get_args():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser('MultiMAE pre-training script', add_help=False)

    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (default: %(default)s)')
    parser.add_argument('--epochs', default=1600, type=int,
                        help='Number of epochs (default: %(default)s)')
    parser.add_argument('--save_ckpt_freq', default=20, type=int,
                        help='Checkpoint saving frequency in epochs (default: %(default)s)')

    # AtteNT Complex parameters
    parser.add_argument('--use_attent_complex', action='store_true',
                        help='Use AtteNT Complex training algorithm')
    parser.add_argument('--attent_ratio_schedule', default='cosine', type=str,
                        choices=['cosine', 'incremental'],
                        help='Ratio schedule for AtteNT (default: %(default)s)')
    parser.add_argument('--attent_interval_schedule', default='fixed', type=str,
                        choices=['fixed', 'incremental'],
                        help='Interval schedule for AtteNT (default: %(default)s)')
    parser.add_argument('--attent_selection_strategy', default='hard', type=str,
                        choices=['random', 'hard', 'soft'],
                        help='Selection strategy for AtteNT (default: %(default)s)')
    parser.add_argument('--attent_initial_ratio', default=0.2, type=float,
                        help='Initial ratio for AtteNT (default: %(default)s)')
    parser.add_argument('--attent_final_ratio', default=0.8, type=float,
                        help='Final ratio for AtteNT (default: %(default)s)')
    parser.add_argument('--attent_initial_interval', default=5, type=int,
                        help='Initial interval for AtteNT (default: %(default)s)')
    parser.add_argument('--attent_max_interval', default=10, type=int,
                        help='Max interval for AtteNT incremental schedule (default: %(default)s)')
    parser.add_argument('--attent_total_cycles', default=10, type=int,
                        help='Total cycles for AtteNT (default: %(default)s)')
    parser.add_argument('--attent_gumbel_temperature', default=1.0, type=float,
                        help='Gumbel temperature for soft selection (default: %(default)s)')
    parser.add_argument('--attent_soft_normalization', default='softmax', type=str,
                        choices=['sigmoid', 'softmax'],
                        help='Normalization for soft selection (default: %(default)s)')

    # Task parameters
    parser.add_argument('--in_domains', default='rgb-depth-semseg', type=str,
                        help='Input domain names, separated by hyphen (default: %(default)s)')
    parser.add_argument('--out_domains', default='rgb-depth-semseg', type=str,
                        help='Output domain names, separated by hyphen (default: %(default)s)')
    parser.add_argument('--standardize_depth', action='store_true')
    parser.add_argument('--no_standardize_depth', action='store_false', dest='standardize_depth')
    parser.set_defaults(standardize_depth=False)
    parser.add_argument('--extra_norm_pix_loss', action='store_true')
    parser.add_argument('--no_extra_norm_pix_loss', action='store_false', dest='extra_norm_pix_loss')
    parser.set_defaults(extra_norm_pix_loss=True)

    # Model parameters
    parser.add_argument('--model', default='pretrain_multimae_base', type=str, metavar='MODEL',
                        help='Name of model to train (default: %(default)s)')
    parser.add_argument('--num_encoded_tokens', default=98, type=int,
                        help='Number of tokens to randomly choose for encoder (default: %(default)s)')
    parser.add_argument('--num_global_tokens', default=1, type=int,
                        help='Number of global tokens to add to encoder (default: %(default)s)')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Base patch size for image-like modalities (default: %(default)s)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='Images input size for backbone (default: %(default)s)')
    parser.add_argument('--alphas', type=float, default=1.0,
                        help='Dirichlet alphas concentration parameter (default: %(default)s)')
    parser.add_argument('--sample_tasks_uniformly', default=False, action='store_true',
                        help='Set to True/False to enable/disable uniform sampling over tasks to sample masks for.')

    parser.add_argument('--decoder_use_task_queries', default=True, action='store_true',
                        help='Set to True/False to enable/disable adding of task-specific tokens to decoder query tokens')
    parser.add_argument('--decoder_use_xattn', default=True, action='store_true',
                        help='Set to True/False to enable/disable decoder cross attention.')
    parser.add_argument('--decoder_dim', default=256, type=int,
                        help='Token dimension inside the decoder layers (default: %(default)s)')
    parser.add_argument('--decoder_depth', default=2, type=int,
                        help='Number of self-attention layers after the initial cross attention (default: %(default)s)')
    parser.add_argument('--decoder_num_heads', default=8, type=int,
                        help='Number of attention heads in decoder (default: %(default)s)')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: %(default)s)')

    parser.add_argument('--loss_on_unmasked', default=False, action='store_true',
                        help='Set to True/False to enable/disable computing the loss on non-masked tokens')
    parser.add_argument('--no_loss_on_unmasked', action='store_false', dest='loss_on_unmasked')
    parser.set_defaults(loss_on_unmasked=False)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer epsilon (default: %(default)s)')
    parser.add_argument('--opt_betas', default=[0.9, 0.95], type=float, nargs='+', metavar='BETA',
                        help='Optimizer betas (default: %(default)s)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='CLIPNORM',
                        help='Clip gradient norm (default: %(default)s)')
    parser.add_argument('--skip_grad', type=float, default=None, metavar='SKIPNORM',
                        help='Skip update if gradient norm larger than threshold (default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: %(default)s)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay (default: %(default)s)')
    parser.add_argument('--weight_decay_end', type=float, default=None,
                        help='Final value of the weight decay')
    parser.add_argument('--decoder_decay', type=float, default=None,
                        help='decoder weight decay')

    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='Base learning rate (default: %(default)s)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='Warmup learning rate (default: %(default)s)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Lower lr bound (default: %(default)s)')
    parser.add_argument('--task_balancer', type=str, default='none',
                        help='Task balancing scheme (default: %(default)s)')
    parser.add_argument('--balancer_lr_scale', type=float, default=1.0,
                        help='Task loss balancer LR scale (default: %(default)s)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='Epochs to warmup LR (default: %(default)s)')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='Steps to warmup LR (default: %(default)s)')

    parser.add_argument('--fp32_output_adapters', type=str, default='',
                        help='Tasks output adapters to compute in fp32 mode')

    # Augmentation parameters
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Probability of horizontal flip (default: %(default)s)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (default: %(default)s)')

    # Dataset parameters
    parser.add_argument('--data_path', default=data_constants.IMAGENET_TRAIN_PATH, type=str,
                        help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')

    # Misc.
    parser.add_argument('--output_dir', default='',
                        help='Path where to save')
    parser.add_argument('--device', default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--no_find_unused_params', action='store_false', dest='find_unused_params')
    parser.set_defaults(find_unused_params=True)

    # Wandb logging
    parser.add_argument('--log_wandb', default=False, action='store_true')
    parser.add_argument('--no_log_wandb', action='store_false', dest='log_wandb')
    parser.set_defaults(log_wandb=False)
    parser.add_argument('--wandb_project', default=None, type=str)
    parser.add_argument('--wandb_entity', default=None, type=str)
    parser.add_argument('--wandb_run_name', default=None, type=str)
    parser.add_argument('--show_user_warnings', default=False, action='store_true')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    # Parse config file if provided
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    return args


def get_model(args):
    """Creates and returns model from arguments"""
    print(f"Creating model: {args.model} for inputs {args.in_domains} and outputs {args.out_domains}")

    input_adapters = {
        domain: DOMAIN_CONF[domain]['input_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=args.patch_size,
        )
        for domain in args.in_domains
    }

    output_adapters = {
        domain: DOMAIN_CONF[domain]['output_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=args.patch_size,
            dim_tokens=args.decoder_dim,
            depth=args.decoder_depth,
            num_heads=args.decoder_num_heads,
            use_task_queries=args.decoder_use_task_queries,
            task=domain,
            context_tasks=list(args.in_domains),
            use_xattn=args.decoder_use_xattn
        )
        for domain in args.out_domains
    }

    if args.extra_norm_pix_loss:
        output_adapters['norm_rgb'] = DOMAIN_CONF['rgb']['output_adapter'](
            stride_level=DOMAIN_CONF['rgb']['stride_level'],
            patch_size_full=args.patch_size,
            dim_tokens=args.decoder_dim,
            depth=args.decoder_depth,
            num_heads=args.decoder_num_heads,
            use_task_queries=args.decoder_use_task_queries,
            task='rgb',
            context_tasks=list(args.in_domains),
            use_xattn=args.decoder_use_xattn
        )

    model = create_model(
        args.model,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        num_global_tokens=args.num_global_tokens,
        drop_path_rate=args.drop_path
    )

    return model


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if not args.show_user_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)

    args.in_domains = args.in_domains.split('-')
    args.out_domains = args.out_domains.split('-')
    args.all_domains = list(set(args.in_domains) | set(args.out_domains))

    model = get_model(args)

    if args.task_balancer == 'uncertainty':
        loss_balancer = UncertaintyWeightingStrategy(tasks=args.out_domains)
    elif args.task_balancer == 'softmax':
        loss_balancer = SoftmaxWeightingStrategy(tasks=args.out_domains)
    else:
        loss_balancer = NoWeightingStrategy()

    tasks_loss_fn = {
        domain: DOMAIN_CONF[domain]['loss'](patch_size=args.patch_size, stride=DOMAIN_CONF[domain]['stride_level'])
        for domain in args.out_domains
    }

    if args.extra_norm_pix_loss:
        tasks_loss_fn['norm_rgb'] = DOMAIN_CONF['rgb']['loss'](
            patch_size=args.patch_size,
            stride=DOMAIN_CONF['rgb']['stride_level'],
            norm_pix=True
        )

    dataset_train = build_multimae_pretraining_dataset(args)

    if True:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True, drop_last=True,
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_wandb:
        log_writer = utils.WandbLogger(args)
    else:
        log_writer = None

    print(args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    model.to(device)
    loss_balancer.to(device)
    model_without_ddp = model
    loss_balancer_without_ddp = loss_balancer
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model = %s" % str(model_without_ddp))
    print(f"Number of params: {n_parameters / 1e6} M")

    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = args.blr * total_batch_size / 256

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params
        )
        model_without_ddp = model.module

    task_balancer_need_grad = ['uncertainty']
    if args.distributed and (args.task_balancer in task_balancer_need_grad):
        loss_balancer = torch.nn.parallel.DistributedDataParallel(loss_balancer, device_ids=[args.gpu])
        loss_balancer_without_ddp = loss_balancer.module

    optimizer = create_optimizer(
        args, {'model': model_without_ddp, 'balancer': loss_balancer_without_ddp}
    )
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch
    )
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler
    )

    # Check if using AtteNT Complex algorithm
    if args.use_attent_complex:
        print("Using AtteNT Complex Training Algorithm")
        attent_trainer = AtteNTComplex(
            model=model,
            train_loader=data_loader_train,
            optimizer=optimizer,
            tasks_loss_fn=tasks_loss_fn,
            loss_balancer=loss_balancer,
            device=device,
            ratio_schedule=args.attent_ratio_schedule,
            interval_schedule=args.attent_interval_schedule,
            selection_strategy=args.attent_selection_strategy,
            initial_ratio=args.attent_initial_ratio,
            final_ratio=args.attent_final_ratio,
            initial_interval=args.attent_initial_interval,
            max_interval=args.attent_max_interval,
            total_cycles=args.attent_total_cycles,
            gumbel_temperature=args.attent_gumbel_temperature,
            soft_normalization=args.attent_soft_normalization,
            num_encoded_tokens=args.num_encoded_tokens,
            in_domains=args.in_domains,
            loss_on_unmasked=args.loss_on_unmasked,
            alphas=args.alphas,
            sample_tasks_uniformly=args.sample_tasks_uniformly,
            standardize_depth=args.standardize_depth,
            extra_norm_pix_loss=args.extra_norm_pix_loss,
            fp32_output_adapters=args.fp32_output_adapters.split('-'),
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
            max_skip_norm=args.skip_grad,
            log_writer=log_writer,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            start_epoch=args.start_epoch,
            save_ckpt_freq=args.save_ckpt_freq,
            output_dir=args.output_dir,
            args=args
        )
        attent_trainer.run(args.epochs)
    else:
        # Original training loop
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            if log_writer is not None:
                log_writer.set_step(epoch * num_training_steps_per_epoch)
            
            train_stats = train_one_epoch(
                model=model,
                data_loader=data_loader_train,
                tasks_loss_fn=tasks_loss_fn,
                loss_balancer=loss_balancer,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                max_norm=args.clip_grad,
                max_skip_norm=args.skip_grad,
                log_writer=log_writer,
                start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values,
                wd_schedule_values=wd_schedule_values,
                num_encoded_tokens=args.num_encoded_tokens,
                in_domains=args.in_domains,
                loss_on_unmasked=args.loss_on_unmasked,
                alphas=args.alphas,
                sample_tasks_uniformly=args.sample_tasks_uniformly,
                standardize_depth=args.standardize_depth,
                extra_norm_pix_loss=args.extra_norm_pix_loss,
                fp32_output_adapters=args.fp32_output_adapters.split('-')
            )

            if log_writer is not None:
                log_writer.update({**{k: v for k, v in train_stats.items()}, 'epoch': epoch})
            if args.output_dir:
                if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler,
                        loss_balancer=loss_balancer_without_ddp, epoch=epoch
                    )

            log_stats = {**{k: v for k, v in train_stats.items()},
                        'epoch': epoch, 'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, 
                   tasks_loss_fn: Dict[str, torch.nn.Module],
                   loss_balancer: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   device: torch.device, epoch: int, loss_scaler, 
                   max_norm: float = None, max_skip_norm: float = None,
                   log_writer=None, lr_scheduler=None, start_steps=None, 
                   lr_schedule_values=None, wd_schedule_values=None,
                   num_encoded_tokens: int = 196, in_domains: List[str] = [],
                   loss_on_unmasked: bool = True, alphas: float = 1.0,
                   sample_tasks_uniformly: bool = False, standardize_depth: bool = True,
                   extra_norm_pix_loss: bool = False, fp32_output_adapters: List[str] = []):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for step, (x, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = start_steps + step
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in x.items()
        }

        if standardize_depth and 'depth' in tasks_dict:
            trunc_depth = torch.sort(rearrange(tasks_dict['depth'], 'b c h w -> b (c h w)'), dim=1)[0]
            trunc_depth = trunc_depth[:,int(0.1 * trunc_depth.shape[1]): int(0.9 * trunc_depth.shape[1])]
            tasks_dict['depth'] = (tasks_dict['depth'] - trunc_depth.mean(dim=1)[:,None,None,None]) / torch.sqrt(trunc_depth.var(dim=1)[:,None,None,None] + 1e-6)

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in in_domains
        }

        with torch.cuda.amp.autocast():
            preds, masks = model(
                input_dict,
                num_encoded_tokens=num_encoded_tokens,
                alphas=alphas,
                sample_tasks_uniformly=sample_tasks_uniformly,
                fp32_output_adapters=fp32_output_adapters
            )

            if extra_norm_pix_loss:
                tasks_dict['norm_rgb'] = tasks_dict['rgb']
                masks['norm_rgb'] = masks.get('rgb', None)

            task_losses = {}
            for task in preds:
                target = tasks_dict[task]
                    
                if loss_on_unmasked:
                    task_losses[task] = tasks_loss_fn[task](preds[task].float(), target)
                else:
                    task_losses[task] = tasks_loss_fn[task](preds[task].float(), target, mask=masks.get(task, None))

            weighted_task_losses = loss_balancer(task_losses)
            loss = sum(weighted_task_losses.values())

        loss_value = sum(task_losses.values()).item()
        task_loss_values = {f'{task}_loss': l.item() for task, l in task_losses.items()}
        weighted_task_loss_values = {f'{task}_loss_weighted': l.item() for task, l in weighted_task_losses.items()}

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm, skip_grad=max_skip_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(**task_loss_values)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(
                {
                    'loss': loss_value,
                    'lr': max_lr,
                    'weight_decay': weight_decay_value,
                    'grad_norm': grad_norm,
                }
            )
            log_writer.update(task_loss_values)
            log_writer.update(weighted_task_loss_values)
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {'[Epoch] ' + k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)