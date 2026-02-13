from omegaconf import DictConfig
from tqdm import tqdm
import torch.nn.functional as F

import clip.clip as clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import get_class_ids_per_task, get_class_names, batch, merge_we_router, wise_we, moving_avg, l2_loss, \
    virtual_vocab, distillation
import copy

from .cc import conceptual_captions

from . import utils
import os
import random

from .dynamic_dataset import DynamicDataset

from types import SimpleNamespace


def _make_clip_args(task_id=0, is_train=True, apply_moe=True, experts_num=22,
                    topk=2, ffn_adapt=True, ffn_option='parallel',
                    ffn_num=64, ffn_adapt_where='AdapterDoubleEncoder',
                    autorouter=False):
    """Create an args namespace with MoE adapter params required by clip.load().
    
    The custom CLIP model in dil/clip/model.py requires these attributes
    on the args object passed through build_model -> CLIP -> ResidualAttentionBlock.
    """
    return SimpleNamespace(
        task_id=task_id,
        is_train=is_train,
        apply_moe=apply_moe,
        experts_num=experts_num,
        topk=topk,
        ffn_adapt=ffn_adapt,
        ffn_option=ffn_option,
        ffn_num=ffn_num,
        ffn_adapt_where=ffn_adapt_where,
        autorouter=autorouter,
    )


class ClassIncremental(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None
        clip_args = _make_clip_args()
        self.model, self.transforms, _ = clip.load(cfg.model_name, device=device, jit=jit, args=clip_args)
        self.ref_model = None
        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        self.dynamic_dataset = DynamicDataset(cfg)

    def forward(self, image, taskid):
        with torch.no_grad():
            logits_per_image, _ = self.model(image, self.text_tokens, 0, is_train=False)
            probs = logits_per_image.softmax(dim=-1)
        return probs

    def adaptation(self, task_id, cfg, train_dataset, train_classes_names):
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)

        if cfg.method != "zeroshot":
            self.train(task_id, cfg, train_dataset, train_classes_names)

    def train(self, task_id, cfg, train_dataset, train_classes_names):
        ### laoding dataset
        train_loader = DataLoader(train_dataset[task_id:task_id + 1],
                                  batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=8)

        train_iter = iter(train_loader)  # 获取每个step的数据集
        # print('cfg.batch_size',cfg.batch_size)


        EPOCH = 1
        num_batches = len(train_loader)
        total_iterations = EPOCH * num_batches

        ### whole-model
        exclude_params_name = ["logit_scale"]

        # 冻结参数
        for k, v in self.model.named_parameters():  # 冻结其他参数
            if "adaptmlp" not in k and "router" not in k and "noise" not in k:
                v.requires_grad = False


        params = [
            v for k, v in self.model.named_parameters() if "adaptmlp" in k or "router" in k or "noise" in k
        ]
        params_name = [
            k for k, v in self.model.named_parameters() if "adaptmlp" in k or "router" in k or "noise" in k
        ]
        # print('========trainable params============', params_name)

        logit_scale = self.model.logit_scale

        # optimizer
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = utils.cosine_lr(
            optimizer, cfg.lr, 30, total_iterations
        )

        # move model to device
        self.model = self.model.cuda()
        devices = list(range(torch.cuda.device_count()))
        # print("Using devices", devices)

        # text
        classnames = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        print(classnames)
        texts = [self.prompt_template.format(c) for c in classnames]

        texts = clip.tokenize(texts).to(self.device)

        # method

        # start training
        self.model.train()
        for iteration in tqdm(range(total_iterations + 1)):
            scheduler(iteration)
            try:
                inputs, targets, task_ids = next(train_iter)
            except:
                train_iter = iter(train_loader)
                inputs, targets, task_ids = next(train_iter)

            if cfg.dataset == "tinyimagenet" and task_id != 0:
                shift = 100 + (task_id - 1) * cfg.increment
                targets -= shift
            elif cfg.dataset == "imagenet100" and task_id != 0:
                shift = cfg.initial_increment + (task_id - 1) * cfg.increment
                targets -= shift
            else:
                shift = task_id * cfg.increment
                targets -= shift

            inputs, targets = inputs.cuda(), targets.cuda()

            logits_per_image, _ = self.model(inputs, texts)
            # -- cross entropy loss --
            loss = F.cross_entropy(logits_per_image, targets, label_smoothing=cfg.ls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        self.model.eval()


class DomainIncremental(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None
        clip_args = _make_clip_args()
        self.model, self.transforms, _ = clip.load(cfg.model_name, device=device, jit=jit, args=clip_args)
        self.ref_model = None
        self.num_classes = cfg.get("num_classes", 65)
        self.text_tokens = None

    def forward(self, image, taskid):
        with torch.no_grad():
            logits_per_image, _ = self.model(image, self.text_tokens, 0, is_train=False)
            probs = logits_per_image.softmax(dim=-1)
        return probs

    def adaptation(self, task_id, cfg, train_dataset, train_classes_names):
        # DIL: always use ALL class names (same classes across domains)
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.classes_names[:self.num_classes]]
        ).to(self.device)

        if cfg.method != "zeroshot":
            self._train(task_id, cfg, train_dataset, train_classes_names)

    def _train(self, task_id, cfg, train_dataset, train_classes_names):
        ### loading dataset - one domain per task
        train_loader = DataLoader(train_dataset[task_id:task_id + 1],
                                  batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=8)

        train_iter = iter(train_loader)

        EPOCH = 1
        num_batches = len(train_loader)
        total_iterations = EPOCH * num_batches

        # Freeze all except adapter/router/noise params
        for k, v in self.model.named_parameters():
            if "adaptmlp" not in k and "router" not in k and "noise" not in k:
                v.requires_grad = False

        params = [
            v for k, v in self.model.named_parameters()
            if "adaptmlp" in k or "router" in k or "noise" in k
        ]
        params_name = [
            k for k, v in self.model.named_parameters()
            if "adaptmlp" in k or "router" in k or "noise" in k
        ]

        logit_scale = self.model.logit_scale

        # Optimizer
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = utils.cosine_lr(
            optimizer, cfg.lr, 30, total_iterations
        )

        # Move model to device
        self.model = self.model.cuda()

        # Text tokens for current domain's classes (same 65 classes)
        texts = clip.tokenize(
            [self.prompt_template.format(c) for c in self.classes_names[:self.num_classes]]
        ).to(self.device)

        # Start training
        self.model.train()
        for iteration in tqdm(range(total_iterations + 1)):
            scheduler(iteration)
            try:
                inputs, targets, task_ids = next(train_iter)
            except:
                train_iter = iter(train_loader)
                inputs, targets, task_ids = next(train_iter)

            # DIL: shift targets to within-domain class range [0, num_classes)
            targets = targets % self.num_classes

            inputs, targets = inputs.cuda(), targets.cuda()

            logits_per_image, _ = self.model(inputs, texts)
            # Cross entropy loss
            loss = F.cross_entropy(logits_per_image, targets, label_smoothing=cfg.ls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.model.eval()


class TaskAgnostic(nn.Module):
    pass


def load_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    r"""Load a CLIP model in different continual scenarios.

    Arguments:
        cfg (DictConfig): Experiment configurations.
        device (torch.device): Device to train (or) evaluate the model on.

    Returns:
        nn.Module: Return scenario specific CLIP model.
    """
    if cfg.scenario == "class":
        return ClassIncremental(cfg, device)
    elif cfg.scenario == "domain":
        return DomainIncremental(cfg, device)
    elif cfg.scenario == "task-aganostic":
        return TaskAgnostic(cfg, device)
    else:
        raise ValueError(f"""
            `{cfg.scenarios}` is not a valid scenario, 
            Please choose from ['class', "domain', 'task-agnostic']
        """)

