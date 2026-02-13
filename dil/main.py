
import os
import json
import hydra
import logging
from omegaconf import DictConfig

from tqdm import tqdm

import torch
import statistics
from torch.utils.data import DataLoader

from continual_clip import utils
from continual_clip.models import load_model
from continual_clip.datasets import build_cl_scenarios


@hydra.main(config_path=None, config_name=None, version_base="1.1") 
def continual_clip(cfg: DictConfig) -> None:

    cfg.workdir = utils.get_workdir(path=os.getcwd())
    cfg.dataset_root = os.path.join(cfg.workdir, cfg.dataset_root)

    utils.save_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.class_order = utils.get_class_order(os.path.join(cfg.workdir, cfg.class_order))
    model = load_model(cfg, device)

    eval_dataset, classes_names = build_cl_scenarios(
        cfg, is_train=False, transforms=model.transforms
    )
    train_dataset, train_classes_names = build_cl_scenarios(
        cfg, is_train=True, transforms=model.transforms
    )
    model.classes_names = classes_names

    with open(cfg.log_path, 'w+') as f: 
        pass

    acc_list = []

    # DIL: evaluate across domains
    for task_id, _ in enumerate(eval_dataset):
        logging.info(f"Task {task_id}: adaptation + evaluation")

        model.adaptation(task_id, cfg, train_dataset, train_classes_names)

        # Evaluate on ALL seen domains so far
        eval_loader = DataLoader(eval_dataset[:task_id + 1], batch_size=64)

        correct = 0
        total = 0
        per_domain_correct = {}
        per_domain_total = {}

        for inputs, targets, task_ids in tqdm(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, task_ids)
            preds = outputs.cpu().argmax(dim=1)

            # DIL accuracy: compare within-class predictions (mod num_classes)
            num_classes = cfg.get("num_classes", 65)
            preds_cls = preds % num_classes
            targets_cls = targets.cpu() % num_classes
            batch_correct = (preds_cls == targets_cls).sum().item()
            correct += batch_correct
            total += targets.size(0)

            # Track per-domain accuracy
            for tid in task_ids.unique():
                mask = task_ids == tid
                tid_val = tid.item()
                if tid_val not in per_domain_correct:
                    per_domain_correct[tid_val] = 0
                    per_domain_total[tid_val] = 0
                domain_preds = preds_cls[mask.cpu()]
                domain_targets = targets_cls[mask.cpu()]
                per_domain_correct[tid_val] += (domain_preds == domain_targets).sum().item()
                per_domain_total[tid_val] += mask.sum().item()

        acc = 100.0 * correct / total if total > 0 else 0.0
        acc_list.append(acc)

        per_domain_acc = {
            k: round(100.0 * per_domain_correct[k] / per_domain_total[k], 2)
            for k in sorted(per_domain_correct.keys())
        }

        with open(cfg.log_path, 'a+') as f:
            f.write(json.dumps({
                'task': task_id,
                'acc': round(acc, 2),
                'avg_acc': round(statistics.mean(acc_list), 2),
                'per_domain_acc': per_domain_acc,
            }) + '\n')

        logging.info(f"Task {task_id} acc: {acc:.2f}%, per-domain: {per_domain_acc}")

    with open(cfg.log_path, 'a+') as f:
        f.write(json.dumps({
            'last': round(acc_list[-1], 2), 
            'avg': round(statistics.mean(acc_list), 2)
        }) + '\n')


if __name__ == "__main__":
    continual_clip()
