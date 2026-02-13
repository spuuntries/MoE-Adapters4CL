

import os
import csv
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from continuum import ClassIncremental, InstanceIncremental
from continuum.datasets import (
    CIFAR100, ImageNet100, TinyImageNet200, ImageFolderDataset, Core50,
)
from .utils import get_dataset_class_names

OFFICEHOME_CLASSNAMES = [
    "Alarm Clock", "Backpack", "Batteries", "Bed", "Bike",
    "Bottle", "Bucket", "Calculator", "Calendar", "Candles",
    "Chair", "Clipboards", "Computer", "Couch", "Curtains",
    "Desk Lamp", "Drill", "Eraser", "Exit Sign", "Fan",
    "File Cabinet", "Flipflops", "Flowers", "Folder", "Fork",
    "Glasses", "Hammer", "Helmet", "Kettle", "Keyboard",
    "Knives", "Lamp Shade", "Laptop", "Marker", "Monitor",
    "Mop", "Mouse", "Mug", "Notebook", "Oven",
    "Pan", "Paper Clip", "Pen", "Pencil", "Postit Notes",
    "Printer", "Push Pin", "Radio", "Refrigerator", "Ruler",
    "Scissors", "Screwdriver", "Shelf", "Sink", "Sneakers",
    "Soda", "Speaker", "Spoon", "Table", "Telephone",
    "ToothBrush", "Toys", "Trash Can", "TV", "Webcam",
]

# Map Office-Home folder names to clean class names
_OFFICEHOME_FOLDER_TO_CLASS = {
    "Alarm_Clock": 0, "Backpack": 1, "Batteries": 2, "Bed": 3, "Bike": 4,
    "Bottle": 5, "Bucket": 6, "Calculator": 7, "Calendar": 8, "Candles": 9,
    "Chair": 10, "Clipboards": 11, "Computer": 12, "Couch": 13, "Curtains": 14,
    "Desk_Lamp": 15, "Drill": 16, "Eraser": 17, "Exit_Sign": 18, "Fan": 19,
    "File_Cabinet": 20, "Flipflops": 21, "Flowers": 22, "Folder": 23, "Fork": 24,
    "Glasses": 25, "Hammer": 26, "Helmet": 27, "Kettle": 28, "Keyboard": 29,
    "Knives": 30, "Lamp_Shade": 31, "Laptop": 32, "Marker": 33, "Monitor": 34,
    "Mop": 35, "Mouse": 36, "Mug": 37, "Notebook": 38, "Oven": 39,
    "Pan": 40, "Paper_Clip": 41, "Pen": 42, "Pencil": 43, "Postit_Notes": 44,
    "Printer": 45, "Push_Pin": 46, "Radio": 47, "Refrigerator": 48, "Ruler": 49,
    "Scissors": 50, "Screwdriver": 51, "Shelf": 52, "Sink": 53, "Sneakers": 54,
    "Soda": 55, "Speaker": 56, "Spoon": 57, "Table": 58, "Telephone": 59,
    "ToothBrush": 60, "Toys": 61, "Trash_Can": 62, "TV": 63, "Webcam": 64,
}

OFFICEHOME_DOMAIN_ORDERS = {
    1: ["Art", "Clipart", "Product", "Real World"],
    2: ["Clipart", "Art", "Real World", "Product"],
    3: ["Product", "Clipart", "Art", "Real World"],
    4: ["Real World", "Product", "Clipart", "Art"],
    5: ["Art", "Real World", "Product", "Clipart"],
}


class ImageNet1000(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "val")
        return super().get_data()


class OfficeHomeTaskSet(torch.utils.data.Dataset):
    """PyTorch Dataset for one or more Office-Home domains.
    Yields (image_tensor, label, task_id) tuples.
    """
    def __init__(self, paths, labels, task_ids, transforms=None):
        self.paths = paths
        self.labels = labels
        self.task_ids = task_ids
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img, self.labels[idx], self.task_ids[idx]


class OfficeHomeDILScenario:
    """Custom DIL scenario for Office-Home, bypasses continuum.
    Splits data by domain. Supports iteration, len, and slicing.
    """
    def __init__(self, data_root, is_train, domain_order=1, transforms=None):
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "utils", "datautils", "officehome.csv")

        domain_names = OFFICEHOME_DOMAIN_ORDERS.get(domain_order,
                                                     OFFICEHOME_DOMAIN_ORDERS[1])
        split_type = "train" if is_train else "test"

        self._domain_data = {i: {"paths": [], "labels": [], "task_ids": []}
                             for i in range(len(domain_names))}

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                domain_type, data_path, data_split = row[0], row[2], row[3]
                if data_split != split_type or domain_type not in domain_names:
                    continue
                data_path_clean = data_path.replace("office_home/", "")
                cls_name = os.path.basename(os.path.dirname(data_path_clean))
                cls_id = _OFFICEHOME_FOLDER_TO_CLASS.get(cls_name, -1)
                if cls_id < 0:
                    continue
                domain_id = domain_names.index(domain_type)
                abs_path = os.path.join(data_root, data_path_clean)
                self._domain_data[domain_id]["paths"].append(abs_path)
                self._domain_data[domain_id]["labels"].append(cls_id + domain_id * 65)
                self._domain_data[domain_id]["task_ids"].append(domain_id)

        self._num_tasks = len(domain_names)
        self._transforms = transforms
        self._domain_names = domain_names
        print(f"[OfficeHomeDIL] Loaded {split_type}: " +
              ", ".join(f"{domain_names[i]}={len(self._domain_data[i]['paths'])}"
                       for i in range(self._num_tasks)))

    def __len__(self):
        return self._num_tasks

    def __iter__(self):
        for i in range(self._num_tasks):
            yield self[i]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._get_slice(idx)
        d = self._domain_data[idx]
        return OfficeHomeTaskSet(d["paths"], d["labels"], d["task_ids"],
                                self._transforms)

    def _get_slice(self, s):
        indices = range(*s.indices(self._num_tasks))
        all_paths, all_labels, all_tids = [], [], []
        for i in indices:
            d = self._domain_data[i]
            all_paths.extend(d["paths"])
            all_labels.extend(d["labels"])
            all_tids.extend(d["task_ids"])
        return OfficeHomeTaskSet(all_paths, all_labels, all_tids,
                                self._transforms)



def get_dataset(cfg, is_train, transforms=None):
    if cfg.dataset == "cifar100":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = CIFAR100(
            data_path=data_path, 
            download=True, 
            train=is_train, 
            # transforms=transforms
        )
        classes_names = dataset.dataset.classes

    elif cfg.dataset == "tinyimagenet":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = TinyImageNet200(
            data_path, 
            train=is_train,
            download=True
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
        
    elif cfg.dataset == "imagenet100":
        data_path = os.path.join(cfg.dataset_root, "ImageNet")
        dataset = ImageNet100(
            data_path, 
            train=is_train,
            data_subset=os.path.join('/home/dhw/yjz_workspace/project1_y/CIL_ours_compare_v3_lr_5e_3_1router_l2/Continual-CLIP/dataset_reqs/imagenet100_splits', "train_100.txt" if is_train else "val_100.txt")
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif cfg.dataset == "imagenet1000":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = ImageNet1000(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif cfg.dataset == "core50":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = dataset = Core50(
            data_path, 
            scenario="domains", 
            classification="category", 
            train=is_train
        )
        classes_names = [
            "plug adapters", "mobile phones", "scissors", "light bulbs", "cans", 
            "glasses", "balls", "markers", "cups", "remote controls"
        ]
    
    elif cfg.dataset == "officehome":
        # Office-Home uses custom scenario, not continuum
        return None, OFFICEHOME_CLASSNAMES

    else:
        raise ValueError(f"'{cfg.dataset}' is a invalid dataset.")

    return dataset, classes_names


def build_cl_scenarios(cfg, is_train, transforms) -> nn.Module:
    # Office-Home: use custom scenario directly (bypasses continuum)
    if cfg.dataset == "officehome":
        data_path = os.path.join(cfg.dataset_root, "office_home")
        domain_order = cfg.get("domain_order", 1)
        scenario = OfficeHomeDILScenario(
            data_path, is_train, domain_order, transforms=transforms)
        return scenario, OFFICEHOME_CLASSNAMES

    dataset, classes_names = get_dataset(cfg, is_train)

    if cfg.scenario == "class":
        scenario = ClassIncremental(
            dataset,
            initial_increment=cfg.initial_increment,
            increment=cfg.increment,
            transformations=transforms.transforms,
            class_order=cfg.class_order,
        )

    elif cfg.scenario == "domain":
        scenario = InstanceIncremental(
            dataset,
            transformations=transforms.transforms,
        )

    elif cfg.scenario == "task-agnostic":
        NotImplementedError("Method has not been implemented. Soon be added.")

    else:
        ValueError(f"You have entered `{cfg.scenario}` which is not a defined scenario, "
                    "please choose from {{'class', 'domain', 'task-agnostic'}}.")

    return scenario, classes_names