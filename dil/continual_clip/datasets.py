

import os
import csv
import numpy as np
import torch.nn as nn

from continuum import ClassIncremental, InstanceIncremental
from continuum.datasets import (
    CIFAR100, ImageNet100, TinyImageNet200, ImageFolderDataset, Core50,
    InMemoryDataset,
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


def _load_officehome_csv(data_root, is_train, domain_order=1):
    """Load Office-Home from officehome.csv with domain-based task IDs.
    
    Returns an InMemoryDataset with (paths, labels, task_ids).
    """
    # Determine the CSV path - look relative to the dil directory
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                            "utils", "datautils", "officehome.csv")
    
    domain_names = OFFICEHOME_DOMAIN_ORDERS.get(domain_order, OFFICEHOME_DOMAIN_ORDERS[1])
    split_type = "train" if is_train else "test"
    
    x_paths, y_labels, t_tasks = [], [], []
    
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # env, label, path, split
        for row in reader:
            domain_type = row[0]
            data_path = row[2]
            data_split = row[3]
            
            if data_split != split_type:
                continue
            if domain_type not in domain_names:
                continue
                
            # Get class ID from folder name
            data_path_clean = data_path.replace("office_home/", "")
            cls_name = os.path.basename(os.path.dirname(data_path_clean))
            cls_id = _OFFICEHOME_FOLDER_TO_CLASS.get(cls_name, -1)
            if cls_id < 0:
                continue
            
            domain_id = domain_names.index(domain_type)
            abs_path = os.path.join(data_root, data_path_clean)
            
            x_paths.append(abs_path)
            y_labels.append(cls_id + domain_id * 65)  # domain-shifted label
            t_tasks.append(domain_id)
    
    x = np.array(x_paths)
    y = np.array(y_labels)
    t = np.array(t_tasks)
    
    return InMemoryDataset(x, y, t)



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
        data_path = os.path.join(cfg.dataset_root, "office_home")
        domain_order = cfg.get("domain_order", 1)
        dataset = _load_officehome_csv(data_path, is_train, domain_order)
        classes_names = OFFICEHOME_CLASSNAMES

    else:
        raise ValueError(f"'{cfg.dataset}' is a invalid dataset.")

    return dataset, classes_names


def build_cl_scenarios(cfg, is_train, transforms) -> nn.Module:

    dataset, classes_names = get_dataset(cfg, is_train)

    if cfg.scenario == "class":
        scenario = ClassIncremental(
            dataset,
            initial_increment=cfg.initial_increment,
            increment=cfg.increment,
            transformations=transforms.transforms, # Convert Compose into list
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