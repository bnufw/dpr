import os
import random
from copy import deepcopy
import numpy as np
import torch

def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def get_class_per_task(n_class_first_task, total_classes, n_task):
    assert n_class_first_task <= total_classes

    if n_class_first_task > -1:
        class_per_task = int((total_classes - n_class_first_task)/(n_task - 1))
        assert class_per_task > 1
        assert n_class_first_task + (n_task - 1) * class_per_task  == total_classes
    else:
        class_per_task =  int(total_classes/ n_task)
        assert class_per_task > 1
        assert n_task * class_per_task == total_classes

    return class_per_task

def result_folder(out_path, name):
    if not os.path.exists(os.path.join(out_path, name)):
        os.mkdir(os.path.join(out_path, name))

def get_task_dict(n_task, total_classes, class_per_task, n_class_first_task):
    d = {}
    l = list(range(total_classes))

    if n_class_first_task > - 1:
        offset = n_class_first_task
        for i in range(n_task):
            if i == 0:
                d[i] = [i for i in range(0, n_class_first_task)]
            else:
                d[i] = l[offset + (i-1)*class_per_task: offset+ (i-1)* class_per_task + class_per_task]

    else:

        for i in range(n_task):
            d[i] = l[i*class_per_task:  i* class_per_task + class_per_task]

    return d

def store_model(approach, out_path, name=""):
    torch.save(approach.model.state_dict(), os.path.join(out_path, name+"_model.pth"))

def remap_targets(train_set, test_set, total_classes, dataset):
    labels = list(range(total_classes))
    shuffled = deepcopy(labels)
    random.shuffle(shuffled)
    label_mapping = dict(zip(labels, shuffled))
    for data in (train_set, test_set):
        for i in range(len(data.targets)):
            data.targets[i] = label_mapping[data.targets[i]]
        for key in data.class_to_idx:
            data.class_to_idx[key] = label_mapping[data.class_to_idx[key]]
    return train_set, test_set, label_mapping
