import os
import json
from torch.utils.data import DataLoader

from continual_learning.DPR import DPR
from dataset.continual_learning_dataset import ContinualLearningDataset
from dataset.dataset_utils import get_dataset
from utilities.generic_utils import (
    result_folder, get_task_dict, seed_everything,
    store_model, get_class_per_task, remap_targets)
from utilities.matrix_logger import Logger
from utilities.parse_utils import get_args

def train_task(approach, task_id, train_loader, valid_loader, args, device):
    print("#"*40 + " --> TRAINING HEAD {}".format(task_id))
    approach.pre_train(task_id)
    for epoch in range(approach.total_epochs):
        print("Epoch {}/{}".format(epoch, approach.total_epochs))
        approach.train(task_id, train_loader, epoch, approach.total_epochs)
        approach.eval(task_id, task_id, valid_loader, verbose=True)
        approach.reduce_lr_on_plateau.step()

def finish_task(approach, logger, task_id, train_loader, test_loaders, output_path):
    approach.post_train(task_id=task_id, trn_loader=train_loader)
    if not approach.no_save_checkpoints:
        store_model(approach, output_path, name=str(task_id))

    for test_id in range(task_id + 1):
        acc_taw_value, acc_tag_value = approach.eval(task_id, test_id,
                                                     test_loaders[test_id], verbose=False)
        logger.update_accuracy(current_training_task_id=task_id,
                               test_id=test_id,
                               acc_taw_value=acc_taw_value,
                               acc_tag_value=acc_tag_value)
        if test_id < task_id:
            logger.update_forgetting(current_training_task_id=task_id, test_id=test_id)
        logger.print_latest(current_training_task_id=task_id, test_id=test_id)

    logger.compute_average()
    logger.print_file()

if __name__ == "__main__":

    # args
    args = get_args()

    print(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)

    # device
    device = "cuda:" + str(args.device)

    out_path = args.outpath

    print("Current Seed {}".format(args.seed))

    # fix seed
    seed_everything(seed=args.seed)

    """
    Dataset Preparation
    """
    train_set, test_set, total_classes = get_dataset(args.dataset, args.data_path, args.n_task)

    # mapping between classes and shuffled classes and re-map dataset classes for different order of classes
    train_set, test_set, label_mapping = remap_targets(train_set, test_set, total_classes, args.dataset)

    # class_per_task: number of classes not in the first task, if the first is larger, otherwise it is equal to total_classes/n_task
    class_per_task = get_class_per_task(args.n_class_first_task, total_classes, args.n_task)

    # task_dict = {task_id: list_of_class_ids}
    task_dict = get_task_dict(args.n_task, total_classes, class_per_task, args.n_class_first_task)

    run_n_tasks = args.n_task if args.max_tasks == -1 else min(args.max_tasks, args.n_task)
    if run_n_tasks < 1:
        raise ValueError("max_tasks must be -1 or a positive integer")
    active_task_dict = {task_id: task_dict[task_id] for task_id in range(run_n_tasks)}

    print("Dataset: {}, N task: {}, First Task Classes: {}, Classes Per Task : {}".format(args.dataset,
                                                                                          args.n_task,
                                                                                          args.n_class_first_task,
                                                                                          class_per_task))

    """
    Generate Subset For Each Task
    """
    cl_train_val = ContinualLearningDataset(train_set, task_dict,
                                            args.n_task, args.n_class_first_task,
                                            class_per_task,total_classes,
                                            valid_size=args.valid_size, train=True)

    train_dataset_list, train_sizes, val_dataset_list, val_sizes = cl_train_val.collect()

    cl_test = ContinualLearningDataset(test_set, task_dict,
                                       args.n_task, args.n_class_first_task,
                                       class_per_task,total_classes,
                                       train=False)

    test_dataset_list, test_sizes, _, _  = cl_test.collect()
    with open(os.path.join(out_path, "class_order.json"), "w") as target:
        json.dump({"original_to_incremental": label_mapping, "tasks": task_dict,
                   "train_sizes": train_sizes, "test_sizes": test_sizes}, target, indent=2)
    test_loaders = [DataLoader(test, batch_size=args.batch_size*4, shuffle=False, num_workers=args.nw, pin_memory=True) for test in test_dataset_list]

    if args.valid_size > 0:
        print("Creating Validation Set")
        train_loaders = [DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.nw, pin_memory=True) for train in train_dataset_list]
        valid_loaders = [DataLoader(valid, batch_size=args.batch_size*4, shuffle=False, num_workers=args.nw, pin_memory=True) for valid in val_dataset_list]

    else:
        print("Not using Validation")
        train_loaders = [DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.nw, pin_memory=True) for train in train_dataset_list]
        valid_loaders = test_loaders

    train_loaders = train_loaders[:run_n_tasks]
    valid_loaders = valid_loaders[:run_n_tasks]
    test_loaders = test_loaders[:run_n_tasks]
    test_sizes = test_sizes[:run_n_tasks]

    """
    Logger Init
    """
    logger = Logger(out_path=out_path, n_task=run_n_tasks,
                    task_dict=active_task_dict, test_sizes=test_sizes)
    result_folder(out_path, "logger")

    approach = DPR(args=args, device=device, out_path=out_path,
                   class_per_task=class_per_task, task_dict=task_dict)
    for task_id, train_loader in enumerate(train_loaders):
        train_task(approach, task_id, train_loader, valid_loaders[task_id], args, device)
        finish_task(approach, logger, task_id, train_loader, test_loaders, out_path)
