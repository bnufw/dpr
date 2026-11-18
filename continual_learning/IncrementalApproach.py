import abc
import os
from sched import scheduler
from tabnanny import verbose
import sys
from continual_learning.utils.OptimizerManager import OptimizerManager

class IncrementalApproach(metaclass=abc.ABCMeta):

   def __init__(self, args, device, out_path, class_per_task, task_dict):
      self.device = device
      self.out_path = out_path
      self.class_per_task = class_per_task
      self.task_dict = task_dict
      self.n_class_first_task = args.n_class_first_task
      # Model args
      self.approach = "dpr"
      self.backbone = args.backbone

      self.batch_size = args.batch_size
      self.milestones_first_task = None
      self.dataset = args.dataset
      if self.dataset=="cifar100":
         self.image_size = 32
      elif self.dataset=="tiny-imagenet":
         self.image_size = 64
      elif self.dataset == "imagenet-subset":
         self.image_size = 224

      self.seed = args.seed
      self.epochs_first_task = args.epochs_first_task
      self.epochs_next_task = args.epochs_next_task

      # SELF-ROTATION classifier
      self.auxiliary_classifier = None
      self.rotation = True
      self.optimizer_manager = OptimizerManager(self.dataset, self.rotation)

   def pre_train(self, task_id, *args):

      self.optimizer, self.reduce_lr_on_plateau  = self.optimizer_manager.get_optimizer(task_id, self.model, self.auxiliary_classifier)
      if task_id == 0:
         self.total_epochs = self.epochs_first_task
      else:
         self.total_epochs =  self.epochs_next_task

      print("Approach total epochs {}".format(self.total_epochs))

   def rescale_targets(self, targets, t):
       offset =  (t-1)*self.class_per_task + self.n_class_first_task  if self.n_class_first_task > -1 and t > 0 else t*self.class_per_task
       targets = targets - offset
       return targets

   @abc.abstractmethod
   def train(self, *args):
      pass

   @abc.abstractmethod
   def post_train(self, *args):
      pass

   @abc.abstractmethod
   def eval(self, *args):
      pass

   def print_running_approach(self):
      print("#"*40 + " --> Dataset: {}".format(self.dataset))
      print("- class first task: {}".format(self.n_class_first_task))
      print("- number of tasks: {}".format(len(self.task_dict)))
      print("\n")
      print("#"*40 + " --> RUNNING APPROACH")
      print("- approach: {}".format(self.approach))
      print("- backbone: {}".format(self.backbone))
      print("- batch size : {}".format(self.batch_size))
      print("- epochs first task: {}".format(self.epochs_first_task))
      print("- epochs next task: {}".format(self.epochs_next_task))

      print()
