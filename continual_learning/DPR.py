from copy import deepcopy
from time import time

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from continual_learning.IncrementalApproach import IncrementalApproach
from continual_learning.models.BaseModel import BaseModel
from continual_learning.utils.centered_head import centered_head_basis, spectral_drift_losses
from continual_learning.utils.proto import ProtoGenerator
from continual_learning.utils.training_utils import compute_rotations, get_task_features, isolated_rng


class DPR(IncrementalApproach):
    """Decision-Preserving Regularization with affine statistics compensation."""

    def __init__(self, args, device, out_path, class_per_task, task_dict):
        super().__init__(args, device, out_path, class_per_task, task_dict)
        self.model = BaseModel(backbone=self.backbone, dataset=args.dataset)
        self.old_backbone = None
        self.stable_basis = None
        self.lambda_s = args.lambda_s
        self.lambda_c = args.lambda_c
        self.transport_ridge = args.transport_ridge
        self.transport_views = args.transport_views
        self.no_save_checkpoints = args.no_save_checkpoints
        self.balanced_bs = args.balanced_bs
        self.balanced_epochs = args.balanced_epochs
        self.balanced_lr = args.balanced_lr
        self.proto_generator = ProtoGenerator(
            device=device, task_dict=task_dict, batch_size=args.batch_size,
            out_path=out_path, feature_space_size=self.model.get_feat_size())
        self.print_running_approach()

    def print_running_approach(self):
        super().print_running_approach()
        print("DPR: lambda_s={}, lambda_c={}".format(self.lambda_s, self.lambda_c))
        print("Affine transport ridge: {}".format(self.transport_ridge))
        print("Affine fitting views: {} (ridge scaled by sample count)".format(self.transport_views))
        print("Task-end feature cache: enabled")
        print("Save checkpoints: {}".format(not self.no_save_checkpoints))
        print("Classifier calibration: batch size {}, epochs {}, lr {}".format(
            self.balanced_bs, self.balanced_epochs, self.balanced_lr))

    def pre_train(self, task_id):
        if task_id == 0 and self.rotation:
            self.auxiliary_classifier = nn.Linear(
                self.model.get_feat_size(), len(self.task_dict[task_id]) * 3).to(self.device)
        else:
            self.auxiliary_classifier = None
            print("\nFreezing Batch Norm")
            self.model.freeze_bn()

        if task_id > 0:
            self.old_backbone = deepcopy(self.model.backbone).requires_grad_(False).eval()

        self.model.add_classification_head(len(self.task_dict[task_id]))
        self.model.to(self.device)
        super().pre_train(task_id)

    def regularization_losses(self, features, old_features):
        stable, complement = spectral_drift_losses(features, old_features, self.stable_basis)
        return stable, complement, self.lambda_s, self.lambda_c

    def train(self, task_id, trn_loader, epoch, epochs):
        if task_id == 0:
            self.model.train()
        else:
            self.model.eval()
        if self.auxiliary_classifier is not None:
            self.auxiliary_classifier.train()

        start = time()
        count_batches = 0
        drift_totals = torch.zeros(2, device=self.device)
        drift_samples = 0
        for images, targets in tqdm(trn_loader):
            count_batches += 1
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            current_batch_size = images.shape[0]

            if task_id == 0 and self.rotation:
                images_rot, targets_rot = compute_rotations(
                    images, self.image_size, self.task_dict, targets, task_id)
                images = torch.cat([images, images_rot], dim=0)
                targets = torch.cat([targets, targets_rot], dim=0)

            if task_id > 0:
                with torch.no_grad():
                    old_features = self.old_backbone(images)

            features = self.model.backbone(images)
            logits = self.model.heads[task_id](features)
            if task_id == 0 and self.rotation:
                logits = torch.cat([logits, self.auxiliary_classifier(features)], dim=1)

            loss = F.cross_entropy(logits, self.rescale_targets(targets, task_id))
            if task_id > 0:
                first_loss, second_loss, first_weight, second_weight = self.regularization_losses(
                    features, old_features)
                loss = loss + first_weight * first_loss + second_weight * second_loss
                drift_totals += torch.stack((first_loss.detach(), second_loss.detach())) * current_batch_size
                drift_samples += current_batch_size

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        if drift_samples:
            first_loss, second_loss = (drift_totals / drift_samples).tolist()
            weighted_first, weighted_second = first_weight * first_loss, second_weight * second_loss
            print("DPR losses: stable {:.6f}, complement {:.6f}".format(weighted_first, weighted_second))
        print("Task {}, Epoch {}/{}, N_batch {}, Elapsed Time {}s".format(
            task_id, epoch, epochs, count_batches, time() - start))

    @torch.no_grad()
    def eval(self, current_training_task, test_id, loader, verbose):
        self.model.eval()
        n_samples = 0
        if verbose:
            cls_loss = torch.zeros((), device=self.device)
        else:
            weights = torch.cat([head[0].weight for head in self.model.heads])
            correct = torch.zeros(2, dtype=torch.long, device=self.device)
            first_class = self.task_dict[test_id][0]
            last_class = first_class + len(self.task_dict[test_id])

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, dtype=torch.long, non_blocking=True)
            features = self.model.backbone(images)
            n_samples += images.shape[0]
            if verbose:
                logits = self.model.heads[current_training_task](features)
                cls_loss += F.cross_entropy(
                    logits, self.rescale_targets(targets, current_training_task), reduction="sum")
            else:
                logits = F.linear(features, weights)
                correct[0] += (logits[:, first_class:last_class].argmax(dim=1)
                               == self.rescale_targets(targets, test_id)).sum()
                correct[1] += (logits.argmax(dim=1) == targets).sum()

        if verbose:
            print(" - classification loss: {}".format(cls_loss.item() / n_samples))
        else:
            return tuple(value / n_samples for value in correct.tolist())

    def balance_head(self, cached_features):
        """Reuse current features and resample old Gaussian features each epoch."""
        if self.balanced_epochs <= 0:
            return
        weights = nn.Parameter(torch.cat([head[0].weight.detach() for head in self.model.heads]))
        optimizer = torch.optim.SGD(
            [weights], lr=self.balanced_lr, weight_decay=5e-4, momentum=0.9)

        generator = self.proto_generator
        for _ in range(self.balanced_epochs):
            current_features, current_targets = cached_features
            new_count, dimensions = current_features.shape
            total = new_count + sum(generator.class_stats[label] for label in generator.gaussians)
            features = current_features.new_empty((total, dimensions))
            targets = current_targets.new_empty(total)
            features[:new_count] = current_features
            targets[:new_count] = current_targets
            start = new_count
            for label, gaussian in generator.gaussians.items():
                end = start + generator.class_stats[label]
                mean, factor = gaussian.mean.to(features), gaussian.scale_tril.to(features)
                noise = torch.randn(end - start, dimensions, device=features.device, dtype=features.dtype)
                torch.mm(noise, factor.T, out=features[start:end])
                features[start:end].add_(mean)
                targets[start:end] = label
                start = end
            del current_features, current_targets
            order = torch.randperm(total, device=features.device)
            for indices in order.split(self.balanced_bs):
                logits = F.linear(features[indices], weights)
                loss = F.cross_entropy(logits, targets[indices])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            del features, targets

        with torch.no_grad():
            start = 0
            for head in self.model.heads:
                end = start + head[0].out_features
                head[0].weight.copy_(weights[start:end])
                start = end

    def post_train(self, task_id, trn_loader):
        if task_id > 0:
            with torch.no_grad():
                print("Computing affine prototype alignment")
                start = time()
                features, targets, old_features = get_task_features(
                    self.model.backbone, self.old_backbone, trn_loader, self.device)
                W, b, transport_mse = self.fit_transport_views(task_id, trn_loader, old_features, features)
                del old_features
                self.old_backbone = None
                self.update_prototypes_affine(W, b)
                print("Affine transport MSE {:.6f}, elapsed time {}".format(transport_mse, time() - start))
            print("Classifier calibration from cached features")
            self.balance_head((features, targets))
        weights = torch.cat([head[0].weight.detach() for head in self.model.heads])
        basis, _ = centered_head_basis(weights)
        self.stable_basis = basis.to(weights)
        print("Centered classifier head: {} stable directions".format(basis.shape[1]))
        if task_id == 0:
            features, targets, _ = get_task_features(
                self.model.backbone, None, trn_loader, self.device)
        self.proto_generator.compute(features.cpu(), targets.cpu(), task_id)

    @torch.no_grad()
    def fit_transport_views(self, task_id, trn_loader, old_features, features):
        """Fit paired augmented views without advancing the training RNG."""
        old_views, new_views = [old_features], [features]
        for view in range(2, self.transport_views + 1):
            seed = self.seed + 100000 + task_id * 10 + view
            with isolated_rng(seed, self.device):
                loader = DataLoader(
                    trn_loader.dataset, batch_size=trn_loader.batch_size, shuffle=True,
                    num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory,
                    collate_fn=trn_loader.collate_fn,
                    generator=torch.Generator().manual_seed(seed))
                new, _, old = get_task_features(
                    self.model.backbone, self.old_backbone, loader, self.device)
            old_views.append(old)
            new_views.append(new)
        if self.transport_views == 1:
            return self.fit_affine_transport(old_features, features)
        old_all, new_all = torch.cat(old_views), torch.cat(new_views)
        return self.fit_affine_transport(old_all, new_all, ridge_scale=len(new_all) / len(features))

    def fit_affine_transport(self, old_features, new_features, ridge_scale=1.0):
        """Fit new = old @ W + b with an identity prior on W."""
        old_mean = old_features.mean(dim=0)
        new_mean = new_features.mean(dim=0)
        old_centered = old_features - old_mean
        new_centered = new_features - new_mean
        eye = torch.eye(old_features.shape[1], device=old_features.device, dtype=old_features.dtype)
        lhs = old_centered.T @ old_centered + self.transport_ridge * ridge_scale * eye
        rhs = old_centered.T @ new_centered + self.transport_ridge * ridge_scale * eye
        W = torch.linalg.solve(lhs, rhs)
        b = new_mean - old_mean @ W
        mse = float(F.mse_loss(old_features @ W + b, new_features).detach())
        return W, b, mse

    def update_prototypes_affine(self, W, b):
        W = W.detach().cpu()
        b = b.detach().cpu()
        for i, (p, var, proto_label) in enumerate(zip(self.proto_generator.prototype,
                                                       self.proto_generator.running_proto_variance,
                                                       self.proto_generator.class_label)):
            mean = p.to(dtype=W.dtype) @ W + b
            covariance = W.T @ var.to(dtype=W.dtype) @ W
            covariance = 0.5 * (covariance + covariance.T)
            self.proto_generator.prototype[i] = mean
            self.proto_generator.running_proto_variance[i] = covariance
            self.proto_generator.update_gaussian(proto_label, mean, covariance)
        self.proto_generator.running_proto = deepcopy(self.proto_generator.prototype)

