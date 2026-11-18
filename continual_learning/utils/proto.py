from copy import deepcopy
import torch
from torch.distributions import MultivariateNormal

class ProtoGenerator:
    def __init__(self, device, task_dict, batch_size, out_path, feature_space_size):
        self.device = device
        self.task_dict = task_dict
        self.batch_size = batch_size
        self.out_path = out_path
        self.feature_space_size = feature_space_size
        self.prototype = []
        self.class_label = []
        self.running_proto = None
        self.running_proto_variance = []
        self.gaussians = {}
        self.class_stats = {}

    @torch.no_grad()
    def compute(self, features, labels, current_task):
        for label in self.task_dict[current_task]:
            feature_classwise = features[labels == label]
            self.class_stats[label] = feature_classwise.shape[0]

            proto = feature_classwise.mean(dim=0).cpu()

            covariance = torch.cov(feature_classwise.T).cpu()

            self.running_proto_variance.append(covariance)
            self.prototype.append(proto)
            self.class_label.append(label)
            self.update_gaussian(label, proto, covariance)

        self.running_proto = deepcopy(self.prototype)

    def update_gaussian(self, proto_label, mean, var):
        mean = mean.detach().cpu()
        covariance = var.detach().cpu().double()
        covariance = 0.5 * (covariance + covariance.T)
        if not torch.isfinite(covariance).all():
            raise ValueError(f"Prototype {proto_label}: non-finite covariance")
        identity = torch.eye(covariance.shape[0], dtype=covariance.dtype)
        jitter = 1e-5
        roundoff = torch.finfo(var.dtype).eps * covariance.diagonal().abs().max().item()
        # Few examples in a large feature space give a rank-deficient covariance.
        # Factor in double precision, retaining correlations and the sample dtype.
        for _ in range(6):
            factor, info = torch.linalg.cholesky_ex(covariance + jitter * identity)
            if info.item() == 0:
                self.gaussians[proto_label] = MultivariateNormal(
                    mean, scale_tril=factor.to(dtype=mean.dtype))
                if jitter > 1e-5:
                    print(f"Prototype {proto_label}: covariance jitter {jitter:g}")
                return
            jitter = max(jitter * 10, roundoff)
        raise ValueError(f"Prototype {proto_label}: covariance factorization failed after 6 attempts")
