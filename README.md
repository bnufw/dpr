# From Joint Risk to Decision Invariance: Decision-Preserving Regularization for Exemplar-Free Class-Incremental Learning

Anonymous implementation of **Decision-Preserving Regularization (DPR)** for CIFAR-100, Tiny-ImageNet, and ImageNet-Subset. DPR extracts an orthonormal decision subspace from the centered old classifier using SVD, regularizes feature drift in that subspace and its orthogonal complement, and transports old Gaussian statistics with a closed-form affine map before calibrating the global classifier. All models use ResNet-18 trained from scratch, without storing old images.

## Environment

Python 3.12, PyTorch 2.7.1, and a CUDA-capable GPU are required. Install the pinned runtime dependencies with:

```bash
conda env create -f environment.yml
conda activate dpr
```

## Data

Prepare the datasets locally. The code does not download datasets or pretrained weights. Each `--data_path` points to the parent of the corresponding dataset directory:

```text
data/
├── cifar-100-python/
│   ├── train
│   ├── test
│   └── meta
├── tiny-imagenet-200/
│   ├── train/<class>/images/*.JPEG
│   └── val/
│       ├── images/*.JPEG
│       └── val_annotations.txt
└── imagenet_subset/
    ├── train/<class>/*.JPEG
    └── val/<class>/ILSVRC2012_val_*.JPEG
```

- **CIFAR-100:** use the official Python distribution.
- **Tiny-ImageNet:** use the original 200-class training directories and validation annotations.
- **ImageNet-Subset:** retain exactly the 100 training class directories listed in `dataset/splits/imagenet_subset_classes.txt`. Organize validation images by class as shown above. The loader reads the bundled `dataset/splits/imagenet_subset_val.txt`, which specifies 5,000 validation images with 50 images per class; a separate annotation file in the data directory is not required. Original labels follow the sorted class list before the seeded incremental class ordering is applied.

## Training

Run commands from this directory. The six settings use equal-sized tasks and seed 0. Replace `./data` with each dataset's local parent directory and `--device 0` with the desired CUDA device index.

| Dataset | Classes per task, T10 / T20 | Affine views | First / later task epochs |
| --- | --- | --- | --- |
| CIFAR-100 | 10 / 5 | 4 | 100 / 100 |
| Tiny-ImageNet | 20 / 10 | 1 | 100 / 100 |
| ImageNet-Subset | 10 / 5 | 1 | 160 / 100 |

The commands use the default coefficients `--lambda_s 0.5 --lambda_c 0.01`, corresponding to the paper's decision-subspace coefficient $\lambda_1$ and orthogonal-complement coefficient $\lambda_2$. Training uses batch size 64. Affine fitting uses `--transport_ridge 0.001`, scaled by the number of views. Current-task features from the first paired pass are always cached for calibration and new-class statistics; additional views are used only for affine fitting. Calibration uses 50 epochs of SGD with learning rate 0.001 and batch size 256, sampling fresh old-class Gaussian features each epoch.

```bash
# CIFAR-100
python -B main.py --dataset cifar100 --data_path ./data --device 0 --seed 0 --n_task 10 --transport_views 4
python -B main.py --dataset cifar100 --data_path ./data --device 0 --seed 0 --n_task 20 --transport_views 4

# Tiny-ImageNet
python -B main.py --dataset tiny-imagenet --data_path ./data --device 0 --seed 0 --n_task 10 --transport_views 1
python -B main.py --dataset tiny-imagenet --data_path ./data --device 0 --seed 0 --n_task 20 --transport_views 1

# ImageNet-Subset
python -B main.py --dataset imagenet-subset --data_path ./data --device 0 --seed 0 --n_task 10 --transport_views 1
python -B main.py --dataset imagenet-subset --data_path ./data --device 0 --seed 0 --n_task 20 --transport_views 1
```

The first task includes auxiliary rotation training. CIFAR-100 and Tiny-ImageNet use Adam with learning rate 0.001 for the first task; ImageNet-Subset uses SGD with learning rate 0.1. Later tasks use Adam with learning rate 0.0001 for CIFAR-100 and Tiny-ImageNet, and 0.00001 for the backbone and 0.0001 for the head on ImageNet-Subset. Batch normalization is frozen after the first task. See `python -B main.py --help` for available options.

## Results

The completed seed-0 runs are included under [logs/reference](logs/reference), with their metrics in [experiment_comparison.csv](logs/reference/experiment_comparison.csv). Each run includes its class order, accuracy and forgetting matrices, and console logs. Machine-specific paths have been replaced with portable paths; numerical results and recorded timings are unchanged. Model checkpoints are not included in the supplementary archive.

New runs default to `<repository>/logs/<dataset>_t<n_task>/`, regardless of the working directory. For example, CIFAR-100 T10 writes to `logs/cifar100_t10/`. Set `--outpath` to override this location; explicit relative paths are resolved from the working directory. The included reference logs remain in their separate directory.

Each `--outpath` contains:

- `logger/`: accuracy and forgetting matrices; `acc_tag_perstep.out` contains accuracy over all classes seen at each stage, without a task identifier.
- `class_order.json`: the original-to-incremental label mapping and task split.
- `<task>_model.pth`: backbone and classifier state dictionaries after each task; use `--no_save_checkpoints` to omit them.
- `stdout-*.txt` and `stderr-*.txt`: console output.

**Acc** is the last value in `logger/acc_tag_perstep.out`; **AAA** is the mean of its per-stage task-agnostic accuracies. Both are percentages. Tasks are numbered from zero in output files. Use a distinct output directory for each run, since repeated runs overwrite matching results.

