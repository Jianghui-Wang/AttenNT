# Nonparametric Teaching of Attention Learners

[Chen Zhang*](https://chen2hang.github.io), [Jianghui Wang*](https://jianghui-wang.github.io/), Bingyang Cheng, Zhongtao Chen, Wendong Xu, Cong Wang, [Marco Canini](https://mcanini.github.io/), [Francesco Orabona](https://francesco.orabona.com/) [Yik-Chung Wu](https://www.eee.hku.hk/~ycwu), [Ngai Wong](https://www.eee.hku.hk/~nwong)

[[`Paper`](https://chen2hang.github.io/_publications/nonparametric_teaching_of_attention_learners/ICLR_2026_Paper.pdf)] | [[`Project Page`](https://chen2hang.github.io/_publications/nonparametric_teaching_of_attention_learners/attent.html)]

This is the official PyTorch implementation of the **[ICLR 2026]** paper: **[Nonparametric Teaching Attention Learners](https://chen2hang.github.io/_publications/nonparametric_teaching_of_attention_learners/ICLR_2026_Paper.pdf)**.

# MultiMAE with AtteNTComplex

This repository contains implementations of the AtteNT (Attention-based Training) algorithms for efficient model training. The repository includes two main projects:

**MultiMAE with AtteNTComplex**: Multi-modal masked autoencoder pre-training
**LLM Fine-tuning with AtteNTSimple**: Large language model adaptation (Llama, Gemma, etc.)

## Overview

The AtteNT family of algorithms improves training efficiency by selectively focusing on samples based on their loss dynamics. This repository provides:

**AtteNTComplex**: Advanced selective retraining for MultiMAE pre-training with multiple selection strategies and adaptive scheduling
**AtteNTSimple**: Streamlined version for LLM fine-tuning

## Repository Structure

```
.
├── multimae/                      # Core MultiMAE implementation
│   ├── multimae.py               # Main MultiMAE model
│   ├── input_adapters.py         # Input adapters for different modalities
│   ├── output_adapters.py        # Output adapters for different tasks
│   ├── criterion.py              # Loss functions
│   └── multimae_utils.py         # Utility functions
│
├── finetuning/                    # Fine-tuning scripts
│   ├── run_finetuning_cls.py     # Image classification
│   ├── run_finetuning_depth.py   # Depth estimation
│   └── run_finetuning_semseg.py  # Semantic segmentation
│
├── utils/                         # Utility modules
│   ├── datasets.py               # Dataset loading and processing
│   ├── datasets_semseg.py        # Semantic segmentation datasets
│   ├── task_balancing.py         # Task loss balancing strategies
│   ├── taskonomy/                # Taskonomy dataset utilities
│   └── ...                       # Other utilities
│
├── run_pretraning_multimae.py    # Main pre-training script with AtteNTComplex
└── train.py                      # Training for LLM
```


## Usage

### Pre-training with AtteNTComplex

The main pre-training script supports the AtteNTComplex algorithm with various configuration options:

```bash
python run_pretraning_multimae.py \
    --config <path/to/config.yaml> \
    --output_dir <output/directory> \
    --batch_size 32 \
    --epochs 800 \
    --ratio_schedule cosine \
    --selection_strategy hard \
    --initial_ratio 0.2 \
    --final_ratio 0.8 \
    --initial_interval 5 \
    --max_interval 50
```

#### AtteNTComplex Parameters

- `--ratio_schedule`: Ratio scheduling strategy (`cosine` or `incremental`)
- `--interval_schedule`: Interval scheduling strategy (`fixed` or `incremental`)
- `--selection_strategy`: Sample selection strategy (`random`, `hard`, or `soft`)
- `--initial_ratio`: Initial sampling ratio (default: 0.2)
- `--final_ratio`: Final sampling ratio (default: 0.8)
- `--initial_interval`: Initial retraining interval in epochs (default: 5)
- `--max_interval`: Maximum retraining interval (default: 10)
- `--total_cycles`: Total number of selection cycles (default: 10)
- `--gumbel_temperature`: Temperature for Gumbel-Softmax (for soft strategy)
- `--soft_normalization`: Normalization method for soft strategy (`sigmoid` or `softmax`)

### Fine-tuning

#### Image Classification

```bash
python finetuning/run_finetuning_cls.py \
    --config <path/to/config.yaml> \
    --pretrained_weights <path/to/pretrained/weights> \
    --output_dir <output/directory> \
    --data_path <path/to/imagenet> \
    --batch_size 128 \
    --epochs 100
```

#### Depth Estimation

```bash
python finetuning/run_finetuning_depth.py \
    --config <path/to/config.yaml> \
    --pretrained_weights <path/to/pretrained/weights> \
    --output_dir <output/directory> \
    --data_path <path/to/dataset> \
    --batch_size 32 \
    --epochs 50
```

#### Semantic Segmentation

```bash
python finetuning/run_finetuning_semseg.py \
    --config <path/to/config.yaml> \
    --pretrained_weights <path/to/pretrained/weights> \
    --output_dir <output/directory> \
    --data_path <path/to/ade20k> \
    --batch_size 16 \
    --epochs 80
```

## AtteNTComplex Algorithm

The AtteNTComplex algorithm performs selective retraining based on sample-level losses:

1. **Forward Pass**: Compute losses for all training samples
2. **Sample Selection**: Select a subset of samples based on the chosen strategy:
   - **Random**: Randomly sample a subset
   - **Hard**: Select samples with highest losses
   - **Soft**: Use Gumbel-Softmax for differentiable sampling
3. **Retraining**: Train on selected samples for a specified interval
4. **Adaptation**: Update sampling ratio and interval according to the schedule
5. **Repeat**: Continue the cycle until pre-training is complete

### Selection Strategies

- **Random Selection**: Provides a baseline by randomly sampling data
- **Hard Selection**: Focuses on the most challenging samples (highest loss)
- **Soft Selection**: Uses Gumbel-Softmax for a differentiable sampling process

### Scheduling

- **Ratio Schedule**: Controls how the sampling ratio evolves (from `initial_ratio` to `final_ratio`)
  - Cosine: Smooth transition using cosine annealing
  - Incremental: Linear increase
- **Interval Schedule**: Controls retraining interval duration
  - Fixed: Constant interval throughout training
  - Incremental: Gradually increase interval length


## Supported Datasets

- **ImageNet**: Image classification
- **ADE20K**: Semantic segmentation
- **NYUv2**: Depth estimation
- **Taskonomy**: Multi-task learning
- **COCO**: Instance segmentation and detection

## Configuration

Example YAML configuration:

```yaml
# Model
model: pretrain_multimae_base
input_size: 224
patch_size: 16

# Input modalities
in_domains: [rgb, depth, semseg]

# Training
batch_size: 32
epochs: 400
lr: 1.5e-4
weight_decay: 0.05
warmup_epochs: 40

# AtteNTComplex
ratio_schedule: cosine
selection_strategy: hard
initial_ratio: 0.2
final_ratio: 0.8
total_cycles: 10
```

## Pre-training with AtteNTSimple

An example.

```bash
pdeepspeed --master_port=<master/port> --include=<localhost>train.py \
    --model_name_or_path <base/model> \
    --data_path ./data.json \
    --output_dir ./output_llm \
    --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
    --lora_rank 128 \
    --lora_alpha 128 \
    --lora_dropout 0 \
    --attent_ratio 0.7
```


## Related works
Related works for developing a deeper understanding of AtteNT are: <br>
<p class="indented">[ICML 2025 Spotlight] <a href="http://arxiv.org/pdf/2505.14170">Nonparametric Teaching for Graph Property Learners</a>,</p>
<p class="indented">[ICML 2024] <a href="https://arxiv.org/pdf/2405.10531">Nonparametric Teaching of Implicit Neural Representations</a>,</p>
<p class="indented">[NeurIPS 2023] <a href="https://arxiv.org/pdf/2311.10318">Nonparametric Teaching for Multiple Learners</a>,</p>
<p class="indented">[ICML 2023] <a href="https://arxiv.org/pdf/2306.03007">Nonparametric Iterative Machine Teaching</a>.<br></p>

## Citation
If you find our work useful in your research, please cite:
```
@inproceedings{zhang2026nonparametric,
  title={Nonparametric Teaching for Attention Learners},
  author={Zhang, Chen and Wang, Jianghui and Cheng, Bingyang and Chen, Zhongtao and Xu, Wendong and Wang, Cong and Canini, Marco and Orabona, Francesco and Wu, Yik-Chung and Wong, Ngai},
  booktitle={ICLR},
  year={2026}
}
```

---

## Contact Us
Please feel free to contact us: [Chen Zhang](https://chen2hang.github.io) or [Jianghui Wang](https://jianghui-wang.github.io/) if you have any questions while starting up AtteNT.

 
