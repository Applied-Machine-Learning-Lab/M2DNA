# M2DNA

This repository contains the implementation of **M2DNA: Multi-Modal Dual-Stream Representation Learning with Joint Relational Modeling for DNA Clustering**.

M2DNA jointly models two representation forms of the same DNA sequence: a visual Frequency Chaos Game Representation (FCGR) and a sequence representation produced by a pretrained nucleotide language model. The model is trained with a Twin Contrastive Loss at both instance and cluster levels. Following an asymmetric optimization strategy, the visual encoder is pretrained offline and then frozen, while the textual encoder is adapted with LoRA. During joint training, the two streams are integrated by the proposed Adaptive Fusion Module, and **Probabilistic Module Dropout (PMD)** randomly suppresses the visual stream to alleviate representation laziness.

## 1. Repository Structure

```text
M2DNA/
├── scripts/
│   └── pretrain_visual_cgrclust.py # Five-run CGRclust checkpoint wrapper
├── src/
│   ├── cluster.py                # Main training and evaluation entry point
│   └── utils/
│       ├── model.py              # Visual encoder, sequence encoder, and fusion
│       ├── engine.py             # Training and evaluation loops
│       ├── data_setup.py         # Dataset classes and k-mer tokenizer
│       ├── data_preprocess.py    # FASTA parsing and preprocessing
│       ├── augmentation_utils.py # Mutation/fragmentation augmentations
│       ├── CGR_utils.py          # CGR and FCGR construction
│       └── loss_function.py      # Instance and cluster losses
```

## 2. Environment

The implementation requires Python 3.8 or later and the following packages:

```bash
pip install torch torchvision transformers peft biopython \
    numpy pandas scipy scikit-learn matplotlib tqdm pillow
```

The sequence encoder is loaded through Hugging Face Transformers. By default, the code uses:

```text
InstaDeepAI/nucleotide-transformer-500m-human-ref
```

The first run may download this model automatically. To use a local copy, pass its path through `--llm_path`.

## 3. Dataset Preparation

The preprocessing and dataset organization follow the format used by [CGRclust](https://github.com/fatemehalipour/CGRclust). Please clone or download the CGRclust repository first and obtain the DNA clustering datasets according to its instructions:

```bash
git clone https://github.com/fatemehalipour/CGRclust.git
```

Keep the required FASTA files in the cloned CGRclust repository. The wrapper
reads them from `CGRclust/data/` and does not require a second copy in M2DNA:

```text
CGRclust/
└── data/
    ├── 01_Cypriniformes.fasta
    ├── 02_*.fasta
    └── ...
```

The default dataset is `01_Cypriniformes.fasta`. A different file can be selected with `--dataset`.

Each FASTA header must contain the sequence identifier followed by its class label, separated by whitespace. For example:

```text
>sequence_0001 Class_A
ACGTACGTACGT...
```

The loader converts `U` to `T` and replaces non-canonical nucleotides with `N`. During FCGR construction, `N`-containing k-mers are ignored. Class labels are read from the second field of each FASTA description, so the header format should be preserved.

## 4. Offline Visual Representation Learning

According to the M2DNA methodology, the visual encoder is first trained offline with the CGRclust implementation. Clone the [CGRclust repository](https://github.com/fatemehalipour/CGRclust) and place the target FASTA file in its `data/` directory. This stage learns global topological features from weakly and strongly augmented FCGR representations using Twin Contrastive Learning.

The FCGR input is generated from non-overlapping k-mers. For the default setting, use `k=6` and heterogeneous mutation augmentations with weak and strong mutation rates of `1e-4` and `1e-2`, respectively. The offline objective is:

```text
L_offline = (1 - w) * L_instance + w * L_cluster
```

From the M2DNA repository root, run the provided wrapper. It reuses the official CGRclust implementation, trains five independent visual models, saves the best epoch of each run, and selects the best run according to clustering accuracy:

```bash
python scripts/pretrain_visual_cgrclust.py \
    --cgrclust_dir /path/to/CGRclust \
    --dataset 01_Cypriniformes.fasta \
    --k 6 \
    --number_of_models 5 \
    --num_epochs 150 \
    --batch_size 512 \
    --lr 7e-5 \
    --temp_ins 0.1 \
    --temp_clu 1.0 \
    --weight 0.7
```

The default output is `checkpoints/<dataset-stem>/`. The wrapper writes `best_visual_model_0.pth` through `best_visual_model_4.pth`, copies the best run to `best_visual_model.pth`, and writes `selection_summary.txt`. Repeat this process for every dataset because M2DNA uses a dataset-specific offline visual encoder.

Organize the selected checkpoints by dataset, for example:

```text
M2DNA/
└── checkpoints/
    ├── 01_Cypriniformes/
    │   └── best_visual_model.pth
    ├── 06_Astrovirus_balanced/
    │   └── best_visual_model.pth
    └── 08_HCV/
        └── best_visual_model.pth
```

The selected checkpoint contains parameters compatible with `backbone.*`, `instance_projector.*`, and `cluster_projector.*` in M2DNA. The wrapper does not modify the downloaded CGRclust repository and does not depend on checkpoint-saving code being present in that repository.

## 5. Training the M2DNA Dual-Stream Architecture

After obtaining the offline visual checkpoint, enable the nucleotide language-model stream with `--use_llm`. The visual checkpoint must be loaded first and the visual encoder must remain frozen during M2DNA training. This preserves the global topological prior learned by CGRclust and allows the textual branch to learn complementary local sequential information.

```bash
python src/cluster.py \
    --dataset 01_Cypriniformes.fasta \
    --k 6 \
    --use_llm \
    --finetune_llm \
    --llm_path InstaDeepAI/nucleotide-transformer-500m-human-ref \
    --pretrained_visual checkpoints/01_Cypriniformes/best_visual_model.pth \
    --freeze_visual \
    --num_epochs 20 \
    --batch_size 40 \
    --lr 7e-5 \
    --lora_lr 1e-4 \
    --weight_decay 1e-4 \
    --temp_ins 0.1 \
    --temp_clu 1.0 \
    --weight 0.7
```

The M2DNA optimization procedure consists of two stages:

1. **Offline visual representation learning.** The CNN encoder is trained with CGRclust on weak and strong FCGR views and then frozen.
2. **Asymmetric dual-stream fine-tuning.** The frozen visual stream extracts topological features, while the nucleotide transformer processes non-overlapping k-mer tokens. Mean pooling produces the textual representation, which is projected to the visual feature dimension. The Adaptive Fusion Module concatenates the projected textual feature with the PMD-processed visual feature, estimates an element-wise sigmoid gate, and computes their weighted sum.

LoRA adapters are inserted into the query, key, and value projections of the nucleotide transformer. In accordance with the paper methodology, only the LoRA parameters and the trainable fusion/projection modules are optimized in this stage; the original nucleotide-transformer weights and the visual encoder remain frozen.

PMD is applied to the entire visual branch rather than individual neurons. With probability `p=0.3`, the visual feature is masked during training, forcing the textual stream to learn discriminative local sequence information independently. PMD is disabled during inference, where the full visual representation is restored.

## 6. Main Arguments

| Argument | Default | Description |
|---|---:|---|
| `--dataset` | `01_Cypriniformes.fasta` | FASTA file under `CGRclust/data/` during visual pretraining |
| `--k` | `6` | k-mer size and FCGR resolution parameter |
| `--weak_mutation_rate` | `1e-4` | Weak mutation rate |
| `--strong_mutation_rate` | `1e-2` | Strong mutation rate |
| `--number_of_pairs` | `1` | Augmented pairs per sequence |
| `--number_of_models` | `5` | Number of independently trained models |
| `--batch_size` | `40` | Training batch size |
| `--num_epochs` | `20` | Joint-training epochs |
| `--embedding_dim` | `512` | Visual representation dimension |
| `--feature_dim` | `128` | Instance representation dimension |
| `--weight` | `0.7` | Balancing coefficient for the Twin Contrastive Loss |
| `--use_llm` | disabled | Enables the nucleotide sequence stream |
| `--finetune_llm` | disabled | Updates the LoRA adapters in the textual stream |
| `--pretrained_visual` | `None` | Visual checkpoint path |
| `--freeze_visual` | disabled | Freezes the loaded visual stream |

## 7. Evaluation

Evaluation is performed automatically after each training run using clustering accuracy with Hungarian matching. When `--number_of_models` is greater than one, the script additionally reports hard-voting and soft-voting ensemble accuracy across the independently trained models.

Example:

```bash
python src/cluster.py \
    --dataset 01_Cypriniformes.fasta \
    --use_llm \
    --finetune_llm \
    --pretrained_visual checkpoints/01_Cypriniformes/best_visual_model.pth \
    --freeze_visual \
    --number_of_models 5
```

The final console output reports the accuracy of each model, hard voting, soft voting, and the corresponding confusion matrices.

## 8. Reproducibility Notes

- Run commands from the repository root; the visual-pretraining wrapper reads `CGRclust/data/<dataset>`.
- Use the same `--k`, augmentation settings, batch size, and temperature values when comparing models.
- Set `--random_seed` and retain the default per-model seed offset when reproducing an ensemble.
- The visual checkpoint must be compatible with the visual dimensions used in the dual-stream run.
- GPU training is recommended for the nucleotide transformer; CPU execution is supported but may be considerably slower.
