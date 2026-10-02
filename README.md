# BiFlow

BiFlow: Bidirectional Knowledge Flow for Dynamic Graph Learning

The main entry point is `main_gpu.py`.

## Features

- Incremental training over graph snapshots
- Node classification evaluation (ACC)
- Node clustering evaluation (NMI, ARI)
- Non-overlapping community detection evaluation (NMI, ARI)
- Continual-learning metrics including PM and FM

## Project Structure

- `main_gpu.py`: main training and evaluation pipeline (GPU version)
- `data/dataset_gpu.py`: dynamic snapshot dataset builder
- `models/progressive_sage_gpu.py`: progressive GraphSAGE backbone
- `models/meta_loss.py`: backward transfer / meta-loss optimization
- `selective_modeling/`: selective temporal modeling modules (Mamba-based)
- `utils/get_params.py`: runtime arguments

## Requirements

Use Python 3.8+ and install common dependencies:

- `torch`
- `torch-geometric`
- `numpy`
- `scikit-learn`

Example:

```bash
pip install torch torch-geometric numpy scikit-learn
```

## Data Format

Each dataset should be placed under `data/<dataset_name>/` and include:

- `attributes`: node feature matrix
- `labels`: node ids and labels
- `train_nodes`, `val_nodes`, `test_nodes`
- `stream_edges/`: one file per snapshot (`0`, `1`, `2`, ......)

Existing example in this repository: `data/highSchool/`.

## Training

Run:

```bash
python main_gpu.py --dataset_name highSchool
```

Useful arguments (from `utils/get_params.py`):

- `--dataset_name`: dataset key (for example `highSchool`, `cora`, `dblp`)
- `--edge_type`: snapshot edge folder (default `stream_edges`)
- `--init_epochs`, `--rectify_epochs`
- `--ParameterE`, `--MemoryAnchor`, `--BackKF`
- `--cuda_enable`

## Output

- Trained checkpoints are stored in `save_models/<dataset_name>/`.
- Console output reports task-wise and average metrics.

