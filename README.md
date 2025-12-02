# HRL Master Project

A hierarchical reinforcement learning framework for robotic manipulation tasks using a slighly modified version of [CALVIN Environment](https://github.com/mees/calvin). This project implements skill-based learning with [TAPAS](https://github.com/robot-learning-freiburg/TAPAS) (Task-Parameterized Gaussian Mixture Models) and supports various agent architectures including GNN-based and baseline approaches.

## Requirements

- Python >= 3.10
- PyTorch 2.1.0
- CUDA (optional, for GPU acceleration)
- Conda

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/JRGruhnert/master-project.git
cd master-project
```

### 2. Create a Virtual Environment (Recommended)

The project was tested with python version 3.10

```bash
conda create -n my_env python=3.10
conda activate my_env
```

### 3. Run the Installation Script

The installation script will automatically clone and install all required dependencies:

```bash
chmod +x install.sh
sh install.sh
```

This script installs:

- **calvin_env_modified**: Modified CALVIN environment
- **riepybdlib**: Riemannian geometry library for robot learning
- **TapasCalvin**: Modified version of TAPAS
- **PyTorch Geometric**: Graph neural network extensions
- **HRL project**: All python packages for the project

### 4. Set Environment Variables

Before running any commands, source the environment variables:

```bash
source venv.sh
```

## Project Structure

```
master-project/
├── conf/                   # Configuration files
│   ├── baseline/           # Baseline agent configurations
│   ├── gnn/                # GNN agent configurations
│   ├── human/              # Human control configurations
│   ├── common/             # Shared configurations
│   └── sweeps/             # Hyperparameter sweep configs
├── data/
│   └── skills/             # Skills are loaded from this directory
├── dependencies/           # External dependencies
│   ├── calvin_env_modified/
│   ├── riepybdlib/
│   └── tapas/
├── scripts/                # Entry point scripts
│   ├── train.py            # Main training script
│   ├── eval_skills.py      # Skill evaluation
│   ├── plot.py             # Plotting utilities
│   └── debug.py            # Debugging utilities
├── src/                    # Source code
│   ├── agents/             # Agent implementations
│   ├── environments/       # Environment wrappers
│   ├── experiments/        # Experiment definitions
│   ├── modules/            # Core modules (buffer, logger, etc.)
│   ├── networks/           # Neural network architectures
│   └── skills/             # Skill management
├── results/                # Training results
└── wandb/                  # Weights & Biases logs
```

## Usage

Use the command line to execute a script together with a specified configuration file:

### Training an Agent

Executes the training loop.

```bash
####---EXAMPLES---###
train --conf conf/gnn/file.py # Train with GNN agent
train --conf conf/baseline/file.py # Train with baseline
train --conf conf/human/file.py # Seperate skill selection
```

### Evaluating Skills

Evaluates the skill library.

```bash
eval-skills --conf <config_file>
```

### Plotting Results

Plot the results of training or evaluation.

```bash
# General plotting
plot --conf conf/plots/<config>.py

# Plot skill trajectories
plot-skills --conf <config_file>
```

## Configuration

Configuration files are Python files located in the `conf/` directory. They use OmegaConf for hierarchical configuration management.

### Example Configuration Structure

```python
from src.agents.ppo.ppo import PPOAgentConfig
from src.environments.calvin import CalvinEnvironmentConfig

config = {
    "agent": PPOAgentConfig(...),
    "environment": CalvinEnvironmentConfig(...),
    "experiment": ExperimentConfig(...),
    "evaluator": EvaluatorConfig(...),
    "buffer": BufferConfig(...),
    "logger": LoggerConfig(...),
    "storage": StorageConfig(...),
}
```

### Configuration Directories

| Directory        | Description                               |
| ---------------- | ----------------------------------------- |
| `conf/baseline/` | Baseline agent configurations             |
| `conf/gnn/`      | Graph Neural Network agent configurations |
| `conf/human/`    | Human-in-the-loop control                 |
| `conf/common/`   | Shared configuration components           |
| `conf/sweeps/`   | Hyperparameter sweep definitions          |

## Available Commands

After installation, the following commands are available:

| Command       | Description                                                              |
| ------------- | ------------------------------------------------------------------------ |
| `train`       | Train an agent with the specified configuration                          |
| `plot`        | Visualize training runs                                                  |
| `eval-skills` | Evaluate pre-trained skills                                              |
| `plot-skills` | Visualize evaluation results                                             |
| `sweep`       | Train an agent with the specified sweep values for hyperparameter tuning |

## Available Skills

The project includes 22 pre-trained manipulation skills:

- **Drawer**: CloseDrawer, OpenDrawer, CloseDrawerBack, OpenDrawerBack
- **Button**: PressButton, PressButtonBack
- **Slide**: OpenSlide, CloseSlide, OpenSlideBack, CloseSlideBack
- **Red Block**: GrabRedTable, PlaceRedTable, GrabRedDrawer, PlaceRedDrawer
- **Pink Block**: GrabPinkTable, PlacePinkTable, GrabPinkDrawer, PlacePinkDrawer
- **Blue Block**: GrabBlueTable, PlaceBlueTable, GrabBlueDrawer, PlaceBlueDrawer

## Troubleshooting

### Common Issues

1. **EGL/OpenGL errors**: Ensure you have proper GPU drivers installed or use software rendering:

   ```bash
   export PYOPENGL_PLATFORM=egl
   ```

2. **PYTHONPATH issues**: Make sure to source `venv.sh` before running commands.

3. **TAPAS Config mismatch warnings**: These are expected when loading pre-trained skills with different configurations. The system will attempt to adapt automatically.

## Acknowledgments

This project is built upon the following projects:

- [CALVIN Environment](https://github.com/mees/calvin)
- [TAPAS](https://github.com/robot-learning-freiburg/TAPAS)
- [riepybdlib](https://github.com/vonHartz/riepybdlib)
