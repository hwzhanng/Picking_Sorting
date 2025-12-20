# Picking_Sorting: JAX/MJX GPU-Accelerated RL Training

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.20+-orange.svg)](https://github.com/google/jax)
[![MuJoCo MJX](https://img.shields.io/badge/MuJoCo_MJX-3.0+-green.svg)](https://mujoco.readthedocs.io/en/stable/mjx.html)
[![Flax](https://img.shields.io/badge/Flax-0.8.0+-purple.svg)](https://github.com/google/flax)

## 📋 Overview

**Picking_Sorting** is a GPU-accelerated reinforcement learning framework for agricultural picking robots using **JAX/MJX** for massively parallel simulation.

### 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **GPU Parallel Simulation** | 1000+ environments on single GPU via MuJoCo MJX |
| **Pure JAX Implementation** | All components JIT-compiled for maximum performance |
| **Two-Stage Training** | Stage 1: Navigation + Arm, Stage 2: Grasping |
| **AVP Technology** | Asymmetric Value Propagation for coordinated learning |
| **Site-based FK** | Precise grasp point tracking (not body center) |

### 🤖 Robot Platform

- **Mobile Base**: Ranger Mini V2 (4-wheel steering)
- **Arm**: xArm6 6-DOF manipulator  
- **Hand**: LEAP Hand 16-joint dexterous hand
- **Sensors**: Wrist depth camera, tactile sensors

---

## 🛠️ Installation

### Option 1: pip (Recommended for Cloud)

```bash
# 1. Clone repository
git clone https://github.com/hwzhanng/Picking_Sorting.git
cd Picking_Sorting

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install JAX with CUDA support
# For CUDA 12.x:
pip install --upgrade "jax[cuda12]"
# For CUDA 11.x:
pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install package
pip install -e .
```

### Option 2: Conda

```bash
# 1. Clone repository
git clone https://github.com/hwzhanng/Picking_Sorting.git
cd Picking_Sorting

# 2. Create environment from yaml
conda env create -f environment.yml
conda activate picking_jax

# 3. Install JAX with CUDA (conda doesn't have official JAX packages)
pip install --upgrade "jax[cuda12]"

# 4. Install package
pip install -e .
```

### Option 3: Docker (Cloud/Server)

```bash
# Pull NVIDIA JAX container
docker pull nvcr.io/nvidia/jax:latest

# Run with GPU access
docker run --gpus all -it -v $(pwd):/workspace nvcr.io/nvidia/jax:latest

# Inside container:
cd /workspace
pip install -r requirements.txt
pip install -e .
```

### Verify Installation

```bash
# Test JAX GPU detection
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

# Test MJX
python -c "import mujoco.mjx as mjx; print('MJX available')"

# Test environment
python test_env_jax.py
```

---

## 🏃 Quick Start

### Training Stage 2 (Grasp) → Stage 1 (Navigation)

> ⚠️ **Important**: Train Stage 2 first, then Stage 1 uses its Critic for AVP rewards.

```bash
# Step 1: Train Stage 2 (Grasping)
python train_stage2_jax.py \
    --num_envs 1024 \
    --total_timesteps 10_000_000 \
    --seed 42

# Step 2: Export Stage 2 Critic for AVP
cp outputs/stage2_best.pkl assets/checkpoints/avp/stage2_critic.pkl

# Step 3: Train Stage 1 (Navigation + Arm)
python train_stage1_jax.py \
    --num_envs 1024 \
    --total_timesteps 25_000_000 \
    --avp_checkpoint assets/checkpoints/avp/stage2_critic.pkl \
    --seed 42
```

### Evaluation

```bash
# Evaluate Stage 1
python eval_jax.py \
    --stage 1 \
    --checkpoint outputs/stage1_best.pkl \
    --num_episodes 100

# Evaluate Stage 2
python eval_jax.py \
    --stage 2 \
    --checkpoint outputs/stage2_best.pkl \
    --num_episodes 100

# Visualization (CPU MuJoCo - MJX has no viewer)
python visualize.py \
    --checkpoint outputs/stage1_best.pkl \
    --stage 1
```

---

## 📁 Project Structure

```
Picking_Sorting/
├── train_stage1_jax.py          # Stage 1 JAX training entry
├── train_stage2_jax.py          # Stage 2 JAX training entry
├── eval_jax.py                  # Evaluation script
├── visualize.py                 # CPU visualization (MJX has no viewer)
├── test_env_jax.py              # Environment test
│
├── gym_dcmm/
│   ├── agents/
│   │   └── MujocoDcmm.py        # MJX robot wrapper + FK functions
│   │
│   ├── envs/
│   │   ├── stage1/
│   │   │   └── RewardManagerStage1.py  # JAX reward computation
│   │   └── stage2/
│   │       └── RewardManagerStage2.py  # JAX reward computation
│   │
│   ├── algs/ppo_dcmm/
│   │   └── stage2/
│   │       └── ModelsStage2.py  # Flax ActorCritic
│   │
│   └── utils/
│       ├── pid.py               # JAX PID controller
│       ├── quat_utils.py        # JAX quaternion operations
│       ├── ik_pkg/
│       │   └── ik_base.py       # JAX mobile base IK
│       └── jax_migration_utils.py  # Conversion utilities
│
├── assets/
│   ├── checkpoints/avp/         # AVP pretrained weights
│   ├── urdf/                    # MuJoCo robot models
│   └── meshes/                  # 3D meshes
│
├── configs/                     # Hydra configuration
├── requirements.txt             # pip dependencies
└── environment.yml              # Conda environment
```

---

## 🔧 JAX/MJX Architecture

### Key Differences from CPU Version

| Aspect | CPU (Old) | GPU/JAX (New) |
|--------|-----------|---------------|
| **Physics** | `mujoco.mj_step()` | `mjx.step()` |
| **Parallelism** | Python multiprocessing | `jax.vmap` over environments |
| **Neural Networks** | PyTorch | Flax |
| **State Management** | Class attributes | Explicit state passing |
| **Random Numbers** | `numpy.random` | `jax.random.PRNGKey` |
| **Conditionals** | Python `if/else` | `jax.lax.cond` |

### Stateless Design Pattern

```python
# Old (CPU): Class with mutable state
class PIDController:
    def __init__(self):
        self.integral = 0.0
    
    def step(self, error):
        self.integral += error  # Mutation!
        return self.Kp * error + self.Ki * self.integral

# New (JAX): Pure function with explicit state
@jax.jit
def pid_step(error, state, params):
    new_integral = state.integral + error
    output = params.Kp * error + params.Ki * new_integral
    new_state = state._replace(integral=new_integral)
    return output, new_state  # Return new state explicitly
```

### Site-based FK (Important for Grasping!)

```python
# WRONG: Body position = wrist center, NOT grasp point
ee_pos = mx_data.xpos[body_id]  # ❌ Misses by wrist-to-finger offset

# CORRECT: Site position = user-defined grasp point
from gym_dcmm.agents.MujocoDcmm import (
    create_site_id_mapping, get_site_position, compute_site_to_site_distance
)

# Define in MJCF: <site name="grasp_site" pos="0 0 0.05"/>
site_ids = create_site_id_mapping(mj_model, {
    'ee': 'grasp_site',
    'target': 'tomato_site'
})

# Use site positions for reward computation
ee_pos = get_site_position(mx_data, site_ids.ee_site_id)  # ✅ Correct
dist = compute_site_to_site_distance(mx_data, site_ids.ee_site_id, site_ids.target_site_id)
```

---

## 📊 Performance

### Environments per Second (Single RTX 4090)

| Configuration | Envs | FPS |
|--------------|------|-----|
| MJX + JAX | 1024 | ~50,000 |
| MJX + JAX | 4096 | ~100,000 |
| CPU (old) | 16 | ~800 |

### GPU Memory Usage

| Envs | VRAM |
|------|------|
| 256 | ~4 GB |
| 1024 | ~12 GB |
| 4096 | ~24 GB |

---

## ⚙️ Configuration

### Training Parameters

```bash
python train_stage1_jax.py \
    --num_envs 1024           # Parallel environments
    --total_timesteps 25e6    # Total training steps
    --learning_rate 3e-4      # Learning rate
    --gamma 0.99              # Discount factor
    --gae_lambda 0.95         # GAE lambda
    --clip_epsilon 0.2        # PPO clip range
    --entropy_coef 0.01       # Entropy bonus
    --batch_size 4096         # Minibatch size
    --num_epochs 4            # PPO epochs per update
    --seed 42                 # Random seed
    --device cuda             # Device (cuda/cpu)
    --wandb_project my_proj   # WandB project (optional)
```

### Environment Configuration

Edit `configs/env/DcmmCfg.py`:

```python
class mjx_config:
    # MJX-specific settings
    dt = 0.002              # Physics timestep
    substeps = 4            # Substeps per control step
    
class curriculum:
    # Curriculum learning
    stage1_steps = 2e6
    collision_penalty_start = -0.1
    collision_penalty_end = -2.0

class avp:
    # Asymmetric Value Propagation
    enabled = True
    lambda_weight_start = 0.8
    lambda_weight_end = 0.2
    gate_distance = 1.5
```

---

## 🐛 Troubleshooting

### JAX GPU Not Detected

```bash
# Check CUDA
nvidia-smi

# Reinstall JAX with correct CUDA version
pip uninstall jax jaxlib
pip install --upgrade "jax[cuda12]"  # or cuda11
```

### Out of Memory

```bash
# Reduce parallel environments
python train_stage1_jax.py --num_envs 256

# Enable memory preallocation
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
```

### MJX Import Error

```bash
# MJX requires mujoco >= 3.0
pip install --upgrade mujoco>=3.0
```

### NaN in Training

1. Check learning rate (try 1e-4)
2. Gradient clipping (--max_grad_norm 0.5)
3. Check reward scaling

---

## 📚 References

- [MuJoCo MJX Documentation](https://mujoco.readthedocs.io/en/stable/mjx.html)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)

---

## 📄 License

MIT License

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request
