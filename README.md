# Picking_Sorting: 基于AVP的两阶段农业采摘强化学习项目

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.2.3+-green.svg)](https://mujoco.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-red.svg)](https://pytorch.org/)

## 📋 项目概述

**Picking_Sorting** 是一个基于深度强化学习的农业采摘机器人控制项目。该项目训练移动操作机器人（移动底盘 + 6自由度机械臂 + 16DOF灵巧手）在复杂农业场景中自主导航至目标果实并进行精确采摘。

### 🎯 核心特性

| 特性 | 描述 |
|------|------|
| **两阶段训练** | Stage 1 训练底盘+机械臂接近目标，Stage 2 训练灵巧手抓取 |
| **AVP技术** | Asymmetric Value Propagation，使用 Stage 2 Critic 为 Stage 1 提供"可抓取性"奖励信号 |
| **动态课程学习** | 课程长度2M步，渐进式调整碰撞惩罚（-0.1→-2.0）和朝向精度要求（1.0→1.5次方） |
| **关节空间控制** | 直接输出关节角度增量（Δθ），避免IK不稳定性 |
| **域随机化** | 深度噪声、光照随机、地面纹理、物体形状/质量随机化 |
| **视觉感知** | 84×84深度图输入，支持模拟RealSense D435i噪声 |

### 🤖 机器人平台

- **移动底盘**: Ranger Mini V2 双阿克曼转向底盘
- **机械臂**: xArm6 6自由度机械臂
- **灵巧手**: LEAP Hand 16关节灵巧手（12个可控自由度）
- **传感器**: 腕部深度相机、手指触觉传感器（4个）

---

## 🚀 快速开始

### 环境要求

- Python 3.8
- CUDA 11.8+ (推荐)
- GPU: NVIDIA RTX 3080 或更高 (推荐)
- 内存: 16GB+ RAM

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/hwzhanng/Picking_Sorting.git
cd Picking_Sorting

# 2. 创建conda环境
conda create -n picking python=3.8
conda activate picking

# 3. 安装PyTorch (根据CUDA版本选择)
pip install torch==2.4.1 torchvision==0.20.0 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# 4. 安装项目依赖
pip install -r requirements.txt

# 5. 安装项目包
pip install -e .
```

### 验证安装

```bash
# 测试环境是否正常
python test_env.py

# 查看MuJoCo渲染
python train_stage1.py test=True num_envs=1 viewer=True
```

---

## 📊 输入输出维度规格

### Stage 1 (Tracking - 底盘+机械臂接近)

| 类型 | 维度 | 组成 |
|------|------|------|
| **观测 (State)** | 15 | 底盘速度(2) + 末端位置(3) + 末端四元数(4) + 末端速度(3) + 目标位置(3) |
| **观测 (Depth)** | 84×84 | 深度图像，单通道，归一化至0-255 |
| **动作** | 8 | 底盘速度(2) + 机械臂关节增量(6) |

**动作范围:**
- 底盘: [-4, 4] m/s (线速度)
- 机械臂: [-0.05, 0.05] rad/step (关节增量)

### Stage 2 (Catching - 灵巧手抓取)

| 类型 | 维度 | 组成 |
|------|------|------|
| **观测 (State)** | 35 | 末端位置(3) + 末端四元数(4) + 末端速度(3) + 机械臂关节(6) + 目标位置(3) + 手部关节(12) + 触觉(4) |
| **观测 (Depth)** | 84×84 | 深度图像，单通道，归一化至0-255 |
| **动作** | 20 | 底盘(2, 锁定为0) + 机械臂关节增量(6) + 手部关节增量(12) |

**动作范围:**
- 底盘: 锁定为0 (Stage 2不控制底盘)
- 机械臂: [-0.025, 0.025] rad/step (关节增量)
- 手部: [-0.05, 0.05] rad/step (关节增量)

---

## 🎯 奖励函数详解

### Stage 1 奖励函数 (`RewardManagerStage1.py`)

| 奖励分量 | 公式 | 权重 | 说明 |
|----------|------|------|------|
| **手臂到达** | `2.0 × (1 - tanh(3d_arm))` | 2.0 | 机械臂末端到目标距离（基座坐标系） |
| **全局到达** | `0.5 × (1 - tanh(2d_ee))` | 0.5 | 末端到目标全局距离 |
| **底盘定位** | `exp(-5(d_base - 0.8)²)` | 1.0 | 鼓励底盘保持0.8m最优距离 |
| **手臂运动** | `0.5 × tanh(3 × joint_dev)` | 0.5 | 奖励手臂关节偏离初始位置 |
| **手臂动作** | `0.2 × ‖action_arm‖` | 0.2 | 奖励使用手臂控制 |
| **朝向对齐** | `max(0, align)^power × 2` | 动态 | 掌心朝向目标（power: 1.0→1.5） |
| **接触奖励** | `10 - 4 × impact_speed` | 10.0 | 轻触目标，惩罚高速撞击 |
| **树干碰撞** | `curriculum_penalty` | -0.1→-2.0 | 课程学习渐进惩罚 |
| **树叶碰撞** | `-0.5 × (1 + velocity)` | -0.5 | 速度相关轻微惩罚 |
| **动作平滑** | `-0.02 × ‖Δaction‖` | -0.02 | 惩罚动作剧烈变化 |
| **控制正则** | `-0.005 × ‖base‖ - 0.001 × ‖arm‖` | 动态 | 底盘惩罚大于手臂 |
| **AVP奖励** | `λ × Critic(virtual_obs)` | 0.2→0.8 | Stage 2 Critic值估计 |

### Stage 2 奖励函数 (`RewardManagerStage2.py`)

| 奖励分量 | 公式 | 权重 | 说明 |
|----------|------|------|------|
| **到达奖励** | `2.0 × exp(-2d_ee)` | 2.0 | 末端到目标指数衰减 |
| **距离里程碑** | `+0.5/+1.0/+1.5/+2.0` | 累加 | d<0.3m/0.15m/0.08m/0.05m |
| **朝向奖励** | `max(0, align) × 1.0` | 1.0 | 仅d<0.5m时计算 |
| **抓取奖励** | `1.0 + 0.5×fingers + bonus` | 动态 | 接触数量+力范围bonus |
| **扰动测试** | `±10.0 / -5.0` | ±10.0 | 抵抗0.5-1.5N随机扰动 |
| **冲击惩罚** | `-min(2(v-0.3), 3)` | -3.0 | v>0.3m/s时惩罚 |
| **成功奖励** | `+20.0` | 20.0 | 稳定抓取1秒 |
| **碰撞惩罚** | `-2.0` | -2.0 | 非成功终止 |
| **植物碰撞** | `curriculum_penalty` | 动态 | 茎秆>叶片 |

---

## 🎓 训练指南

> ⚠️ **重要**: 正确的训练顺序是 **先训练 Stage 2，再训练 Stage 1**。Stage 1 会加载预训练的 Stage 2 Critic 进行 AVP 奖励计算。

### Step 1: Stage 2 训练 (灵巧手抓取)

**独立训练**灵巧手在近距离完成抓取任务，训练完成后导出 Critic 用于 AVP。

```bash
# 基础训练 (推荐配置)
python train_stage2.py num_envs=8 device_id=0

# 使用更多并行环境加速 (需要更大显存)
python train_stage2.py num_envs=16 device_id=0

# 从检查点恢复训练
python train_stage2.py checkpoint_catching="outputs/Dcmm_Catch/2025-12-19/nn/best_reward_XXX.pth"

# 指定随机种子进行可复现训练
python train_stage2.py seed=42 output_name=Dcmm_Catch_seed42
```

**训练完成后**，复制最佳模型到 AVP 目录：
```bash
cp outputs/Dcmm_Catch/.../nn/best_reward_XXX.pth assets/checkpoints/avp/stage2_critic.pth
```

### Step 2: Stage 1 训练 (底盘+机械臂接近)

使用预训练的 **Stage 2 Critic** (通过 AVP) 引导底盘和机械臂找到便于抓取的位置。

```bash
# 基础训练 (AVP 自动加载 Stage 2 Critic)
python train_stage1.py num_envs=8 device_id=0

# 关闭 AVP (消融实验基线)
python train_stage1.py avp_enabled=False output_name=Dcmm_NoAVP

# 从检查点恢复训练
python train_stage1.py checkpoint_tracking="outputs/Dcmm/2025-12-19/nn/best_reward_XXX.pth"
```

### 关键配置参数

**Stage 1 配置** (`configs/config_stage1.yaml`):

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_envs` | 32 | 并行环境数（建议8-16，根据CPU核心数调整） |
| `seed` | -1 | 随机种子（-1=自动随机） |
| `device_id` | 0 | GPU设备ID |
| `avp_enabled` | True | AVP开关 |
| `train.ppo.max_agent_steps` | 25M | 最大训练步数 |
| `train.ppo.learning_rate` | 3e-4 | 学习率 |
| `train.ppo.horizon_length` | 512 | 采样长度 |
| `train.ppo.minibatch_size` | 512 | Mini-batch大小 |
| `train.ppo.mini_epochs` | 6 | PPO更新轮数 |
| `train.ppo.entropy_coef` | 0.01 | 熵正则系数 |
| `train.ppo.gamma` | 0.99 | 折扣因子 |
| `train.ppo.tau` | 0.95 | GAE参数 |

**Stage 2 配置** (`configs/config_stage2.yaml`):

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_envs` | 32 | 并行环境数（建议8-16，根据CPU核心数调整） |
| `train.ppo.img_dim` | [224, 224] | 深度图尺寸 |
| `train.ppo.action_catch_denorm` | [0.0, 0.025, 0.05] | 动作反归一化 [底盘, 手臂, 手部] |

---

## 🧪 实验与测试

### 可视化验证 (单窗口)

> ⚠️ **重要**: 可视化时请使用 `num_envs=1` 避免多窗口导致系统卡死

```bash
# Stage 1 可视化训练 (AVP 开启)
python train_stage1.py num_envs=1 viewer=True

# Stage 1 可视化测试 (加载checkpoint)
python train_stage1.py test=True num_envs=1 viewer=True \
    checkpoint_tracking="outputs/Dcmm/2025-12-19/nn/best_reward_XXX.pth"

# Stage 2 可视化测试
python train_stage2.py test=True num_envs=1 viewer=True \
    checkpoint_catching="outputs/Dcmm_Catch/2025-12-19/nn/best_reward_XXX.pth"
```

### 消融实验

```bash
# AVP消融 - 基线 (无AVP)
python train_stage1.py avp_enabled=False output_name=Ablation_NoAVP seed=42

# AVP消融 - 完整方法 (有AVP)
python train_stage1.py avp_enabled=True output_name=Ablation_WithAVP seed=42

# 不同AVP权重实验
python train_stage1.py avp_enabled=True output_name=AVP_lambda_0.3
# 修改DcmmCfg.py中 avp.lambda_weight_start 和 avp.lambda_weight_end
```

### 多种子实验

```bash
# 批量训练不同种子
for seed in 42 123 456 789 1000; do
    python train_stage2.py seed=$seed output_name=Dcmm_Catch_seed$seed &
done

# 等待所有训练完成
wait
```

### WandB 日志配置

```bash
# 禁用 WandB (本地调试)
python train_stage2.py wandb_mode=disabled

# 离线模式 (稍后同步)
python train_stage2.py wandb_mode=offline

# 在线模式 (默认) - 设置项目名
python train_stage2.py wandb_project=Picking_Sorting_Exp wandb_entity=your_team output_name=Exp001
```

### 评估模式

```bash
# Stage 1 评估 (100个episodes)
python train_stage1.py test=True num_envs=1 \
    checkpoint_tracking="outputs/Dcmm/.../best_reward_XXX.pth"

# Stage 2 评估
python train_stage2.py test=True num_envs=1 \
    checkpoint_catching="outputs/Dcmm_Catch/.../best_reward_XXX.pth"
```

---

## 🔧 AVP 配置详解

**AVP (Asymmetric Value Propagation)** 使用预训练的 Stage 2 Critic 为 Stage 1 提供"可抓取性"奖励信号。

### 配置位置 (`configs/env/DcmmCfg.py`)

```python
class avp:
    enabled = True                    # 主开关 (False=关闭AVP)
    lambda_weight_start = 0.8         # 早期训练权重（强AVP引导）
    lambda_weight_end = 0.2           # 后期训练权重（依赖原始奖励）
    gate_distance = 1.5               # 距离门限 (仅在此距离内计算AVP)
    checkpoint_path = "assets/checkpoints/avp/stage2_critic.pth"
    ready_pose = np.array([0.0, 0.0, 0.0, 1.8, 0.0, -0.785])  # 虚拟就绪姿态
    state_dim = 35                    # Stage 2 状态维度
    img_size = 84                     # 深度图尺寸
```

### AVP 工作原理

```
Stage 1 当前状态
      │
      ▼
构造虚拟观测:
┌─────────────────────────────────┐
│ • 虚拟手臂姿态 (ready_pose)      │
│ • 真实物体位置                   │
│ • 真实深度图                     │
│ • 虚拟手部张开状态               │
└─────────────────────────────────┘
      │
      ▼
Stage 2 Critic(虚拟观测) → value_estimate
      │
      ▼
AVP奖励 = λ(t) × clip(value_estimate, -5, 5)
      │
      ▼
总奖励 = 原始Stage1奖励 + AVP奖励
```

### 更新 AVP 权重

当训练出更好的 Stage 2 模型时：
```bash
cp outputs/Dcmm_Catch/.../nn/best_reward_XXX.pth assets/checkpoints/avp/stage2_critic.pth
```

---

## 🏗️ 神经网络架构

### Stage 1 网络 (`ModelsStage1.py`)

```
输入:
├── State (15维) → MLP [256, 128] → 特征
└── Depth (84×84) → CNN → 256维特征
                          │
                          ▼
              Concatenate → [特征 + 256]
                          │
           ┌──────────────┴──────────────┐
           ▼                             ▼
      Actor MLP                    Critic MLP
      [256, 128]                   [256, 128]
           │                             │
           ▼                             ▼
        μ (8维)                     Value (1维)
        σ (8维)
```

**CNN架构:**
- Conv2d(1→32, k=3, s=1, p=1) + ReLU + MaxPool(2)
- Conv2d(32→64, k=3, s=1, p=1) + ReLU + MaxPool(2)
- Conv2d(64→64, k=3, s=1, p=1) + ReLU + MaxPool(2)
- Flatten → Linear → 256

### Stage 2 网络 (`ModelsStage2.py`)

```
输入: Flattened [State(35) + Depth(84×84=7056)] = 7091维
            │
            ├── State → Actor MLP [256, 128] → μ (20维), σ (20维)
            │
            └── State → Value MLP [256, 128] ─┬─→ Concat
                Depth → DepthCNN → 256维 ────┘    │
                                                  ▼
                                           Value Head → 1维
```

**DepthCNN架构:**
- Conv2d(1→32, k=8, s=4) + ReLU
- Conv2d(32→64, k=4, s=2) + ReLU
- Conv2d(64→32, k=3, s=1) + ReLU
- Flatten → Linear(32×7×7 → 256) + ReLU

---

## 📁 项目结构

```
Picking_Sorting/
├── train_stage1.py                    # Stage 1 训练入口
├── train_stage2.py                    # Stage 2 训练入口
├── test_env.py                        # 环境测试脚本
├── setup.py                           # 包安装配置
├── requirements.txt                   # Python依赖
├── environment.yml                    # Conda环境配置
│
├── configs/
│   ├── config_stage1.yaml             # Stage 1 主配置
│   ├── config_stage2.yaml             # Stage 2 主配置
│   ├── env/
│   │   └── DcmmCfg.py                 # 环境参数 + AVP配置 + 课程学习
│   └── train/
│       ├── stage1/PPO_Stage1.yaml     # Stage 1 PPO超参数
│       └── stage2/PPO_Stage2.yaml     # Stage 2 PPO超参数
│
├── gym_dcmm/
│   ├── __init__.py                    # 注册Gym环境
│   ├── agents/
│   │   └── MujocoDcmm.py              # 机器人MuJoCo模型封装
│   ├── envs/
│   │   ├── stage1/
│   │   │   ├── DcmmVecEnvStage1.py    # Stage 1 环境
│   │   │   └── RewardManagerStage1.py # Stage 1 奖励管理 (含AVP)
│   │   ├── stage2/
│   │   │   ├── DcmmVecEnvStage2.py    # Stage 2 环境
│   │   │   └── RewardManagerStage2.py # Stage 2 奖励管理
│   │   ├── observation_manager.py     # 观测处理
│   │   ├── control_manager.py         # 控制管理
│   │   ├── randomization_manager.py   # 域随机化
│   │   ├── render_manager.py          # 渲染管理
│   │   └── constants.py               # 常量定义
│   ├── algs/ppo_dcmm/
│   │   ├── stage1/
│   │   │   ├── PPO_Stage1.py          # Stage 1 PPO算法
│   │   │   └── ModelsStage1.py        # Stage 1 神经网络
│   │   ├── stage2/
│   │   │   ├── PPO_Stage2.py          # Stage 2 PPO算法
│   │   │   └── ModelsStage2.py        # Stage 2 神经网络
│   │   ├── experience.py              # 经验缓冲
│   │   └── utils.py                   # 工具函数 (RunningMeanStd等)
│   └── utils/
│       ├── quat_utils.py              # 四元数工具
│       └── util.py                    # 通用工具
│
├── assets/
│   ├── checkpoints/avp/               # AVP预训练权重
│   │   └── stage2_critic.pth
│   ├── urdf/                          # MuJoCo机器人模型
│   ├── meshes/                        # 3D网格模型
│   ├── textures/                      # 纹理贴图
│   └── objects/                       # 物体模型
│
├── outputs/                           # 训练输出目录
│   ├── Dcmm/                          # Stage 1 输出
│   └── Dcmm_Catch/                    # Stage 2 输出
│
└── wandb/                             # WandB日志目录
```

---

## 📈 课程学习配置

课程学习参数位于 `configs/env/DcmmCfg.py`:

```python
class curriculum:
    # Stage 1 课程学习步数
    stage1_steps = 2e6      # 2M步完成课程
    stage2_steps = 10e6     # Stage 2 扩展课程
    
    # 碰撞惩罚渐进 (从轻到重)
    collision_stem_start = -0.1   # 初始轻微惩罚
    collision_stem_end = -2.0     # 最终严厉惩罚
    
    # 朝向要求渐进 (从宽松到严格)
    orient_power_start = 1.0      # 初始线性
    orient_power_end = 1.5        # 最终1.5次方
    
    # 两阶段训练配置 (Stage 2)
    phase1_steps = 15e6     # Phase 1: Actor + Critic
    phase2_steps = 10e6     # Phase 2: Critic only
    
    # 自适应课程
    success_rate_threshold = 0.3
    phase_switch_success_threshold = 0.30
```

---

## 🌍 域随机化配置

### 深度噪声 (`DcmmCfg.depth_noise`)

模拟RealSense D435i在户外农业场景的噪声特性：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cutout_num` | (1, 3) | 矩形遮挡数量 |
| `cutout_size` | (0.05, 0.15) | 遮挡大小比例 |
| `salt_ratio` | (0.01, 0.03) | 盐噪声比例 |
| `pepper_ratio` | (0.02, 0.05) | 胡椒噪声比例 |
| `dropout_rate` | (0.05, 0.10) | 像素丢失率 |
| `base_std` | 0.01 | 基础高斯噪声 |

### 光照随机化 (`DcmmCfg.lighting_dr`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ambient_range` | (0.1, 0.5) | 环境光强度 |
| `diffuse_range` | (0.3, 0.8) | 漫反射强度 |
| `dir_noise` | 0.3 | 光源方向噪声 |

---

## ❓ 常见问题

### Q: 训练时多个窗口弹出导致系统卡死?
**A**: 使用可视化时确保 `num_envs=1 viewer=True`，代码会自动强制设置。

### Q: AVP 如何开关?
**A**: 命令行传参 `avp_enabled=False` 或修改 `configs/env/DcmmCfg.py` 中 `avp.enabled = False`。

### Q: 训练速度太慢?
**A**: 
1. 增大 `num_envs`（推荐8-16，不超过CPU核心数×2，代码限制最大为18以防止内存溢出）
2. 确保使用GPU (`device_id=0`)
3. 关闭WandB (`wandb_mode=disabled`)

### Q: 显存不足 (OOM)?
**A**: 减少 `num_envs` 或 `minibatch_size`。

### Q: Stage 2 训练不稳定?
**A**: 检查Stage 2 Critic是否已训练收敛后再用于AVP。

### Q: 如何查看训练曲线?
**A**: 
1. WandB: 访问 wandb.ai 查看在线日志
2. TensorBoard: `tensorboard --logdir outputs/Dcmm/YYYY-MM-DD/tb`

### Q: 如何添加新的物体?
**A**: 
1. 在 `assets/objects/` 添加物体mesh
2. 修改 `assets/urdf/` 中的MJCF文件
3. 在 `DcmmCfg.py` 的 `object_shape` 和 `object_size` 中添加配置

---

## 📚 参考文献

- [PPO: Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [MuJoCo: Advanced Physics Simulation](https://mujoco.org/)
- [LEAP Hand: Low-cost Efficient Adaptable Programmable Hand](https://leaphand.com/)

---

## 📄 许可证

MIT License

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request
