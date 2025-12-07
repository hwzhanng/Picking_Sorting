# Picking_Sorting: 基于AVP的两阶段农业采摘强化学习项目

## 项目概述

**Picking_Sorting** 是一个强化学习项目，训练移动操作机器人（移动底盘+6自由度机械臂+灵巧手）在农业场景中自主导航至目标果实并进行采摘。

### 核心技术

- **两阶段训练 (Two-Stage Training)**: Stage 1 训练底盘+机械臂接近目标，Stage 2 训练灵巧手抓取
- **AVP (Asymmetric Value Propagation)**: 使用 Stage 2 Critic 为 Stage 1 提供"可抓取性"奖励信号
- **动态课程学习**: 0→6M步渐进式调整碰撞惩罚和朝向精度要求
- **关节空间控制**: 直接输出关节角度增量，避免IK不稳定性

---

## 快速开始

### 安装

```bash
git clone <repository_url>
cd catch_it
conda create -n dcmm python=3.8
conda activate dcmm
pip install -r requirements.txt
pip install -e .
```

---

## 🎯 训练指南

### Stage 1: 追踪任务 (Tracking)

训练底盘和机械臂接近目标果实，同时避开植株障碍。

```bash
# 基础训练 (AVP 默认开启)
python train_stage1.py

# 关闭 AVP (消融实验基线)
python train_stage1.py avp_enabled=False

# 调整并行环境数 (根据GPU显存调整，推荐16-32)
python train_stage1.py num_envs=16

# 从检查点恢复训练
python train_stage1.py checkpoint_tracking="outputs/Dcmm/xxx/nn/best_reward_XXX.pth"
```

**关键参数** (`configs/config_stage1.yaml`):
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_envs` | 8 | 并行环境数（增大可提高吞吐量，但增加显存占用） |
| `train.ppo.horizon_length` | 512 | 每次收集的步数（减小可加速更新频率） |
| `train.ppo.learning_rate` | 3e-4 | 学习率 |
| `train.ppo.max_agent_steps` | 25M | 最大训练步数 |

---

### Stage 2: 抓取任务 (Catching)

使用预训练的 Stage 1 模型（冻结），训练灵巧手进行抓取。

```bash
# 基础训练（必须指定 Stage 1 检查点）
python train_stage2.py checkpoint_tracking="outputs/Dcmm/.../best_reward_XXX.pth"

# 调整并行环境数
python train_stage2.py num_envs=16 checkpoint_tracking="..."
```

**关键参数** (`configs/config_stage2.yaml`):
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_envs` | 8 | 并行环境数 |
| `checkpoint_tracking` | 无 | **必填** Stage 1 模型路径 |

---

## 👁️ 可视化指南

> ⚠️ **重要**: 可视化时请使用 `num_envs=1` 避免多窗口导致系统卡死

### 可视化训练过程 (单窗口)

```bash
# Stage 1 可视化训练 (AVP 开启)
python train_stage1.py num_envs=1 viewer=True

# Stage 1 可视化训练 (AVP 关闭)
python train_stage1.py num_envs=1 viewer=True avp_enabled=False

# Stage 2 可视化训练
python train_stage2.py num_envs=1 viewer=True checkpoint_tracking="..."
```

### 加载检查点验证 (单窗口)

```bash
# 验证 Stage 1 (AVP 开启)
python train_stage1.py test=True num_envs=1 viewer=True \
    checkpoint_tracking="outputs/Dcmm/.../best_reward_XXX.pth"

# 验证 Stage 1 (AVP 关闭)
python train_stage1.py test=True num_envs=1 viewer=True avp_enabled=False \
    checkpoint_tracking="outputs/Dcmm/.../best_reward_XXX.pth"

# 验证 Stage 2 完整采摘流程
python train_stage2.py test=True num_envs=1 viewer=True \
    checkpoint_tracking="outputs/Dcmm/.../track.pth" \
    checkpoint_catching="outputs/Dcmm_Catch/.../best.pth"
```

---

## 🔧 AVP 配置

**AVP (Asymmetric Value Propagation)** 使用预训练的 Stage 2 Critic 为 Stage 1 提供"可抓取性"奖励信号。

### 配置位置

修改 `configs/env/DcmmCfg.py` 中的 `avp` 类:

```python
class avp:
    enabled = True           # 主开关 (False=关闭AVP)
    lambda_weight = 0.5      # AVP奖励权重 (0.3-0.8推荐)
    gate_distance = 1.5      # 距离门限 (仅在此距离内计算AVP)
    checkpoint_path = "assets/checkpoints/avp/stage2_critic.pth"
```

### 消融实验

```bash
# 基线 (无AVP)
python train_stage1.py avp_enabled=False

# 完整方法 (有AVP)  
python train_stage1.py avp_enabled=True
```

### 更新 AVP 权重

当训练出更好的 Stage 2 模型时，复制到指定位置:
```bash
cp outputs/Dcmm_Catch/.../best_reward_XXX.pth assets/checkpoints/avp/stage2_critic.pth
```

---

## 📁 项目结构

```
catch_it/
├── train_stage1.py              # Stage 1 训练入口
├── train_stage2.py              # Stage 2 训练入口
├── configs/
│   ├── config_stage1.yaml       # Stage 1 主配置
│   ├── config_stage2.yaml       # Stage 2 主配置
│   └── env/DcmmCfg.py           # 环境+AVP配置
├── gym_dcmm/
│   ├── envs/
│   │   ├── stage1/              # Stage 1 环境
│   │   └── stage2/              # Stage 2 环境
│   └── algs/ppo_dcmm/
│       ├── stage1/              # Stage 1 算法
│       └── stage2/              # Stage 2 算法
├── assets/
│   ├── checkpoints/avp/         # AVP 预训练权重
│   │   └── stage2_critic.pth
│   └── urdf/                    # MuJoCo 模型
└── outputs/                     # 训练输出
```

---

## 🎓 AVP 原理

**问题**: Stage 1 仅使用手工设计的到达奖励，无法评估"当前位置是否便于后续抓取"。

**解决**: 加载预训练的 Stage 2 Critic，构造"虚拟观测"（假设手臂已在就绪姿态），获取价值估计作为辅助奖励。

```
Stage 1 当前状态
      |
      v
构造虚拟观测: [就绪姿态手臂, 真实物体位置, 真实深度图]
      |
      v
Stage 2 Critic(虚拟观测) → value_estimate
      |
      v
AVP 奖励 = lambda_weight × value_estimate
      |
      v
总奖励 = 原始奖励 + AVP奖励
```

**效果**: 引导 Stage 1 将机器人导航至"便于抓取"的位置，而非仅仅"距离目标近"。

---

## 常见问题

### Q: 训练时多个窗口弹出导致卡死?
**A**: 确保使用 `num_envs=1 viewer=True`，只开单窗口。

### Q: AVP 如何开关?
**A**: 修改 `configs/env/DcmmCfg.py` 中 `avp.enabled = False`，或命令行传参 `avp_enabled=False`。

### Q: 训练速度太慢?
**A**: 增大 `num_envs`（如32或64），确保使用GPU (`rl_device='cuda:0'`)。

### Q: Stage 2 训练失败?
**A**: 确认 `checkpoint_tracking` 指向有效的 Stage 1 模型。

---

## 许可证

MIT License