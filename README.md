# Catch It: 基于强化学习的两阶段移动抓取项目

## 项目概述
"Catch It" 是一个强化学习 (RL) 项目，旨在训练一个移动操作机器人（配备移动底盘、6自由度机械臂和灵巧手）自主追踪并抓取物体。本项目采用 **两阶段训练 (Two-Stage Training)** 方法，高效解决移动抓取这一复杂的长视距任务。

### 核心特性
- **两阶段 RL 框架**: 将任务分解为 **追踪 (Tracking)** 和 **抓取 (Catching)** 两个阶段。
- **Sim2Real 就绪**: 内置域随机化（观测噪声、动作延迟），便于从 MuJoCo 仿真迁移到真机硬件。
- **模块化设计**: 环境 (`gym_dcmm`)、算法 (`PPO`) 和配置 (`hydra`) 分离，结构清晰。

---

## 架构与工作流

### 第一阶段：追踪 (Tracking)
- **目标**: 控制移动底盘和机械臂接近目标物体，并保持末端执行器在物体附近。
- **输入**: 机器人状态 (底盘, 手臂) + 物体位置。
- **输出**: 底盘速度 + 手臂关节速度。
- **成果**: 训练出一个“领航员”策略，负责将机器人引导至目标附近。

### 第二阶段：抓取 (Catching)
- **目标**: 在底盘和手臂由第一阶段策略控制的情况下，训练灵巧手抓取并抬起物体。
- **机制**: 
    - 加载预训练的追踪模型并 **冻结** 其参数。
    - 训练一个新的策略专门控制灵巧手进行抓取动作。
- **成果**: 一个能够接近并抓取物体的完整系统。

---

## 安装指南

1.  **克隆仓库**:
    ```bash
    git clone https://github.com/hwzhanng/Picking_Sorting.git
    cd catch_it
    ```

2.  **创建 Conda 环境**:
    ```bash
    conda create -n dcmm python=3.8
    conda activate dcmm
    ```

3.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 使用说明

### 1. 训练第一阶段：追踪 (Tracking)
训练底盘和手臂跟随物体。
```bash
python train_DCMM.py task=Tracking
```
*模型检查点将保存在 `assets/models/` 目录下。*

### 2. 训练第二阶段：抓取 (Catching)
使用预训练的追踪模型，训练灵巧手进行抓取。
**注意**: 请确保 `configs/config.yaml` 中的 `checkpoint_tracking` 指向你最好的第一阶段模型路径。
```bash
python train_DCMM.py task=Catching_TwoStage
```

### 3. 测试 / 可视化
可视化训练好的智能体（渲染模式）：
```bash
python train_DCMM.py task=Catching_TwoStage test=True
```

---

## 配置说明

- **通用配置**: `configs/config.yaml`
    - 设置任务类型 (`task`)、环境数量 (`num_envs`) 和模型加载路径。
- **环境配置**: `configs/env/DcmmCfg.py`
    - **奖励 (Rewards)**: 调整接近、接触和抬起的奖励权重。
    - **Sim2Real**: 修改 `k_obs_object` (噪声) 和 `act_delay` (延迟) 以匹配真实环境。

## 项目结构
```
catch_it/
├── train_DCMM.py           # 训练/测试的主入口脚本
├── configs/                # 配置文件 (Hydra & Python)
├── gym_dcmm/               # 环境包
│   ├── envs/DcmmVecEnv.py  # Gym 环境核心逻辑
│   ├── agents/MujocoDcmm.py# MuJoCo 接口
│   └── algs/ppo_dcmm/      # PPO 算法实现
├── assets/                 # 机器人 XML 文件和模型检查点
└── 项目技术文档/            # 详细的技术文档
```

## Sim2Real (仿真到真机) 注意事项
为了部署到真实机器人，建议使用以下配置：
- **观测噪声**: 设置 `k_obs_object = 0.025` 以模拟感知噪声。
- **动作延迟**: 启用随机延迟 `[1, 2, 3]` 步 (约 40-120ms)。
- **静态速度**: 如果使用视觉追踪静态物体，建议强制将物体速度观测置零。
