# Catch It: 基于强化学习的两阶段农业采摘项目

## 项目概述
"Catch It" 是一个强化学习 (RL) 项目,旨在训练一个移动操作机器人(配备移动底盘、6自由度机械臂和灵巧手)在农业场景中自主导航至目标果实并进行采摘。本项目采用 **两阶段训练 (Two-Stage Training)** 方法和**动态课程学习 (Curriculum Learning)**,高效解决移动采摘这一复杂的长视距任务。

### 核心特性
- **🌾 农业场景适配**: 静态果实目标、植株障碍物(树干/树叶)、视野约束
- **📚 动态课程学习**: 从"探索"到"约束"的渐进式训练(0-6M步三阶段)
- **🎯 精细化奖励函数**: 8种奖励分量,区分刚性/柔性碰撞,速度感知惩罚
- **🔧 关节空间控制**: 直接输出关节角度增量,避免IK不稳定性
- **🎨 两阶段 RL 框架**: 将任务分解为 **追踪 (Tracking)** 和 **抓取 (Catching)** 两个阶段
- **🤖 Sim2Real 就绪**: 内置域随机化(观测噪声、动作延迟、深度缺失),便于从 MuJoCo 仿真迁移到真机硬件
- **🧩 模块化设计**: 环境 (`gym_dcmm`)、算法 (`PPO`) 和配置 (`hydra`) 分离,结构清晰

---

## 🆕 最新更新亮点

### 1. 动态课程学习机制
训练过程自动分为三个阶段,逐步提高任务难度:

| 阶段 | 训练步数 | 树干碰撞惩罚 | 朝向精度要求 | 训练目标 |
|------|---------|------------|------------|---------|
| 🏫 幼儿园 | 0 - 2M | -1.0 (轻微) | 1次方 (宽松) | 建立信心,学会探索 |
| 🎓 小学 | 2M - 6M | -1.0 → -20.0 | 1 → 4次方 | 学会规避,精确定位 |
| 💼 职业 | > 6M | -20.0 (严厉) | 4次方 (严格) | 精准作业,零容忍 |

**优势**: 避免"过早失败",智能体能在宽松环境中快速建立基本能力,再逐步适应严格约束。

### 2. 精细化奖励函数
重构后的奖励系统包含8个分量:

1. **到达奖励**: `1.0 * (1.0 - tanh(2.0 * distance))` - 归一化防爆炸
2. **底盘定位奖励**: `exp(-5.0 * (distance - 0.8)^2)` - 引导至最佳操作距离
3. **朝向奖励**: `max(0, alignment)^orient_power * 2.0` - 动态严格度(1→4次方)
4. **接触奖励**: `10.0 - 4.0 * impact_speed` - 鼓励温柔接触
5. **正则化**: `-0.01 * ||ctrl||` - 平滑控制
6. **碰撞惩罚**: `-10.0` - 底盘碰撞地面(终止)
7. **植物碰撞**: 
   - 树干(刚性): 动态惩罚 `-1.0 → -20.0`
   - 树叶(柔性): 轻微惩罚 `-0.5 * (1.0 + vel)`
8. **动作平滑**: `-0.05 * ||action_diff||` - 惩罚突变

### 3. 农业场景特性
- **静态果实**: 使用 mocap 固定目标位置,去除速度观测
- **植株障碍**: 随机生成树干和树叶,模拟真实农田
- **视野约束**: 果实和植株仅在机器人前方视野内生成
- **顶部摄像头**: 使用车顶深度相机观测场景

---

## 架构与工作流

### 第一阶段:追踪 (Tracking)
- **目标**: 控制移动底盘和机械臂接近目标果实,手掌朝向目标,避开植株障碍
- **输入**: 机器人状态(底盘, 手臂) + 果实位置 + 深度图像
- **输出**: 底盘速度 + 手臂关节角度增量
- **课程学习**: 训练过程中自动从"宽松探索"过渡到"精准约束"
- **成果**: 训练出一个"采摘导航员"策略,能安全高效地将机器人引导至果实附近

### 第二阶段:抓取 (Catching)
- **目标**: 在底盘和手臂由第一阶段策略控制的情况下,训练灵巧手抓取并采摘果实
- **机制**: 
    - 加载预训练的追踪模型并 **冻结** 其参数
    - 训练一个新的策略专门控制灵巧手进行抓取动作
- **成果**: 一个能够接近、朝向并采摘果实的完整系统

---

## 安装指南

1.  **克隆仓库**:
    ```bash
    git clone <repository_url>
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

### 1. 训练第一阶段:追踪 (Tracking)
训练底盘和手臂接近果实,同时避开植株。
```bash
python train_DCMM.py task=Tracking
```
**训练过程**:
- 0-2M步:智能体大胆探索,可能频繁碰撞树干(惩罚-1.0)
- 2M-6M步:逐渐学会绕行,朝向要求提高
- >6M步:精准避障,手掌完美对准果实

*模型检查点将保存在 `outputs/Dcmm/YYYY-MM-DD/HH:MM:SS/nn/` 目录下。*

### 2. 训练第二阶段:抓取 (Catching)
使用预训练的追踪模型,训练灵巧手进行抓取。

**注意**: 请确保 `configs/config.yaml` 中的 `checkpoint_tracking` 指向你最好的第一阶段模型路径。
```bash
python train_DCMM.py task=Catching_TwoStage checkpoint_tracking="outputs/Dcmm/.../best_reward_XXX.pth"
```

### 3. 测试 / 可视化
可视化训练好的智能体(渲染模式):
```bash
# 测试追踪阶段
python train_DCMM.py task=Tracking test=True checkpoint_tracking="path/to/best.pth" num_envs=1 viewer=True

# 测试完整采摘流程
python train_DCMM.py task=Catching_TwoStage test=True checkpoint_tracking="path/to/track.pth" checkpoint_catching="path/to/catch.pth" num_envs=1 viewer=True
```

---

## 配置说明

### 通用配置 (`configs/config.yaml`)
- 设置任务类型 (`task`): `Tracking` / `Catching_TwoStage` / `Catching_OneStage`
- 环境数量 (`num_envs`): 并行环境数(建议16-32)
- 模型加载路径 (`checkpoint_tracking`, `checkpoint_catching`)
- WandB 实验跟踪配置

### 环境配置 (`configs/env/DcmmCfg.py`)

#### 🆕 课程学习配置 (`curriculum` 类)
```python
class curriculum:
    stage1_steps = 2e6  # 第一阶段步数阈值
    stage2_steps = 6e6  # 第二阶段步数阈值
    
    # 树干碰撞惩罚范围
    collision_stem_start = -1.0
    collision_stem_end = -20.0
    
    # 朝向精度要求范围(次方数)
    orient_power_start = 1.0
    orient_power_end = 4.0
```

**调整建议**:
- 如果智能体学习太慢,可以延长 `stage1_steps` (增加探索时间)
- 如果想要更严格的最终行为,可以增大 `collision_stem_end` 或 `orient_power_end`

#### 奖励权重 (`reward_weights`)
```python
reward_weights = {
    "r_ee_pos": 1.0,        # 到达奖励
    "r_orient": 1.0,        # 朝向奖励
    "r_touch": 10.0,        # 接触奖励
    "r_collision": -10.0,   # 底盘碰撞
    # ... 其他权重
}
```

#### Sim2Real 配置
```python
# 观测噪声(模拟传感器不确定性)
k_obs_object = 0.025  # 目标物体位置噪声

# 动作延迟(模拟真实执行延迟)
act_delay = {
    'base': [1, 2, 3],  # 底盘:20-60ms
    'arm': [1, 2, 3],   # 机械臂:20-60ms
    'hand': [1, 2, 3],  # 灵巧手:20-60ms
}
```

---

## 项目结构
```
catch_it/
├── train_DCMM.py              # 训练/测试的主入口脚本
├── test_env.py                # 环境测试脚本
├── configs/                   # 配置文件 (Hydra & Python)
│   ├── config.yaml            # 主配置
│   ├── env/DcmmCfg.py         # 环境配置(含课程学习参数)
│   └── train/DcmmPPO.yaml     # PPO 训练参数
├── gym_dcmm/                  # 环境包
│   ├── envs/
│   │   ├── DcmmVecEnv.py      # 主环境 + 课程调度器
│   │   ├── reward_manager.py  # 奖励函数(8种分量)
│   │   ├── observation_manager.py  # 观测处理 + 噪声
│   │   ├── control_manager.py      # 控制 + 碰撞检测
│   │   ├── randomization_manager.py # 植株/果实随机化
│   │   └── render_manager.py       # 深度图渲染
│   ├── agents/MujocoDcmm.py   # MuJoCo 接口 + PID 控制
│   └── algs/ppo_dcmm/         # PPO 算法实现
│       ├── ppo_dcmm_track.py       # 第一阶段训练
│       ├── ppo_dcmm_catch_two_stage.py  # 第二阶段训练
│       ├── models_track.py         # 追踪网络(CNN+MLP)
│       └── models_catch.py         # 抓取网络
├── assets/                    # 机器人 XML 文件和模型检查点
│   ├── urdf/                  # MuJoCo XML 模型
│   ├── meshes/                # 3D 网格文件
│   └── models/                # 预训练权重(.pth)
└── outputs/                   # 训练输出(检查点、日志)
```

---

## 🎓 课程学习的工作原理

### 核心思想
传统RL训练对智能体要求"从一开始就做对",导致大量早期失败和低效探索。课程学习模仿人类学习过程,**先建立信心,再追求完美**。

### 实现机制
1. **训练脚本** (`ppo_dcmm_track.py`):
   - 每次收集经验前,调用 `env.set_global_step(current_step)`
   
2. **环境调度器** (`DcmmVecEnv.update_curriculum_difficulty()`):
   ```python
   difficulty = min(step / 6M, 1.0)  # 0.0 → 1.0
   current_w_stem = -1.0 + (-20.0 - (-1.0)) * difficulty
   current_orient_power = 1.0 + (4.0 - 1.0) * difficulty
   ```

3. **奖励管理器** (`reward_manager.compute_reward()`):
   - 使用 `env.current_w_stem` 作为树干碰撞惩罚
   - 使用 `env.current_orient_power` 作为朝向奖励次方数

### 监控课程进度
训练时每10k步会打印课程状态:
```
[Curriculum] Step: 0, Stem Penalty: -1.00, Orient Power: 1.00
[Curriculum] Step: 1000000, Stem Penalty: -4.17, Orient Power: 1.50
[Curriculum] Step: 3000000, Stem Penalty: -10.50, Orient Power: 2.50
[Curriculum] Step: 6000000, Stem Penalty: -20.00, Orient Power: 4.00
```

---

## 🌾 农业场景设计

### 静态果实设定
- 使用 **mocap body** 固定果实位置(不受物理影响)
- 移除物体速度观测(从15维降至12维)
- 模拟真实农田中果实挂在树上的状态

### 植株障碍生成
- **树干 (plant_stem)**: 刚性障碍,严禁碰撞
- **树叶 (plant_leaf)**: 柔性遮挡,允许轻触
- 随机位置:仅在机器人前方视野内生成

### 视觉观测
- 使用车顶深度相机(俯视角度)
- 添加高斯噪声和像素缺失(5-10%)模拟真实传感器

---

## Sim2Real (仿真到真机) 部署指南

### 1. 增大域随机化强度
```python
# 在 DcmmCfg.py 中调整
k_obs_object = 0.05   # 增大观测噪声(默认0.025)
act_delay = {
    'base': [2, 3, 4],  # 增大延迟范围(默认[1,2,3])
    'arm': [2, 3, 4],
    'hand': [2, 3, 4],
}
```

### 2. 真机硬件同步
- **传感器**: 确保深度相机位置和FOV与仿真一致
- **控制频率**: MuJoCo 默认500Hz,真机需匹配 `steps_per_policy` (默认20步=40ms一次策略)
- **关节限位**: 验证真机关节限位与 XML 定义一致

### 3. 部署流程
```bash
# 1. 导出策略权重
weights = torch.load("best_reward_XXX.pth")
actor_weights = weights['model']['actor_mlp']

# 2. 在真机ROS节点中加载
model = ActorCritic(net_config)
model.load_state_dict(actor_weights)
model.eval()

# 3. 实时推理
obs = get_robot_state()  # 从传感器获取
action = model.act_inference(obs)
send_to_robot(action)    # 发送到底层控制器
```

---

## 常见问题 (FAQ)

### Q1: 训练时奖励一直是负数?
**A**: 这是正常的,尤其是在"幼儿园阶段"。观察 `[Curriculum]` 打印,确认课程参数在变化即可。关注奖励趋势(是否上升),而非绝对值。

### Q2: 如何加快训练速度?
**A**: 
1. 增加 `num_envs` (并行环境数)到你的CPU/GPU能承受的上限
2. 减小 `horizon_length` (如64→32)以增加更新频率
3. 确保使用GPU: `rl_device='cuda:0'`

### Q3: 智能体学会了碰撞树干后"卡住"?
**A**: 可能是惩罚太强。尝试:
1. 延长 `stage1_steps` (给更多探索时间)
2. 减小 `collision_stem_end` (如-20.0→-10.0)
3. 增大 `reward_reaching` 权重(鼓励继续前进)

### Q4: 第二阶段训练时底盘不动?
**A**: 确认 `checkpoint_tracking` 路径正确,且成功冻结追踪模型。检查日志中是否有"加载预训练模型"的提示。

---


**依赖项目**:
- MuJoCo: 高性能物理引擎
- Gymnasium: 强化学习环境接口
- Stable-Baselines3: PPO 算法参考实现
- WandB: 实验跟踪和可视化

---

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

---

## 联系方式

如有问题或建议,请通过以下方式联系:
- GitHub Issues: [项目Issues页面]
- Email: your.email@example.com

祝训练顺利! 🚀🌾