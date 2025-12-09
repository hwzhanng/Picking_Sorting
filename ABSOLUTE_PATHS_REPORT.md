# 代码中的绝对路径报告 (Absolute Paths Report)

## 摘要 (Summary)
本报告详细列出了在 Picking_Sorting 项目中发现的所有绝对路径和相对路径引用。这些路径主要用于系统路径设置、资源文件加载和模型检查点管理。

---

## 1. sys.path.append 使用绝对路径 (sys.path with Absolute Paths)

以下文件使用 `sys.path.append(os.path.abspath(...))` 来修改 Python 模块搜索路径：

### 1.1 gym_dcmm/utils/ik_pkg/ik_base.py
- **行号**: 2
- **代码**: `sys.path.append(os.path.abspath('../../'))`
- **说明**: 添加相对于当前文件的父级目录到系统路径

### 1.2 gym_dcmm/utils/ik_pkg/ik_arm.py
- **行号**: 9-10
- **代码**: 
  ```python
  sys.path.append(os.path.abspath('../'))
  sys.path.append(os.path.abspath('./gym_dcmm/'))
  ```
- **说明**: 添加两个路径到系统路径用于导入模块

### 1.3 gym_dcmm/agents/MujocoDcmm.py
- **行号**: 2
- **代码**: `sys.path.append(os.path.abspath('../'))`
- **说明**: 添加父级目录到系统路径

### 1.4 gym_dcmm/algs/ppo_dcmm/stage1/PPO_Stage1.py
- **行号**: 2
- **代码**: `sys.path.append(os.path.abspath('../../../gym_dcmm'))`
- **说明**: 添加 gym_dcmm 模块路径

### 1.5 gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_two_stage.py
- **行号**: 2
- **代码**: `sys.path.append(os.path.abspath('../gym_dcmm'))`
- **说明**: 添加 gym_dcmm 模块路径

### 1.6 gym_dcmm/algs/ppo_dcmm/stage2/PPO_Stage2.py
- **行号**: 2
- **代码**: `sys.path.append(os.path.abspath('../../../gym_dcmm'))`
- **说明**: 添加 gym_dcmm 模块路径

### 1.7 gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_one_stage.py
- **行号**: 2
- **代码**: `sys.path.append(os.path.abspath('../gym_dcmm'))`
- **说明**: 添加 gym_dcmm 模块路径

### 1.8 gym_dcmm/envs/stage1/DcmmVecEnvStage1.py
- **行号**: 8-9
- **代码**: 
  ```python
  sys.path.append(os.path.abspath('../../'))
  sys.path.append(os.path.abspath('../gym_dcmm/'))
  ```
- **说明**: 添加多个路径用于导入

### 1.9 gym_dcmm/envs/stage2/DcmmVecEnvStage2.py
- **行号**: 8-9
- **代码**: 
  ```python
  sys.path.append(os.path.abspath('../../'))
  sys.path.append(os.path.abspath('../gym_dcmm/'))
  ```
- **说明**: 添加多个路径用于导入

### 1.10 gym_dcmm/envs/stage2/test_stage2_optimizations.py
- **行号**: 17
- **代码**: `sys.path.insert(0, os.path.abspath('../../'))`
- **说明**: 在系统路径开头插入路径

---

## 2. 资源文件路径 (Asset File Paths)

### 2.1 configs/env/DcmmCfg.py

**文件位置**: Line 6-15

```python
path = os.path.realpath(__file__)
root = str(Path(path).parent)
ASSET_PATH = os.path.join(root, "../../assets")

# XML 文件路径
XML_DCMM_LEAP_OBJECT_PATH = "urdf/x1_xarm6_leap_right_object.xml"
XML_DCMM_LEAP_UNSEEN_OBJECT_PATH = "urdf/x1_xarm6_leap_right_unseen_object.xml"
XML_ARM_PATH = "urdf/xarm6_right.xml"

# 权重保存路径
WEIGHT_PATH = os.path.join(ASSET_PATH, "weights")
```

**说明**: 
- `ASSET_PATH`: 相对路径构建，指向 `configs/env/../../assets`
- XML 文件路径为相对于 `ASSET_PATH` 的相对路径
- 实际完整路径为: `assets/urdf/x1_xarm6_leap_right_object.xml` 等

### 2.2 gym_dcmm/agents/MujocoDcmm.py

**文件位置**: Line 59-66

```python
if not object_eval: 
    model_path = os.path.join(DcmmCfg.ASSET_PATH, DcmmCfg.XML_DCMM_LEAP_OBJECT_PATH)
else: 
    model_path = os.path.join(DcmmCfg.ASSET_PATH, DcmmCfg.XML_DCMM_LEAP_UNSEEN_OBJECT_PATH)

model_arm_path = os.path.join(DcmmCfg.ASSET_PATH, DcmmCfg.XML_ARM_PATH)
```

**说明**: 使用 `DcmmCfg` 中定义的路径变量来构建模型文件路径

---

## 3. 模型检查点路径 (Model Checkpoint Paths)

### 3.1 configs/env/DcmmCfg.py - AVP 配置

**文件位置**: Line 186

```python
# AVP (Asymmetric Value Propagation) Configuration
class avp:
    # Checkpoint path for Stage 2 Critic (relative to project root)
    checkpoint_path = "assets/checkpoints/avp/stage2_critic.pth"
```

**说明**: Stage 2 Critic 模型的检查点路径，相对于项目根目录

### 3.2 gym_dcmm/envs/stage1/RewardManagerStage1.py

**文件位置**: Line 61-72

```python
def _load_stage2_critic(self):
    """Load Stage 2 Critic model for AVP reward computation."""
    # Find checkpoint path (relative to project root)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    ckpt_path = os.path.join(project_root, self.avp_checkpoint_path)
    
    if not os.path.exists(ckpt_path):
        print(f">>> AVP Warning: Stage 2 Checkpoint not found at {ckpt_path}")
```

**说明**: 
- 通过多级 `os.path.dirname()` 计算项目根目录
- 拼接相对路径获取检查点完整路径
- 实际路径: `<project_root>/assets/checkpoints/avp/stage2_critic.pth`

### 3.3 configs/config_stage1.yaml

**文件位置**: Line 21-24

```yaml
# used to set checkpoint path
checkpoint_tracking: ''  # Empty for fresh training (15-dim obs space)
checkpoint_catching: ''
# checkpoint_tracking: 'assets/models/track.pth'
# checkpoint_catching: 'assets/models/catch_two_stage.pth'
```

**说明**: 
- 注释中包含示例检查点路径
- 相对于项目根目录的路径

### 3.4 configs/config_stage2.yaml

**文件位置**: Line 21-24

```yaml
# used to set checkpoint path
checkpoint_tracking: ''  # Empty for fresh training (15-dim obs space)
checkpoint_catching: ''
# checkpoint_tracking: 'assets/models/track.pth'
# checkpoint_catching: 'assets/models/catch_two_stage.pth'
```

**说明**: 与 stage1 配置相同的检查点路径设置

### 3.5 train_stage1.py 和 train_stage2.py

**文件位置**: train_stage1.py Line 25-32

```python
if config.task == 'Tracking' and config.checkpoint_tracking:
    config.checkpoint_tracking = to_absolute_path(config.checkpoint_tracking)
    model_path = config.checkpoint_tracking
elif (config.task == 'Catching_TwoStage' \
    or config.task == 'Catching_OneStage') \
    and config.checkpoint_catching:
    config.checkpoint_catching = to_absolute_path(config.checkpoint_catching)
    model_path = config.checkpoint_catching
```

**说明**: 使用 Hydra 的 `to_absolute_path()` 函数将配置中的相对路径转换为绝对路径

---

## 4. 输出目录路径 (Output Directory Paths)

### 4.1 train_stage1.py 和 train_stage2.py

**文件位置**: train_stage1.py Line 71-77

```python
output_dif = os.path.join('outputs', config.output_name)
# Get the local date and time
local_tz = pytz.timezone('Asia/Shanghai')
current_datetime = datetime.datetime.now().astimezone(local_tz)
current_datetime_str = current_datetime.strftime("%Y-%m-%d/%H:%M:%S")
output_dif = os.path.join(output_dif, current_datetime_str)
os.makedirs(output_dif, exist_ok=True)
```

**说明**: 
- 创建输出目录，格式为 `outputs/<output_name>/<date>/<time>`
- 使用上海时区的时间戳
- 相对于项目根目录

### 4.2 gym_dcmm/algs/ppo_dcmm/stage1/PPO_Stage1.py

**文件位置**: Line 66-71

```python
self.output_dir = output_dif
self.nn_dir = os.path.join(self.output_dir, 'nn')
self.tb_dif = os.path.join(self.output_dir, 'tb')
os.makedirs(self.nn_dir, exist_ok=True)
os.makedirs(self.tb_dif, exist_ok=True)
```

**说明**: 
- `nn_dir`: 神经网络权重保存目录
- `tb_dif`: TensorBoard 日志目录

---

## 5. 路径类型总结 (Path Type Summary)

| 路径类型 | 数量 | 说明 |
|---------|------|------|
| sys.path 修改 | 10个文件 | 使用 `os.path.abspath()` 添加相对路径到系统路径 |
| 资源文件路径 | 5个 | XML/URDF 模型文件，相对于 `assets/` 目录 |
| 检查点路径 | 3个 | 模型权重文件 (.pth, .pt)，相对于项目根目录 |
| 输出目录 | 3个 | 训练输出、模型保存、日志目录 |

---

## 6. 建议 (Recommendations)

### 6.1 路径硬编码问题

**问题**: 
- 多个文件使用硬编码的相对路径进行 `sys.path.append`
- 这些路径依赖于文件的具体位置，移动文件会导致导入失败

**建议**: 
1. 使用 `setup.py` 或 `pyproject.toml` 正确设置包结构
2. 避免使用 `sys.path.append`，改用正确的包导入
3. 或者创建一个中央配置文件统一管理所有路径

### 6.2 配置路径管理

**现状良好**: 
- `DcmmCfg.py` 集中管理了资源路径
- 使用相对路径而非绝对路径，便于移植

**可改进**: 
- 可以考虑使用环境变量或配置文件来管理根目录
- 添加路径验证逻辑，确保文件存在

### 6.3 检查点路径

**现状良好**: 
- 使用 Hydra 的 `to_absolute_path()` 处理配置中的路径
- 相对路径便于版本控制和项目移植

---

## 7. 完整文件列表 (Complete File List)

包含路径引用的所有文件：

```
1. gym_dcmm/utils/ik_pkg/ik_base.py
2. gym_dcmm/utils/ik_pkg/ik_arm.py
3. gym_dcmm/agents/MujocoDcmm.py
4. gym_dcmm/algs/ppo_dcmm/stage1/PPO_Stage1.py
5. gym_dcmm/algs/ppo_dcmm/stage2/PPO_Stage2.py
6. gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_two_stage.py
7. gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_one_stage.py
8. gym_dcmm/envs/stage1/DcmmVecEnvStage1.py
9. gym_dcmm/envs/stage2/DcmmVecEnvStage2.py
10. gym_dcmm/envs/stage1/RewardManagerStage1.py
11. gym_dcmm/envs/stage2/test_stage2_optimizations.py
12. configs/env/DcmmCfg.py
13. configs/config_stage1.yaml
14. configs/config_stage2.yaml
15. train_stage1.py
16. train_stage2.py
```

---

## 8. 注意事项 (Notes)

1. **无真正的绝对路径**: 代码中没有使用类似 `/home/user/...` 或 `C:\Users\...` 这样的硬编码绝对路径
2. **相对路径为主**: 大部分路径都是相对路径，通过 `os.path.join()` 和 `os.path.abspath()` 构建
3. **可移植性**: 当前的路径设计相对灵活，便于项目在不同环境中运行
4. **sys.path 问题**: 主要问题在于过多使用 `sys.path.append`，应该改用标准的 Python 包管理方式

---

生成时间: 2025-12-08
报告版本: 1.0
