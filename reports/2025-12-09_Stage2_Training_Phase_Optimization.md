# Stage 2 训练阶段优化报告

**日期**: 2025-12-09  
**训练运行**: `Dcmm_Catch/2025-12-08/18:31:22` (WandB run: `obgdo5ts`)  
**问题类型**: 训练停滞 & Episode 过早终止 & Phase 切换策略缺陷

---

## 📊 训练日志分析结果

### 原始指标

| 指标 | 起始值 | 最终值 (9.19M steps) | 评价 |
|------|--------|---------------------|------|
| Agent Steps | 0 | 9,191K | 37% 完成 |
| Episode Reward | -5.48 | **3.19** (停滞) | ⚠️ Phase 2 后无进步 |
| Episode Length | - | **8.33 步** | ❌ 异常短 (应 ~100 步) |
| Success Rate | - | **4.9%** | ❌ 极低 |
| Training Phase | Phase 1 → Phase 2 | Phase 2 | 5M 步后切换 |

### 关键发现

1. **奖励在 583K 步达到最佳值 3.19，之后 8.6M 步完全没有提升**
2. **Episode 长度只有 8 步**，正常应该是 100 步（5秒 × 20步/秒）
3. **Phase 2 切换后训练完全停滞**，Actor 被冻结但策略尚未成熟
4. **每步平均奖励为负** (`episode_rewards_per_step: -5.916`)

### 训练时间线

| 步数 | Phase | 最佳奖励 | 状态 |
|------|-------|---------|------|
| 0 - 583K | Phase 1 | -5.48 → 3.19 | ✅ 正常学习 |
| 583K - 5M | Phase 1 | 3.19 | ⚠️ 饱和 |
| 5M - 9.19M | Phase 2 | 3.19 | ❌ 完全停滞 |

---

## 🔴 发现的问题

### 问题1: Grasping 阶段距离违规立即终止 (最严重)

**位置**: `DcmmVecEnvStage2.py` Line 577-579

**原代码**:
```python
if self.task == 'Catching':
    if info['ee_distance'] < DcmmCfg.distance_thresh and self.stage == "tracking":
        self.stage = "grasping"
    elif info['ee_distance'] >= DcmmCfg.distance_thresh and self.stage == "grasping":
        self.terminated = True  # 立即终止！
```

**问题分析**:
- `distance_thresh = 0.25m`
- 当机器人进入 grasping 阶段后，如果距离稍微变大（≥ 0.25m），episode **立即终止**
- 这导致 Episode 长度只有 8 步，机器人根本没时间学习抓取动作
- 机器人被"惩罚"了探索行为

**修复方案**:
```python
# 添加容忍计数器
self.grasping_distance_violations = 0
self.max_distance_violations = 10  # 允许 10 步的容忍

# 修改终止逻辑
elif info['ee_distance'] >= DcmmCfg.distance_thresh and self.stage == "grasping":
    self.grasping_distance_violations += 1
    if self.grasping_distance_violations >= self.max_distance_violations:
        self.terminated = True
else:
    if self.stage == "grasping":
        self.grasping_distance_violations = max(0, self.grasping_distance_violations - 1)
```

---

### 问题2: Phase 1 训练时间不足

**位置**: `configs/env/DcmmCfg.py` Line 163-164

**原配置**:
```python
phase1_steps = 5e6   # Phase 1: 只有 5M 步
phase2_steps = 3e6   # Phase 2: 3M 步
```

**问题分析**:
- Phase 1 只有 5M 步，但策略在 583K 步后就停滞了
- 5M 步时最佳奖励只有 3.19，远低于目标 (>10)
- Phase 2 切换时策略尚未成熟，冻结 Actor 后无法继续学习

**修复方案**:
```python
phase1_steps = 15e6  # 延长到 15M 步
phase2_steps = 10e6  # Phase 2 也适当延长
```

---

### 问题3: Phase 切换无成功率门槛

**位置**: `gym_dcmm/algs/ppo_dcmm/stage2/PPO_Stage2.py` Line 270

**原代码**:
```python
if self.agent_steps >= self.phase1_steps and not self.phase_switched:
    self._switch_to_phase2()  # 只检查步数，不检查性能
```

**问题分析**:
- 只要步数达到 5M，就自动切换到 Phase 2
- 不检查当前策略的成功率是否达标
- 成功率只有 4.9% 时就冻结了 Actor

**修复方案**:
```python
if self.agent_steps >= self.phase1_steps and not self.phase_switched:
    try:
        success_rate = self.env.env_method("get_recent_success_rate")[0]
    except:
        success_rate = 0.0
    
    if success_rate >= self.phase_switch_success_threshold:  # 新增: 30% 门槛
        self._switch_to_phase2()
    else:
        # 继续 Phase 1 训练
        if self.agent_steps % 1000000 < self.batch_size:
            print(f"[PPO_Stage2] Phase 1 extended: success_rate {success_rate:.1%} < 30%")
```

---

## ✅ 修改总结

| 文件 | 修改项 | 原值 | 新值 |
|------|--------|------|------|
| `DcmmCfg.py` | `phase1_steps` | 5M | **15M** |
| `DcmmCfg.py` | `phase2_steps` | 3M | **10M** |
| `DcmmCfg.py` | `phase_switch_success_threshold` | (无) | **0.30** |
| `PPO_Stage2.py` | Phase 切换条件 | 仅步数 | 步数 + 成功率 ≥ 30% |
| `DcmmVecEnvStage2.py` | 距离违规容忍 | 0 次 | **10 次** |
| `DcmmVecEnvStage2.py` | `grasping_distance_violations` | (无) | 新增计数器 |

---

## 📐 关键参数配置

### Two-Phase 训练参数

| 参数 | 原值 | 新值 | 说明 |
|------|------|------|------|
| `phase1_steps` | 5e6 | **15e6** | Phase 1 训练步数 |
| `phase2_steps` | 3e6 | **10e6** | Phase 2 训练步数 |
| `phase_switch_success_threshold` | - | **0.30** | 切换到 Phase 2 的成功率门槛 |

### 终止条件参数

| 参数 | 原值 | 新值 | 说明 |
|------|------|------|------|
| `distance_thresh` | 0.25m | 0.25m | 不变 |
| `max_distance_violations` | 0 | **10** | 允许的距离违规次数 |
| `env_time` | 5.0s | 5.0s | 不变 (已确认足够) |

### 预期 Episode 长度

| 条件 | 原预期 | 实际 | 修复后预期 |
|------|--------|------|-----------|
| 正常训练 | ~100 步 | 8 步 | ~50-100 步 |
| 距离违规 | 立即终止 | - | 10 步后终止 |

---

## 📈 WandB 监控指标

修复后重点关注以下指标：

### 训练进度
- `train/phase`: 当前训练阶段 (1 或 2)
- `train/recent_success_rate`: 最近成功率 (Phase 切换依据)

### Episode 指标
- `metrics/episode_lengths_per_step`: **应从 ~8 提升到 ~50-100**
- `metrics/episode_rewards_per_step`: **应从负值变为正值**
- `metrics/episode_success_per_step`: **应从 ~5% 提升到 >30%**

### 奖励分解
- `rewards/reaching_mean`: 接近目标奖励
- `rewards/grasp_mean`: 抓取奖励
- `rewards/collision_mean`: 碰撞惩罚
- `rewards/impact_mean`: 撞击惩罚

---

## 🚀 建议后续行动

1. **重新训练** (推荐):
   ```bash
   python train_stage2.py num_envs=16 seed=44 output_name=Dcmm_Catch_optimized
   ```

2. **监控重点**:
   - Episode 长度是否恢复到 50-100 步
   - 成功率是否能突破 30%
   - Phase 1 是否会因成功率不足而延长

3. **如果成功率仍然很低**:
   - 降低 `phase_switch_success_threshold` 到 15-20%
   - 检查 `distance_thresh` 是否需要放宽

4. **如果 Episode 仍然很短**:
   - 检查是否有其他隐藏的终止条件
   - 增加 `max_distance_violations` 到 20

---

## 📁 相关文件

- 修改文件:
  - `configs/env/DcmmCfg.py`
  - `gym_dcmm/algs/ppo_dcmm/stage2/PPO_Stage2.py`
  - `gym_dcmm/envs/stage2/DcmmVecEnvStage2.py`
- 计划文件:
  - `plan-stage2TrainingOptimization.prompt.md`
- 规格文档:
  - `TRAINING_TARGETS.md`
  - `Stage2_Specification.md`

