# Plan: Stage 2 训练优化修改方案

通过延长 Phase 1 训练时间、添加成功率阈值检查、以及放宽过严的终止条件来改善训练效果。

## 问题分析

### 核心问题
1. **Episode 长度异常短** (8.3 步 vs 正常 100+ 步) - 机器人过早被终止
2. **Phase 2 后奖励完全停滞** - 从 583K 步到 9.19M 步，最佳奖励一直是 3.19
3. **成功率极低** (~4.9%) - 机器人无法完成抓取任务

### 根本原因
1. 第 577-579 行的终止条件过于严格：一旦进入 grasping 阶段，距离变大就立即终止
2. Phase 1 只有 5M 步，策略还未成熟就被冻结
3. Phase 2 切换没有成功率门槛检查

## Steps

1. **修改 [DcmmCfg.py](/home/cle/catch_it/configs/env/DcmmCfg.py) 第 163-165 行**：将 `phase1_steps` 从 `5e6` 改为 `15e6`，添加 `phase_switch_success_threshold = 0.30`

2. **修改 [PPO_Stage2.py](/home/cle/catch_it/gym_dcmm/algs/ppo_dcmm/stage2/PPO_Stage2.py) 第 270 行**：在 `_switch_to_phase2` 调用前添加成功率检查，只有当 `success_rate >= 0.30` 时才切换

3. **修改 [DcmmVecEnvStage2.py](/home/cle/catch_it/gym_dcmm/envs/stage2/DcmmVecEnvStage2.py) 第 577-579 行**：放宽 grasping 阶段的终止条件，添加宽容计数器而非立即终止

4. **增强 WandB 日志**：确保 reward 分布数据被记录到 `rewards/*` 指标中

## 具体修改内容

### 文件 1: `/home/cle/catch_it/configs/env/DcmmCfg.py`

**第 159-180 行**，替换为：

```python
class curriculum:
    # Define stage switching thresholds (steps)
    stage1_steps = 2e6  # First 2M steps
    stage2_steps = 10e6  # Extended curriculum period
    
    # Two-phase training configuration
    # [Modified 2025-12-09] Phase 1 extended from 5M to 15M steps
    phase1_steps = 15e6  # Phase 1: Learn grasping (Actor + Critic)
    phase2_steps = 10e6  # Phase 2: Learn value discrimination (Critic only)
    
    # [New] Success rate threshold for phase switching
    phase_switch_success_threshold = 0.30  # 30% success rate required
    
    # Stem collision penalty changes (reduced severity)
    collision_stem_start = -0.5
    collision_stem_end = -5.0
    
    # Orientation strictness changes (reduced severity)
    orient_power_start = 1.0
    orient_power_end = 2.0
    
    # Adaptive curriculum parameters
    success_rate_threshold = 0.3
    difficulty_decay_rate = 0.1
```

---

### 文件 2: `/home/cle/catch_it/gym_dcmm/algs/ppo_dcmm/stage2/PPO_Stage2.py`

**第 145-151 行**，替换为：

```python
        # ========================================
        # Two-Phase Training Configuration
        # ========================================
        import configs.env.DcmmCfg as DcmmCfg
        self.phase1_steps = int(getattr(DcmmCfg.curriculum, 'phase1_steps', 15e6))
        self.phase2_steps = int(getattr(DcmmCfg.curriculum, 'phase2_steps', 10e6))
        self.phase_switch_success_threshold = getattr(DcmmCfg.curriculum, 'phase_switch_success_threshold', 0.30)
        self.current_phase = 1  # 1 = Actor + Critic, 2 = Critic only
        self.phase_switched = False
        print(f"[PPO_Stage2] Two-Phase Training: Phase 1 = {self.phase1_steps/1e6:.1f}M steps, Phase 2 = {self.phase2_steps/1e6:.1f}M steps")
        print(f"[PPO_Stage2] Phase switch requires success_rate >= {self.phase_switch_success_threshold:.0%}")
```

**第 268-272 行**，替换为：

```python
            # ========================================
            # Two-Phase Training: Check for Phase Switch
            # ========================================
            if self.agent_steps >= self.phase1_steps and not self.phase_switched:
                # [New] Check success rate before switching
                try:
                    success_rate = self.env.env_method("get_recent_success_rate")[0]
                except:
                    success_rate = 0.0
                
                if success_rate >= self.phase_switch_success_threshold:
                    print(f"[PPO_Stage2] Success rate {success_rate:.1%} >= {self.phase_switch_success_threshold:.0%}, switching to Phase 2")
                    self._switch_to_phase2()
                else:
                    # Log warning every 1M steps
                    if self.agent_steps % 1000000 < self.batch_size:
                        print(f"[PPO_Stage2] Phase 1 extended: success_rate {success_rate:.1%} < {self.phase_switch_success_threshold:.0%}")
```

---

### 文件 3: `/home/cle/catch_it/gym_dcmm/envs/stage2/DcmmVecEnvStage2.py`

**在 `__init__` 方法中（约第 245 行后）添加新的变量**：

```python
        self.grasping_distance_violations = 0  # [New] Counter for distance violations
        self.max_distance_violations = 10  # [New] Allow some tolerance before terminating
```

**第 574-580 行**，替换为：

```python
            if self.task == 'Catching':
                if info['ee_distance'] < DcmmCfg.distance_thresh and self.stage == "tracking":
                    self.stage = "grasping"
                    self.grasping_distance_violations = 0  # Reset counter when entering grasping
                elif info['ee_distance'] >= DcmmCfg.distance_thresh and self.stage == "grasping":
                    # [Modified 2025-12-09] Add tolerance instead of immediate termination
                    self.grasping_distance_violations += 1
                    if self.grasping_distance_violations >= self.max_distance_violations:
                        self.terminated = True
                        print(f"[Terminated] Distance violations exceeded {self.max_distance_violations}")
                else:
                    # Reset counter if back within threshold
                    if self.stage == "grasping":
                        self.grasping_distance_violations = max(0, self.grasping_distance_violations - 1)
```

**在 `reset` 方法中（约第 447 行后）添加重置**：

```python
        self.grasping_distance_violations = 0  # Reset violation counter
```

---

## Further Considerations

1. **`env_time` 参数**: 已确认 `train_stage2.py` 中设置为 `env_time = 5.0` 秒，对应约 100 步 (5s * 20 steps/s)，这是合理的。问题不在此处。

2. **Episode 长度仍然过短？** 如果修改后 Episode 长度仍然很短，可能需要检查是否有其他隐藏的终止条件

3. **成功率阈值调整**: 如果 30% 太高导致 Phase 2 永远无法触发，可以降低到 15-20%

4. **WandB 查看**: 修改后运行训练，在 WandB 网页端查看 `rewards/reaching_mean`, `rewards/grasp_mean`, `rewards/collision_mean` 等指标来诊断瓶颈

## 预期效果

| 指标 | 修改前 | 修改后预期 |
|------|--------|-----------|
| Phase 1 时长 | 5M steps | 15M steps |
| Phase 切换条件 | 仅步数 | 步数 + 成功率 ≥ 30% |
| Episode 长度 | ~8 步 | ~50-100 步 |
| 距离违规容忍 | 0 次 | 10 次 |

