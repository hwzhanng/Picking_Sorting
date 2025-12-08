# DCMM 项目训练目标与 WandB 监控指南

本文档详细说明了 Stage 1 (Tracking) 和 Stage 2 (Catching) 模型的训练目标、理论满分奖励计算，以及 WandB 中关键参数的解读方法。

---

## 1. 理论满分奖励计算

### 1.1 Stage 1 (Tracking) 理论满分

**每步最大奖励组成：**

| 奖励组件 | 最大值 | 条件 |
|---------|--------|------|
| `reward_reaching` | **1.0** | 当 `ee_distance = 0` 时，`1.0 * (1.0 - tanh(0)) = 1.0` |
| `reward_base_approach` | **1.0** | 当 `base_distance = 0.8m` (最优距离) 时，`exp(-5.0 * 0²) = 1.0` |
| `reward_orientation` | **2.0** | 当 `ee_distance < 2.0m` 且 `alignment = 1.0` (完美对齐)，`1.0^4 * 2.0 = 2.0` |
| `reward_touch` | **10.0** | 成功触碰目标，`impact_speed = 0` 时 |
| `reward_avp` | **5.0** | AVP 最大值 (被 clip 到 ±5.0) |
| `reward_regularization` | **~0** | 控制量极小时接近 0 |
| `reward_action_rate` | **~0** | 动作平滑时接近 0 |

**每步最大正奖励：≈ 19.0**

> [!NOTE]
> 实际训练中，`reward_touch = 10.0` 是稀疏奖励，只有在成功触碰时才会获得。

**每步最小惩罚：**

| 惩罚组件 | 最小值 (负数) | 条件 |
|---------|--------------|------|
| `reward_collision` | **-10.0** | 发生致命碰撞 |
| `reward_plant_collision (stem)` | **-1.0 → -20.0** | 碰撞植物茎干 (课程学习) |
| `reward_plant_collision (leaf)` | **~-0.6** | 碰撞叶子，`-0.5 * (1.0 + ee_vel)` |
| `reward_impact` | **~-4.0** | 高速撞击 |

**Episode 满分计算 (125 步)：**

```
理想情况（无碰撞，持续接近目标）:
- 持续 reaching + base_approach: 125 × 2.0 = 250.0
- 接近后 orientation: ~50 × 2.0 = 100.0
- 成功 touch (1次): 10.0
- AVP 奖励 (接近时): ~25 × 3.0 = 75.0

理论上限: ≈ 400-500 per episode
实际优秀目标: ≈ 100-200 per episode
```

> [!IMPORTANT]
> **Stage 1 训练目标:**
> - 初期目标 (0-5M steps): `mean_reward > 20`
> - 中期目标 (5-15M steps): `mean_reward > 50`
> - 优秀目标 (15-25M steps): `mean_reward > 100`
> - 成功率目标: `> 70%`

---

### 1.2 Stage 2 (Catching) 理论满分

**每步最大奖励组成：**

| 奖励组件 | 最大值 | 条件 |
|---------|--------|------|
| `reward_reaching` | **1.0** | 当 `ee_distance = 0` 时 |
| `reward_orientation` | **2.0** | 完美对齐 (`ee_distance < 0.5m`) |
| `reward_grasp` | **9.0** | 完美抓握力 + 4个手指接触，`5.0 + 4 * 1.0` |
| `reward_perturbation` | **10.0** | 成功抵抗扰动测试 |
| `reward_success` | **50.0** | 成功抓取（稀疏奖励） |
| `reward_regularization` | **~0** | 控制量极小时 |
| `reward_action_rate` | **~0** | 动作平滑时 |

**每步最大正奖励：≈ 22.0 (不含成功奖励)**

**每步惩罚：**

| 惩罚组件 | 最小值 (负数) | 条件 |
|---------|--------------|------|
| `reward_collision` | **-10.0** | 致命碰撞 |
| `reward_impact` | **-386** (极端) | 高速撞击 (1.0 m/s) |
| `reward_plant_collision (stem)` | **-20.0** | 碰撞植物茎干 |
| `reward_plant_collision (leaf)` | **~-0.2** | 碰撞叶子 |

**Episode 满分计算 (125 步)：**

```
理想情况:
- 持续 reaching: 125 × 1.0 = 125.0
- orientation (接近时): ~60 × 2.0 = 120.0
- 抓握奖励: ~60 × 8.0 = 480.0
- 扰动测试通过: ~2 × 10.0 = 20.0
- 成功奖励: 50.0

理论上限: ≈ 700-800 per episode
实际优秀目标: ≈ 200-400 per episode
```

> [!IMPORTANT]
> **Stage 2 训练目标:**
> - 初期目标 (0-5M steps): `mean_reward > 10`
> - 中期目标 (5-15M steps): `mean_reward > 50`
> - 优秀目标 (15-25M steps): `mean_reward > 150`
> - 成功率目标: `> 80%`

---

## 2. WandB 关键参数解读

### 2.1 核心性能指标

| 参数名 | 含义 | 健康范围 | 异常信号 |
|--------|------|----------|---------|
| `metrics/episode_rewards_per_step` | 平均 episode 奖励 | **持续上升** | 停滞/下降 |
| `metrics/episode_lengths_per_step` | 平均 episode 长度 | 依任务而定 | 过短 (<20) = 早期失败 |
| `metrics/episode_success_per_step` | 成功率 | **>0.5 目标** | <0.1 持续 = 策略未学习 |

> [!TIP]
> **解读技巧:**
> - `episode_rewards` 上升 + `episode_lengths` 稳定 = 策略质量提升
> - `episode_lengths` 急剧下降 = 可能过度惩罚导致"找死"
> - `episode_success` 接近 0 持续 5M+ steps = 需检查奖励设计

---

### 2.2 损失函数监控

| 参数名 | 含义 | 健康范围 | 异常信号 |
|--------|------|----------|---------|
| `losses/actor_loss` | 策略梯度损失 | **0.01 - 0.1** | >1.0 = 不稳定 |
| `losses/critic_loss` | 价值函数损失 | **0.1 - 1.0** | >10 = 价值估计偏差大 |
| `losses/entropy` | 策略熵 | **0.5 - 3.0** | <0.1 = 过早收敛 |
| `losses/bounds_loss` | 动作边界损失 | **<0.01** | >0.1 = 动作饱和 |

> [!WARNING]
> **熵过低 (entropy < 0.2) 的危害:**
> - 策略过早收敛到次优解
> - 探索不足，无法发现更好策略
> - **解决方案:** 增加 `entropy_coef` 或降低学习率

**熵变化曲线解读:**

```
正常曲线: 高(3.0) → 缓慢下降 → 稳定(0.5-1.0)
异常曲线: 高(3.0) → 急剧下降 → 接近0 = 需调整
```

---

### 2.3 训练稳定性指标

| 参数名 | 含义 | 健康范围 | 异常信号 |
|--------|------|----------|---------|
| `info/kl` | KL 散度 | **0.005 - 0.02** | >0.05 = 策略更新过激 |
| `info/last_lr` | 当前学习率 | 依调度器而定 | 过早降为 min = 调度器问题 |
| `info/e_clip` | PPO clip 参数 | **0.2 (固定)** | N/A |

> [!CAUTION]
> **KL 散度异常高 (>0.1) 的处理:**
> 1. 检查学习率是否过高
> 2. 增加 `mini_epochs` 数量
> 3. 减小 `minibatch_size`

---

### 2.4 性能指标

| 参数名 | 含义 | 参考值 | 优化建议 |
|--------|------|--------|---------|
| `performance/RLTrainFPS` | RL 训练帧率 | **>1000** | 低 = 网络过大/minibatch 过小 |
| `performance/EnvStepFPS` | 环境采样帧率 | **>500** | 低 = 环境计算瓶颈 |

---

### 2.5 AVP 专用指标 (仅 Stage 1)

| 参数名 | 含义 | 健康范围 | 异常信号 |
|--------|------|----------|---------|
| `avp/reward_mean` | 平均 AVP 奖励 | **1.0 - 5.0** | <0 = Critic 值负面 |
| `avp/critic_value_mean` | Stage 2 Critic 平均值 | **正值** | 持续负值 = 模型问题 |
| `avp/lambda` | 当前 AVP 权重 | **0.8 → 0.2** (课程) | 不变化 = 课程未生效 |
| `avp/gate_ratio` | AVP 被门控跳过比例 | **30-70%** | >95% = 门控距离过严 |

> [!TIP]
> **AVP 调试技巧:**
> - 如果 `gate_ratio` 接近 100%，说明智能体从不进入 AVP 激活范围
> - 可以增大 `gate_distance` 或优化 Stage 1 的 reaching 能力

---

## 3. 训练健康状态速查表

### 3.1 健康训练模式

```
✅ episode_rewards: 持续缓慢上升
✅ episode_success: 逐步提升到 >50%
✅ entropy: 缓慢下降但保持 >0.5
✅ kl: 稳定在 0.01-0.02
✅ actor_loss: 稳定在 0.01-0.1
✅ critic_loss: 稳定在 0.1-1.0
```

### 3.2 需要调整的信号

| 现象 | 可能原因 | 调整建议 |
|------|----------|---------|
| Reward 持平不变 | 稀疏奖励/探索不足 | 增加 `entropy_coef` |
| Reward 剧烈震荡 | 学习率过高/奖励不平衡 | 降低 `learning_rate` |
| Entropy 快速归零 | 策略过早收敛 | 增加 `entropy_coef` 到 0.05 |
| KL 持续过高 | 策略更新过激 | 降低 `learning_rate` |
| Episode length 很短 | 过度惩罚/碰撞频繁 | 检查碰撞惩罚权重 |
| Success rate = 0 | 任务过难/奖励设计问题 | 简化初始条件/检查奖励 |

---

## 4. 奖励分解监控参数 (已实现)

以下参数已自动记录到 WandB：

### 4.1 Stage 1 奖励分解

| 参数名 | 含义 |
|--------|------|
| `rewards/reaching_mean` | 平均 reaching 奖励 |
| `rewards/base_approach_mean` | 平均底盘接近奖励 |
| `rewards/orientation_mean` | 平均朝向奖励 |
| `rewards/touch_mean` | 平均触碰奖励 |
| `rewards/collision_mean` | 平均碰撞惩罚 |
| `rewards/plant_collision_mean` | 平均植物碰撞惩罚 |
| `distance/ee_distance_mean` | 平均末端到目标距离 |
| `distance/base_distance_mean` | 平均底盘到目标距离 |
| `curriculum/difficulty` | 课程学习难度系数 (0→1) |
| `curriculum/w_stem` | 当前茎干碰撞惩罚权重 |
| `curriculum/orient_power` | 当前朝向严格度 |

### 4.2 Stage 2 奖励分解

| 参数名 | 含义 |
|--------|------|
| `rewards/reaching_mean` | 平均 reaching 奖励 |
| `rewards/orientation_mean` | 平均朝向奖励 |
| `rewards/grasp_mean` | 平均抓握奖励 |
| `rewards/perturbation_mean` | 平均扰动测试奖励 |
| `rewards/impact_mean` | 平均冲击惩罚 |
| `rewards/collision_mean` | 平均碰撞惩罚 |
| `rewards/success_mean` | 平均成功奖励 |
| `grasp/contact_force_mean` | 平均接触力 (N) |
| `grasp/fingers_touching_mean` | 平均接触手指数 |



## 5. 训练监控检查清单

训练时定期检查以下内容：

- [ ] `episode_rewards` 是否持续上升？
- [ ] `episode_success` 是否逐步提升？
- [ ] `entropy` 是否维持在合理范围 (>0.5)？
- [ ] `kl` 是否稳定在 0.01-0.02？
- [ ] `actor_loss` 和 `critic_loss` 是否稳定？
- [ ] `FPS` 是否正常 (>500)？
- [ ] (Stage 1) `avp/lambda` 是否按课程衰减？
- [ ] (Stage 2) 成功率是否达到目标 (>50%)？

---

## 6. 快速参考

### Stage 1:
- **理论每步最大奖励:** ~19.0
- **理论每 Episode 最大奖励:** ~400-500
- **实际优秀目标:** 100-200 per episode
- **成功率目标:** >70%

### Stage 2:
- **理论每步最大奖励:** ~22.0 (不含成功奖励)
- **理论每 Episode 最大奖励:** ~700-800
- **实际优秀目标:** 200-400 per episode
- **成功率目标:** >80%

