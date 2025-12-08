# Absolute Paths Report - Summary

## Quick Overview

This document provides a quick summary of all path references found in the Picking_Sorting project codebase.

## Key Findings

### ✅ Good News
- **No hardcoded absolute paths found** (e.g., `/home/user/...` or `C:\...`)
- All paths use relative references
- Project is portable across different environments

### ⚠️ Areas for Improvement
- **10 files** use `sys.path.append(os.path.abspath(...))` which is not best practice
- These should be replaced with proper package structure using `setup.py` or `pyproject.toml`

## Path Categories

| Category | Count | Examples |
|----------|-------|----------|
| sys.path modifications | 10 files | `sys.path.append(os.path.abspath('../../'))` |
| Asset file paths | 5 paths | `assets/urdf/x1_xarm6_leap_right_object.xml` |
| Model checkpoints | 3 paths | `assets/checkpoints/avp/stage2_critic.pth` |
| Output directories | 3 paths | `outputs/<name>/<timestamp>/` |

## Files with Path References

### 1. System Path Modifications (sys.path.append)

1. `gym_dcmm/utils/ik_pkg/ik_base.py` (line 2)
2. `gym_dcmm/utils/ik_pkg/ik_arm.py` (lines 9-10)
3. `gym_dcmm/agents/MujocoDcmm.py` (line 2)
4. `gym_dcmm/algs/ppo_dcmm/stage1/PPO_Stage1.py` (line 2)
5. `gym_dcmm/algs/ppo_dcmm/stage2/PPO_Stage2.py` (line 2)
6. `gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_two_stage.py` (line 2)
7. `gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_one_stage.py` (line 2)
8. `gym_dcmm/envs/stage1/DcmmVecEnvStage1.py` (lines 8-9)
9. `gym_dcmm/envs/stage2/DcmmVecEnvStage2.py` (lines 8-9)
10. `gym_dcmm/envs/stage2/test_stage2_optimizations.py` (line 17)

### 2. Asset Paths Configuration

**Primary file:** `configs/env/DcmmCfg.py`

Defines:
- `ASSET_PATH`: Base path for assets (computed as `../../assets` relative to config file)
- `XML_DCMM_LEAP_OBJECT_PATH`: `"urdf/x1_xarm6_leap_right_object.xml"`
- `XML_DCMM_LEAP_UNSEEN_OBJECT_PATH`: `"urdf/x1_xarm6_leap_right_unseen_object.xml"`
- `XML_ARM_PATH`: `"urdf/xarm6_right.xml"`
- `WEIGHT_PATH`: `<ASSET_PATH>/weights`

### 3. Model Checkpoint Paths

**Configuration:** `configs/env/DcmmCfg.py` (line 186)
```python
class avp:
    checkpoint_path = "assets/checkpoints/avp/stage2_critic.pth"
```

**YAML configs:** `configs/config_stage1.yaml`, `configs/config_stage2.yaml`
```yaml
checkpoint_tracking: 'assets/models/track.pth'  # (commented out by default)
checkpoint_catching: 'assets/models/catch_two_stage.pth'  # (commented out by default)
```

**Loading:** `gym_dcmm/envs/stage1/RewardManagerStage1.py`
- Computes project root using: `os.path.dirname(__file__)` chain
- Joins with relative checkpoint path

### 4. Output Directories

**Training scripts:** `train_stage1.py`, `train_stage2.py`
- Output format: `outputs/<output_name>/<YYYY-MM-DD>/<HH:MM:SS>/`
- Uses Shanghai timezone (`pytz.timezone('Asia/Shanghai')`)

**PPO algorithms:** `gym_dcmm/algs/ppo_dcmm/stage1/PPO_Stage1.py`
- `nn_dir`: `<output_dir>/nn/` (neural network weights)
- `tb_dif`: `<output_dir>/tb/` (TensorBoard logs)

## Recommendations

### High Priority
1. **Remove sys.path.append usage**
   - Implement proper package structure
   - Use `setup.py` or `pyproject.toml`
   - Install package in development mode: `pip install -e .`

### Medium Priority
2. **Centralize path configuration**
   - Consider using environment variables for root directory
   - Add path validation to ensure files exist
   - Document expected directory structure

### Low Priority
3. **Path handling improvements**
   - Use `pathlib.Path` instead of `os.path` for cleaner code
   - Add unit tests for path resolution logic

## Detailed Report

For the complete detailed report in Chinese, see: [ABSOLUTE_PATHS_REPORT.md](./ABSOLUTE_PATHS_REPORT.md)

---

**Generated:** 2025-12-08  
**Version:** 1.0  
**Report Type:** Code Analysis - Path References
