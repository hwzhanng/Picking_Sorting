# 部署说明 (Deployment Guide)

本文档提供了如何在新电脑或主机上部署本项目的详细说明。

## 部署方式 1: 使用 Conda 环境 (推荐)

这是最推荐的方式，可以完整复制当前的环境，包括 CUDA 依赖。

### 步骤:

1. **安装 Anaconda 或 Miniconda**
   ```bash
   # 下载 Miniconda (如果尚未安装)
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

2. **创建 Conda 环境**
   ```bash
   cd /path/to/catch_it
   conda env create -f environment.yml
   ```

3. **激活环境**
   ```bash
   conda activate dcmm
   ```

4. **安装本地包**
   ```bash
   pip install -e .
   ```

5. **验证安装**
   ```bash
   python -c "import gym_dcmm; import mujoco; import torch; print('安装成功!')"
   ```

## 部署方式 2: 使用 pip (仅核心依赖)

如果不需要完整的 Conda 环境，可以仅使用 pip 安装核心依赖。

### 步骤:

1. **创建虚拟环境**
   ```bash
   python3.8 -m venv venv
   source venv/bin/activate
   ```

2. **安装 PyTorch (带 CUDA 支持)**
   ```bash
   # 访问 https://pytorch.org/ 获取适合您系统的安装命令
   pip install torch==2.4.1 torchvision==0.20.0 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
   ```

3. **安装项目依赖**
   ```bash
   cd /path/to/catch_it
   pip install -r requirements.txt
   ```

4. **安装本地包**
   ```bash
   pip install -e .
   ```

5. **验证安装**
   ```bash
   python -c "import gym_dcmm; import mujoco; import torch; print('安装成功!')"
   ```

## 部署方式 3: 在已有的 Conda 环境中安装

如果您已经有一个类似的环境，可以直接安装依赖。

### 步骤:

1. **激活您的环境**
   ```bash
   conda activate your_env_name
   ```

2. **安装依赖**
   ```bash
   cd /path/to/catch_it
   pip install -r requirements.txt
   ```

3. **安装本地包**
   ```bash
   pip install -e .
   ```

## 系统要求

- **操作系统**: Linux (Ubuntu 18.04+)
- **Python**: 3.8.x
- **GPU**: NVIDIA GPU (支持 CUDA 12.1)
- **CUDA**: 12.1 或更高版本
- **显存**: 建议 8GB+

## 常见问题

### 1. CUDA 版本不匹配

如果您的系统 CUDA 版本不是 12.1，请修改 `environment.yml` 中的 CUDA 版本或访问 [PyTorch 官网](https://pytorch.org/) 获取适合您的 PyTorch 安装命令。

### 2. Mujoco 许可证问题

MuJoCo 3.x 及以上版本已经开源，无需许可证。确保您安装的是 `mujoco>=3.2.3`。

### 3. 显示问题 (无显示器服务器)

如果在没有显示器的服务器上运行，可能需要配置虚拟显示：

```bash
# 安装 xvfb
sudo apt-get install xvfb

# 使用虚拟显示运行
xvfb-run -a -s "-screen 0 1400x900x24" python train_stage1.py
```

### 4. ROS 依赖 (可选)

如果您需要 ROS 功能，请确保系统已安装 ROS Humble 或 Foxy。本项目的核心功能不依赖 ROS。

## 测试部署

安装完成后，可以运行以下测试脚本验证环境：

```bash
# 测试环境
python test_env.py

# 测试 Stage 1 训练 (单步测试)
python train_stage1.py --help
```

## 获取帮助

如果遇到问题，请检查：
1. Python 版本是否为 3.8.x
2. 所有依赖是否正确安装
3. CUDA 驱动是否正确安装
4. GPU 是否可用: `python -c "import torch; print(torch.cuda.is_available())"`

---

**最后更新**: 2025-12-06
