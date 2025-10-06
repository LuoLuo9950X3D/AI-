# NVIDIA GPU配置指南

本指南提供在有NVIDIA GPU（如4060Ti）的环境中配置五子棋AI训练系统的详细步骤。

## 环境要求

### 硬件要求
- NVIDIA GPU（推荐RTX 30系列或40系列，如4060Ti）
- 至少8GB GPU内存
- 足够的磁盘空间用于存储模型和训练数据

### 软件要求
- Windows 10/11 64位或Linux操作系统
- Python 3.8-3.10
- NVIDIA驱动程序（版本520或更高）
- CUDA Toolkit 11.8或更高版本
- cuDNN 8.6或更高版本

## TensorFlow GPU配置

### 1. 安装CUDA和cuDNN

#### Windows系统：
1. 下载并安装[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. 下载并安装[cuDNN](https://developer.nvidia.com/cudnn)
3. 配置环境变量，确保CUDA相关路径已添加到系统PATH中

#### Linux系统：
```bash
# 安装CUDA
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# 安装cuDNN（需要NVIDIA开发者账号）
tar -xzvf cudnn-linux-x86_64-8.x.x.x_cuda12-archive.tar.xz
sudo cp -P cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### 2. 安装TensorFlow GPU版本

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux
# 或 
# venv\Scripts\activate  # Windows

# 安装TensorFlow GPU版本
pip install tensorflow[and-cuda]
# 或指定版本
# pip install tensorflow==2.15.0
```

### 3. 验证GPU配置

创建一个简单的Python脚本来验证TensorFlow是否能正确识别GPU：

```python
import tensorflow as tf
print('TensorFlow版本:', tf.__version__)
print('可用GPU设备:', tf.config.list_physical_devices('GPU'))
if len(tf.config.list_physical_devices('GPU')) > 0:
    print('GPU可用！')
    # 测试GPU计算
    with tf.device('/GPU:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
        print(c)
else:
    print('未检测到可用GPU，将使用CPU进行计算。')
```

## 常见问题解决方案

### 1. TensorFlow无法识别GPU

- 检查NVIDIA驱动程序是否已正确安装：运行`nvidia-smi`命令
- 确认CUDA和cuDNN版本与TensorFlow版本兼容
- 检查环境变量配置是否正确
- 尝试重新安装TensorFlow：`pip uninstall tensorflow && pip install tensorflow[and-cuda]`

### 2. 内存错误

- 如果遇到OOM（内存不足）错误，可以尝试：
  - 减小批次大小
  - 减少模型复杂度（在`training_optimization_config.json`中调整参数）
  - 使用`tf.config.experimental.set_memory_growth(gpu, True)`设置内存增长（已在代码中添加）

### 3. 训练速度慢

- 确保代码正在使用GPU而非CPU
- 尝试增加批次大小以充分利用GPU并行计算能力
- 在`training_optimization_config.json`中调整模型参数

## 优化建议

针对NVIDIA 4060Ti显卡，以下是一些优化建议：

- **批次大小**：推荐设置为64-128（根据可用显存调整）
- **学习率**：初始学习率设置为0.001-0.005
- **模型复杂度**：对于4060Ti，推荐使用8-12个残差块，128-256个过滤器
- **数据增强**：充分利用数据增强功能提高模型泛化能力

## 运行训练

配置完成后，可以使用以下命令开始训练：

```bash
# 生成初始训练数据
python trainable_gobang/self_play.py --num_games 10

# 训练模型
python trainable_gobang/train.py --data_dir data

# 使用训练好的模型进行自我对弈
sudo python trainable_gobang/self_play.py --num_games 100 --model_path models/best_model.h5
```

## 注意事项

- 确保您的系统满足最低要求
- 定期更新NVIDIA驱动程序和CUDA工具包以获得最佳性能
- 在训练大型模型时，确保有足够的散热和稳定的电源供应
- 如果在多GPU环境中，代码会自动选择第一个可用GPU