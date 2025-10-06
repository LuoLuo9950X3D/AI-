import tensorflow as tf
import sys

print("===== AI五子棋环境检测报告 =====")
print(f"Python版本: {sys.version}")
print(f"TensorFlow版本: {tf.__version__}")

# 检查TensorFlow构建信息
print(f"- 是否支持CUDA: {tf.test.is_built_with_cuda()}")
print(f"- 是否支持GPU: {tf.test.is_built_with_gpu_support()}")

# 列出所有物理设备
all_devices = tf.config.list_physical_devices()
print(f"\n可用物理设备总数: {len(all_devices)}")
for device in all_devices:
    print(f"- {device.device_type}: {device.name}")

# 检测GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"\n可用GPU设备: {len(gpus)}")
if len(gpus) > 0:
    for gpu in gpus:
        print(f"- {gpu.name}")
    print("\n注意: 根据检测结果，当前TensorFlow版本不支持CUDA加速。")
    print("程序将自动使用CPU模式运行，游戏功能不受影响。")
    print("如需GPU加速，请安装支持CUDA的TensorFlow版本。")
else:
    print("未检测到GPU设备")
    print("程序将使用CPU模式运行，游戏功能不受影响。")

print("\n===== 检测完成 =====")