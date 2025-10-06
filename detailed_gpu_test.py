import tensorflow as tf
import sys
import os

print("===== 详细GPU检测报告 =====")
print(f"Python版本: {sys.version}")
print(f"TensorFlow版本: {tf.__version__}")

# 检查TensorFlow构建信息
print("\nTensorFlow构建信息:")
print(f"- 是否支持CUDA: {tf.test.is_built_with_cuda()}")
print(f"- 是否支持GPU: {tf.test.is_built_with_gpu_support()}")

# 列出所有物理设备
print("\n可用物理设备:")
all_devices = tf.config.list_physical_devices()
for device in all_devices:
    print(f"- {device.device_type}: {device.name}")

# 尝试不同方式检测GPU
try:
    # 方式1: list_physical_devices('GPU')
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\n使用list_physical_devices('GPU')检测:")
    if gpus:
        print(f"  找到 {len(gpus)} 个GPU设备:")
        for gpu in gpus:
            print(f"  - {gpu.name}")
    else:
        print("  未找到GPU设备")

except Exception as e:
    print(f"  检测失败: {e}")

try:
    # 方式2: 直接尝试创建GPU上下文
    print("\n尝试创建GPU上下文:")
    with tf.device('/GPU:0'):
        a = tf.constant([1.0, 2.0, 3.0])
        b = tf.constant([4.0, 5.0, 6.0])
        c = a + b
        print(f"  成功在GPU上执行简单计算: {c}")
except Exception as e:
    print(f"  在GPU上执行计算失败: {e}")

try:
    # 方式3: 检查CUDA可见设备
    print("\nCUDA环境变量信息:")
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')
    print(f"  CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")

except Exception as e:
    print(f"  读取环境变量失败: {e}")

try:
    # 方式4: 打印GPU设备属性
    if gpus:
        print("\nGPU设备属性:")
        for gpu in gpus:
            try:
                # 获取GPU设备详情
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print(f"  设备: {gpu.name}")
                print(f"  详情: {gpu_details}")
            except Exception as e:
                print(f"  无法获取设备 {gpu.name} 的详情: {e}")

except Exception as e:
    print(f"  获取GPU属性失败: {e}")

# 提供建议
print("\n===== 建议与解决方案 =====")
if not gpus:
    print("1. 请确保已安装兼容的NVIDIA显卡驱动")
    print("2. 请确保已安装匹配TensorFlow版本的CUDA和cuDNN")
    print("3. 检查是否有其他程序正在占用GPU资源")
    print("4. 尝试以管理员权限运行程序")
    print("5. 如果仍然无法检测，程序将自动回退到CPU模式")
else:
    print("GPU已成功检测，程序将使用GPU进行计算")

print("\n===== 检测完成 =====")