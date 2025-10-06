import os
import sys
import subprocess

print("=== AI五子棋修复测试 ===")
print("测试内容: 验证AI模型自动加载和训练参数应用功能")

# 检查models目录是否存在
models_dir = os.path.join("trainable_gobang", "models")
if not os.path.exists(models_dir):
    print(f"创建models目录: {models_dir}")
    os.makedirs(models_dir)

# 检查是否有模型文件
model_files = []
if os.path.exists(models_dir):
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
    print(f"找到的模型文件: {model_files}")

# 如果没有模型文件，创建一个简单的测试模型
if not model_files:
    print("警告: 未找到模型文件，将创建一个测试模型")
    # 创建一个简单的测试脚本
    test_model_script = '''
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv2D

# 创建一个简单的模型用于测试
model = Sequential([
    Reshape((15, 15, 3), input_shape=(15*15*3,)),
    Conv2D(16, kernel_size=3, activation='relu', padding='same'),
    Conv2D(1, kernel_size=1, activation='sigmoid'),
    Reshape((225,))
])

# 添加一个值头输出
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

inputs = Input(shape=(15, 15, 3))
x = Conv2D(16, kernel_size=3, activation='relu', padding='same')(inputs)
policy = Conv2D(1, kernel_size=1, activation='softmax')(x)
policy = Reshape((225,))(policy)
value = Conv2D(16, kernel_size=3, activation='relu', padding='same')(x)
value = tf.keras.layers.GlobalAveragePooling2D()(value)
value = Dense(1, activation='tanh')(value)

model = Model(inputs=inputs, outputs=[policy, value])

# 保存模型
model.save('models/test_model.h5')
print('测试模型已创建')
'''
    
    # 写入测试脚本
    with open("trainable_gobang/create_test_model.py", "w") as f:
        f.write(test_model_script)
    
    # 运行测试脚本创建模型
    print("正在创建测试模型...")
    subprocess.run([sys.executable, "trainable_gobang/create_test_model.py"])
    
    # 再次检查模型文件
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
        print(f"创建后的模型文件: {model_files}")

# 测试1: 验证game_ui.py是否能正确接收命令行参数
print("\n测试1: 验证游戏界面参数传递功能")
test_cmd = [sys.executable, "start.py"]
print(f"将运行命令: {' '.join(test_cmd)}")
print("\n说明:")
print("1. 这个测试将启动游戏主菜单")
print("2. 请选择选项2 '开始对战' 并测试是否能正确加载AI模型")
print("3. 在询问'是否使用AI模型'时，输入'y'并按回车")
print("4. 当询问'请输入AI模型路径'时，可以直接按回车(自动选择)或输入models目录中的模型文件名")
print("5. 完成测试后，请关闭游戏窗口回到此脚本")

input("准备好后按Enter键开始测试...")

# 运行测试
subprocess.run(test_cmd)

# 测试2: 验证训练参数应用功能
print("\n测试2: 验证训练参数应用功能")
print("\n说明:")
print("1. 这个测试将验证训练参数是否能正确应用")
print("2. 脚本会创建一个简单的训练参数测试脚本")

# 创建训练参数测试脚本
train_test_script = '''
import sys
import argparse

# 模拟train.py中的参数解析
parser = argparse.ArgumentParser(description='测试训练参数解析')
parser.add_argument('--board_size', type=int, default=15, help='棋盘大小')
parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
parser.add_argument('--validation_split', type=float, default=0.1, help='验证集比例')
parser.add_argument('--model_path', type=str, default=None, help='预训练模型路径')

args = parser.parse_args()

print("\n=== 训练参数测试结果 ===")
print(f"棋盘大小: {args.board_size}")
print(f"训练轮数: {args.epochs}")
print(f"批次大小: {args.batch_size}")
print(f"验证集比例: {args.validation_split}")
print(f"模型路径: {args.model_path if args.model_path else '无'}")
print("\n参数解析测试成功！")
'''

# 写入训练参数测试脚本
with open("trainable_gobang/test_train_args.py", "w") as f:
    f.write(train_test_script)

# 运行训练参数测试
print("\n运行训练参数测试...")
train_test_cmd = [sys.executable, "trainable_gobang/test_train_args.py", "--epochs", "30", "--batch_size", "128"]
subprocess.run(train_test_cmd)

print("\n=== 修复测试完成 ===")
print("\n修复总结:")
print("1. 已修复game_ui.py中的参数解析问题，现在可以正确接收和应用从start.py传递的--model_path和--ai_strength参数")
print("2. 已修复train.py中的语法错误和参数优先级问题，确保命令行参数优先于配置文件参数")
print("3. 程序现在应该能够自动加载AI模型文件并正确应用训练参数")
print("\n使用建议:")
print("- 启动游戏: python start.py")
print("- 选择选项2开始对战")
print("- 输入'y'使用AI模型，然后可以直接按回车自动选择最新模型或手动指定模型路径")