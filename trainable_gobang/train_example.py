import os
import subprocess
import sys

# 检查Python版本
if sys.version_info < (3, 7):
    print("错误: 本项目需要Python 3.7或更高版本")
    sys.exit(1)

# 安装依赖
print("===== 安装必要的依赖库 =====")
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    print("依赖安装成功！")
except subprocess.CalledProcessError:
    print("警告: 依赖安装可能失败，请手动安装requirements.txt中的依赖")

# 确认目录结构
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# 打印训练流程说明
print("\n===== 五子棋AI训练流程 =====")
print("1. 首先生成初始训练数据")
print("2. 然后进行模型训练")
print("3. 接着使用训练好的模型进行自我对弈，生成更高质量的数据")
print("4. 重复步骤2和3，迭代优化模型")
print("5. 评估模型性能")
print("6. 与训练好的AI对战")
print("\n请按照以下步骤开始训练：\n")

# 生成初始训练数据（使用随机自我对弈）
print("\n===== 步骤1: 生成初始训练数据 =====")
print("这将使用随机策略进行10局自我对弈，生成初始训练数据")
input("按Enter键继续...")

initial_data_command = [
    sys.executable, "self_play.py",
    "--num_games=10",
    "--output_dir=data"
]

try:
    subprocess.run(initial_data_command, check=True)
    print("初始训练数据生成成功！")
except subprocess.CalledProcessError:
    print("错误: 初始训练数据生成失败")
    sys.exit(1)

# 第一次训练模型
print("\n===== 步骤2: 第一次模型训练 =====")
print("这将使用生成的初始数据训练一个基础模型")
input("按Enter键继续...")

first_train_command = [
    sys.executable, "train.py",
    "--data_dir=data",
    "--epochs=10",
    "--batch_size=32",
    "--output_dir=models",
    "--model_name=initial_model"
]

try:
    subprocess.run(first_train_command, check=True)
    print("模型训练成功！")
except subprocess.CalledProcessError:
    print("错误: 模型训练失败")
    sys.exit(1)

# 使用训练后的模型进行自我对弈
print("\n===== 步骤3: 使用训练后的模型进行自我对弈 =====")
print("这将使用刚刚训练的模型进行50局自我对弈，生成更高质量的训练数据")
input("按Enter键继续...")

model_self_play_command = [
    sys.executable, "self_play.py",
    "--model_path=models/initial_model.h5",
    "--num_games=50",
    "--output_dir=data",
    "--temperature=0.8"
]

try:
    subprocess.run(model_self_play_command, check=True)
    print("使用模型自我对弈成功！")
except subprocess.CalledProcessError:
    print("错误: 使用模型自我对弈失败")
    sys.exit(1)

# 第二次训练模型（使用模型生成的数据）
print("\n===== 步骤4: 第二次模型训练 =====")
print("这将使用模型自我对弈生成的数据进一步训练模型")
input("按Enter键继续...")

second_train_command = [
    sys.executable, "train.py",
    "--data_dir=data",
    "--epochs=20",
    "--batch_size=32",
    "--output_dir=models",
    "--model_name=improved_model",
    "--learning_rate=0.0001"
]

try:
    subprocess.run(second_train_command, check=True)
    print("模型二次训练成功！")
except subprocess.CalledProcessError:
    print("错误: 模型二次训练失败")
    sys.exit(1)

# 评估模型性能
print("\n===== 步骤5: 评估模型性能 =====")
print("这将评估训练后的模型与随机策略的对战表现")
input("按Enter键继续...")

evaluate_command = [
    sys.executable, "evaluate.py",
    "--model_path=models/improved_model.h5",
    "--num_games=10",
    "--opponent_type=random"
]

try:
    subprocess.run(evaluate_command, check=True)
    print("模型评估完成！")
except subprocess.CalledProcessError:
    print("错误: 模型评估失败")
    sys.exit(1)

# 提示用户与AI对战
print("\n===== 五子棋AI训练示例完成 =====")
print("\n现在你可以与训练好的AI对战了！")
print("\n执行以下命令开始游戏：")
print(f"{sys.executable} play.py --model_path=models/improved_model.h5")
print("\n你也可以继续优化模型：")
print("1. 使用improved_model进行更多自我对弈")
print("2. 使用新生成的数据再次训练模型")
print("3. 比较不同版本模型的性能")
print("\n祝你游戏愉快！")