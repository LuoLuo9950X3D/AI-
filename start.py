import os
import sys
import subprocess

# 主菜单函数
def main_menu():
    """显示主菜单并处理用户选择"""
    while True:
        # 清屏（根据操作系统）
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=== AI五子棋 ==")
        print("1. 启动对战界面")
        print("2. 开始训练模型")
        print("3. 生成自我对弈数据")
        print("4. 退出")
        
        choice = input("请输入您的选择 (1-4): ")
        
        if choice == '1':
            start_game_ui()
        elif choice == '2':
            start_training()
        elif choice == '3':
            start_self_play()
        elif choice == '4':
            print("感谢使用AI五子棋！再见！")
            break
        else:
            input("无效的选择，请按Enter键继续...")

# 启动游戏界面
def start_game_ui():
    """启动五子棋对战界面"""
    try:
        print("正在启动对战界面...")
        # 切换到trainable_gobang目录
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainable_gobang"))
        # 运行game_ui.py
        subprocess.run([sys.executable, "game_ui.py"])
    except Exception as e:
        print(f"启动对战界面失败: {str(e)}")
        input("请按Enter键返回主菜单...")
    finally:
        # 切换回原来的目录
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 启动训练
def start_training():
    """启动模型训练"""
    try:
        print("正在准备训练模型...")
        # 切换到trainable_gobang目录
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainable_gobang"))
        
        print("请设置训练参数:")
        epochs = input("训练轮数 (默认: 50): ")
        batch_size = input("批次大小 (默认: 64): ")
        learning_rate = input("学习率 (默认: 0.001): ")
        use_gpu = input("是否使用GPU加速 (y/n, 默认: y): ")
        
        # 构建命令参数
        cmd_args = [sys.executable, "train.py"]
        
        if epochs:
            cmd_args.extend(["--epochs", epochs])
        if batch_size:
            cmd_args.extend(["--batch-size", batch_size])
        if learning_rate:
            cmd_args.extend(["--learning-rate", learning_rate])
        if use_gpu.lower() == 'n':
            cmd_args.extend(["--no-gpu"])
        
        print(f"开始训练，参数: {cmd_args}")
        # 运行train.py
        subprocess.run(cmd_args)
        
        input("训练完成，请按Enter键返回主菜单...")
    except Exception as e:
        print(f"启动训练失败: {str(e)}")
        input("请按Enter键返回主菜单...")
    finally:
        # 切换回原来的目录
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 启动自我对弈
def start_self_play():
    """启动自我对弈数据生成"""
    try:
        print("正在准备自我对弈...")
        # 切换到trainable_gobang目录
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainable_gobang"))
        
        print("请设置自我对弈参数:")
        num_games = input("对弈局数 (默认: 100): ")
        simulations = input("每步模拟次数 (默认: 200): ")
        
        # 构建命令参数
        cmd_args = [sys.executable, "self_play.py"]
        
        if num_games:
            cmd_args.extend(["--num-games", num_games])
        if simulations:
            cmd_args.extend(["--simulations", simulations])
        
        print(f"开始自我对弈，参数: {cmd_args}")
        # 运行self_play.py
        subprocess.run(cmd_args)
        
        input("自我对弈完成，请按Enter键返回主菜单...")
    except Exception as e:
        print(f"启动自我对弈失败: {str(e)}")
        input("请按Enter键返回主菜单...")
    finally:
        # 切换回原来的目录
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 创建必要的目录
def create_necessary_dirs():
    """创建训练过程中需要的目录"""
    dirs = [
        "models",
        "training_data",
        "training_data/self_play",
        "training_data/human_vs_ai"
    ]
    
    for dir_name in dirs:
        dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"创建目录: {dir_path}")

# 主函数
if __name__ == "__main__":
    print("欢迎使用AI五子棋！")
    
    # 创建必要的目录
    create_necessary_dirs()
    
    # 显示主菜单
    main_menu()