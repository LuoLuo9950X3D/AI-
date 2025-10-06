import os
import sys
import subprocess
import os

# 主菜单函数
def main_menu():
    """显示主菜单并处理用户选择"""
    while True:
        # 清屏（根据操作系统）
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=== AI五子棋 ===")
        print("1. 启动对战界面")
        print("2. 开始训练模型")
        print("3. 生成自我对弈数据")
        print("4. 评估模型性能")
        print("5. 比较两个模型")
        print("6. 退出")
        
        choice = input("请输入您的选择 (1-6): ")
        
        if choice == '1':
            start_game_ui()
        elif choice == '2':
            start_training()
        elif choice == '3':
            start_self_play()
        elif choice == '4':
            evaluate_model()
        elif choice == '5':
            compare_models()
        elif choice == '6':
            print("感谢使用AI五子棋！再见！")
            break
        else:
            input("无效的选择，请按Enter键继续...")

# 启动游戏界面
def start_game_ui():
    """启动五子棋对战界面"""
    try:
        # 检查TensorFlow环境
        try:
            import tensorflow as tf
            is_gpu_supported = tf.test.is_built_with_cuda()
            physical_devices = tf.config.list_physical_devices('GPU')
            has_gpu = len(physical_devices) > 0
            
            print("正在启动对战界面...")
            if has_gpu and not is_gpu_supported:
                print("\n注意: 检测到GPU设备，但当前TensorFlow版本不支持CUDA加速。")
                print("程序将使用CPU模式运行，游戏功能不受影响。")
                print("如需GPU加速，请安装支持CUDA的TensorFlow版本。\n")
            elif not has_gpu:
                print("\n注意: 未检测到GPU设备，程序将使用CPU模式运行。\n")
        except ImportError:
            print("警告: 无法导入TensorFlow，可能影响AI功能。\n")
            
        # 切换到trainable_gobang目录
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainable_gobang"))
        
        # 询问是否使用AI模型和设置参数
        use_ai = input("是否使用AI模型 (y/n, 默认: n): ")
        cmd_args = [sys.executable, "game_ui.py"]
        
        if use_ai.lower() == 'y':
            model_path = input("请输入AI模型路径 (默认: 自动选择最新模型): ")
            if model_path:
                cmd_args.extend(["--model_path", model_path])
            
            ai_strength = input("请设置AI强度 (0.0-1.0, 默认: 0.8): ")
            if ai_strength:
                cmd_args.extend(["--ai_strength", ai_strength])
        
        # 运行game_ui.py
        subprocess.run(cmd_args)
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
        board_size = input("棋盘大小 (默认: 15): ")
        validation_split = input("验证集比例 (默认: 0.1): ")
        model_path = input("预训练模型路径 (默认: 无): ")
        use_self_play_data = input("是否使用自我对弈数据 (y/n, 默认: n): ")
        
        # 构建命令参数
        cmd_args = [sys.executable, "train.py"]
        
        if epochs:
            cmd_args.extend(["--epochs", epochs])
        if batch_size:
            cmd_args.extend(["--batch_size", batch_size])
        if board_size:
            cmd_args.extend(["--board_size", board_size])
        if validation_split:
            cmd_args.extend(["--validation_split", validation_split])
        if model_path:
            cmd_args.extend(["--model_path", model_path])
        if use_self_play_data.lower() == 'y':
            cmd_args.append("--use_self_play_data")
            self_play_file = input("自我对弈数据文件路径 (例如: data/self_play_data_10games.npz): ")
            if self_play_file:
                cmd_args.extend(["--self_play_file", self_play_file])
        
        use_battle_mode = input("是否启用对战模式训练 (y/n, 默认: n): ")
        if use_battle_mode.lower() == 'y':
            cmd_args.append("--use_battle_mode")
            num_battles = input("对战模式中的对战次数 (默认: 50): ")
            if num_battles:
                cmd_args.extend(["--num_battles", num_battles])
        
        plot_history = input("是否绘制训练历史图 (y/n, 默认: n): ")
        if plot_history.lower() == 'y':
            cmd_args.append("--plot_history")
        
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
        board_size = input("棋盘大小 (默认: 15): ")
        model_path = input("AI模型路径 (默认: 无): ")
        num_simulations = input("每步模拟次数 (默认: 200): ")
        initial_temperature = input("初始温度参数 (默认: 1.0): ")
        final_temperature = input("最终温度参数 (默认: 0.1): ")
        c_puct = input("MCTS探索参数 (默认: 1.5): ")
        output_dir = input("数据保存目录 (默认: data/): ")
        
        # 构建命令参数
        cmd_args = [sys.executable, "self_play.py"]
        
        if num_games:
            cmd_args.extend(["--num_games", num_games])
        if board_size:
            cmd_args.extend(["--board_size", board_size])
        if model_path:
            cmd_args.extend(["--model_path", model_path])
        if num_simulations:
            cmd_args.extend(["--num_simulations", num_simulations])
        if initial_temperature:
            cmd_args.extend(["--initial_temperature", initial_temperature])
        if final_temperature:
            cmd_args.extend(["--final_temperature", final_temperature])
        if c_puct:
            cmd_args.extend(["--c_puct", c_puct])
        if output_dir:
            cmd_args.extend(["--output_dir", output_dir])
        
        save_every = input("每多少局保存一次数据 (默认: 0，只在最后保存): ")
        if save_every:
            cmd_args.extend(["--save_every", save_every])
        
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

# 评估模型性能
def evaluate_model():
    """评估AI模型性能"""
    try:
        print("正在准备评估模型...")
        # 切换到trainable_gobang目录
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainable_gobang"))
        
        print("请设置评估参数:")
        model_path = input("AI模型路径 (必填): ")
        if not model_path:
            print("错误: 模型路径不能为空")
            input("请按Enter键返回主菜单...")
            return
        
        board_size = input("棋盘大小 (默认: 15): ")
        num_games = input("评估的游戏局数 (默认: 10): ")
        opponent_type = input("对手类型 (random/human, 默认: random): ")
        
        # 构建命令参数
        cmd_args = [sys.executable, "evaluate.py"]
        cmd_args.extend(["--model_path", model_path])
        
        if board_size:
            cmd_args.extend(["--board_size", board_size])
        if num_games:
            cmd_args.extend(["--num_games", num_games])
        if opponent_type:
            cmd_args.extend(["--opponent_type", opponent_type])
        
        verbose = input("是否打印详细信息 (y/n, 默认: n): ")
        if verbose.lower() == 'y':
            cmd_args.append("--verbose")
        
        print(f"开始评估模型，参数: {cmd_args}")
        # 运行evaluate.py
        subprocess.run(cmd_args)
        
        input("评估完成，请按Enter键返回主菜单...")
    except Exception as e:
        print(f"评估模型失败: {str(e)}")
        input("请按Enter键返回主菜单...")
    finally:
        # 切换回原来的目录
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 比较两个模型
def compare_models():
    """比较两个AI模型的性能"""
    try:
        print("正在准备比较模型...")
        # 切换到trainable_gobang目录
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainable_gobang"))
        
        print("请设置比较参数:")
        model1_path = input("第一个模型路径 (必填): ")
        if not model1_path:
            print("错误: 第一个模型路径不能为空")
            input("请按Enter键返回主菜单...")
            return
        
        model2_path = input("第二个模型路径 (必填): ")
        if not model2_path:
            print("错误: 第二个模型路径不能为空")
            input("请按Enter键返回主菜单...")
            return
        
        board_size = input("棋盘大小 (默认: 15): ")
        num_games = input("比较的游戏局数 (默认: 10): ")
        
        # 构建命令参数
        cmd_args = [sys.executable, "evaluate.py"]
        cmd_args.extend(["--model_path", model1_path])
        cmd_args.extend(["--compare_model", model2_path])
        
        if board_size:
            cmd_args.extend(["--board_size", board_size])
        if num_games:
            cmd_args.extend(["--num_games", num_games])
        
        verbose = input("是否打印详细信息 (y/n, 默认: n): ")
        if verbose.lower() == 'y':
            cmd_args.append("--verbose")
        
        print(f"开始比较模型，参数: {cmd_args}")
        # 运行evaluate.py
        subprocess.run(cmd_args)
        
        input("比较完成，请按Enter键返回主菜单...")
    except Exception as e:
        print(f"比较模型失败: {str(e)}")
        input("请按Enter键返回主菜单...")
    finally:
        # 切换回原来的目录
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 创建必要的目录
def create_necessary_dirs():
    """创建训练过程中需要的目录"""
    dirs = [
        "trainable_gobang/models",
        "trainable_gobang/checkpoints",
        "trainable_gobang/data",
        "trainable_gobang/logs"
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