import numpy as np
import tensorflow as tf
import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import json
from gobang_game import GobangGame
from model import GobangModel
from self_play import SelfPlay
from evaluate import Evaluator

# 导入必要的库
import sys
from datetime import datetime

# 确保logs目录存在
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists('logs'):
    os.makedirs('logs', exist_ok=True)

# 加载优化配置
def load_optimization_config(config_path='training_optimization_config.json'):
    """加载训练优化配置"""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
                # 处理新的配置文件结构，确保返回的配置包含所有需要的键
                result = {}
                
                # 从optimizer_config中提取learning_rate
                if 'optimizer_config' in config and 'learning_rate' in config['optimizer_config']:
                    result['learning_rate'] = config['optimizer_config']['learning_rate']
                else:
                    result['learning_rate'] = 0.001
                    
                # 从training_config中提取batch_size和epochs
                if 'training_config' in config:
                    if 'batch_size' in config['training_config']:
                        result['batch_size'] = config['training_config']['batch_size']
                    if 'epochs' in config['training_config']:
                        result['epochs'] = config['training_config']['epochs']
                else:
                    result['batch_size'] = 128
                    result['epochs'] = 100
                    
                # 从model_config中提取residual_blocks
                if 'model_config' in config and 'residual_blocks' in config['model_config']:
                    result['residual_blocks'] = config['model_config']['residual_blocks']
                else:
                    result['residual_blocks'] = 8
                    
                # 从loss_config中提取policy_weight和value_weight
                if 'loss_config' in config:
                    if 'policy_weight' in config['loss_config']:
                        result['policy_weight'] = config['loss_config']['policy_weight']
                    if 'value_weight' in config['loss_config']:
                        result['value_weight'] = config['loss_config']['value_weight']
                else:
                    result['policy_weight'] = 1.0
                    result['value_weight'] = 1.0
                    
                # 处理data_augmentation
                result['data_augmentation'] = config.get('data_augmentation', {
                    "horizontal_flip": True,
                    "vertical_flip": True,
                    "rotation": True,
                    "noise": True
                })
                
                return result
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    # 返回默认配置
    return {
        "learning_rate": 0.001,
        "batch_size": 128,
        "epochs": 100,
        "residual_blocks": 8,
        "policy_weight": 1.0,
        "value_weight": 1.0,
        "data_augmentation": {
            "horizontal_flip": True,
            "vertical_flip": True,
            "rotation": True,
            "noise": True
        }
    }

# 配置CUDA加速
def configure_gpu(memory_fraction=0.8):
    """配置GPU以进行高效训练"""
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # 设置GPU内存增长，避免一次性占用过多内存
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
                # 设置内存限制
                # tf.config.experimental.set_virtual_device_configuration(
                #     gpu,
                #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                # )
            print(f"CUDA加速已启用，检测到{len(physical_devices)}个GPU设备")
            return True
        except Exception as e:
            print(f"CUDA加速配置失败: {e}")
    else:
        print("未检测到GPU，将使用CPU进行训练")
    return False

class EnhancedDataGenerator:
    def __init__(self, board_size=15):
        """增强的数据生成器"""
        self.board_size = board_size
        self.game = GobangGame(board_size)
        self.model = None
    
    def set_model(self, model):
        """设置用于数据生成的模型"""
        self.model = model
    
    def generate_random_game(self):
        """生成随机对局数据"""
        self.game.reset()
        game_history = []
        
        while not self.game.game_over:
            valid_moves = self.game.get_valid_moves()
            if not valid_moves:
                break
            
            # 随机选择一个位置
            move = valid_moves[np.random.randint(len(valid_moves))]
            
            # 记录落子前的局面
            board_feature = self.game.get_board_feature().copy()
            game_history.append((board_feature, move))
            
            # 落子
            self.game.make_move(*move)
        
        return self._process_game_history(game_history)
    
    def generate_model_guided_game(self, temperature=1.0):
        """使用模型指导生成对局数据"""
        if not self.model:
            return self.generate_random_game()
        
        self.game.reset()
        game_history = []
        
        while not self.game.game_over:
            valid_moves = self.game.get_valid_moves()
            if not valid_moves:
                break
            
            # 使用模型预测
            board_feature = self.game.get_board_feature()
            policy_2d, _ = self.model.predict(board_feature)
            
            # 过滤合法位置的概率
            valid_probs = []
            valid_move_indices = []
            for i, (r, c) in enumerate(valid_moves):
                valid_probs.append(policy_2d[r, c])
                valid_move_indices.append(i)
            
            # 根据温度参数调整概率
            valid_probs = np.array(valid_probs)
            if temperature > 0:
                valid_probs = valid_probs ** (1.0 / temperature)
                valid_probs = valid_probs / np.sum(valid_probs)
            else:
                valid_probs = np.zeros_like(valid_probs)
                valid_probs[np.argmax(valid_probs)] = 1.0
            
            # 根据概率选择落子位置
            chosen_idx = np.random.choice(valid_move_indices, p=valid_probs)
            move = valid_moves[chosen_idx]
            
            # 记录落子前的局面
            board_feature = self.game.get_board_feature().copy()
            game_history.append((board_feature, move))
            
            # 落子
            self.game.make_move(*move)
        
        return self._process_game_history(game_history)
    
    def _process_game_history(self, game_history):
        """处理游戏历史，生成训练数据"""
        # 确定游戏结果
        if self.game.winner == 1:
            result = 1
        elif self.game.winner == 2:
            result = -1
        else:
            result = 0
        
        # 为每个局面生成对应的价值标签
        training_data = []
        for i, (board_feature, move) in enumerate(game_history):
            # 计算该局面的玩家视角的价值
            player = 1 if i % 2 == 0 else 2
            value = result if player == 1 else -result
            
            # 生成策略标签
            policy = np.zeros(self.board_size * self.board_size)
            policy[move[0] * self.board_size + move[1]] = 1.0
            
            training_data.append((board_feature, policy, value))
        
        return training_data
    
    def generate_training_data(self, num_games, use_model=False, temperature=1.0):
        """生成指定数量的训练数据"""
        all_training_data = []
        progress_interval = max(1, num_games // 10)
        
        for i in range(num_games):
            if use_model and self.model:
                # 逐渐降低温度参数
                current_temp = max(temperature * (1 - i / num_games), 0.1)
                game_data = self.generate_model_guided_game(current_temp)
            else:
                game_data = self.generate_random_game()
            
            all_training_data.extend(game_data)
            
            # 打印进度
            if (i + 1) % progress_interval == 0:
                print(f"已生成{i + 1}/{num_games}局数据，当前总数据量: {len(all_training_data)}")
        
        # 打乱数据顺序
        np.random.shuffle(all_training_data)
        
        # 分割为输入和标签
        x_data = np.array([data[0] for data in all_training_data])
        policy_data = np.array([data[1] for data in all_training_data])
        value_data = np.array([data[2] for data in all_training_data])
        
        return x_data, policy_data, value_data

def load_self_play_data(data_file):
    """加载自我对弈生成的数据"""
    if os.path.exists(data_file):
        print(f"加载自我对弈数据: {data_file}")
        try:
            data = np.load(data_file)
            x_data = data['x']
            policy_data = data['policy']
            value_data = data['value']
            print(f"成功加载{len(x_data)}条数据")
            return x_data, policy_data, value_data
        except Exception as e:
            print(f"加载数据失败: {e}")
    else:
        print(f"自我对弈数据文件不存在: {data_file}")
    return None, None, None

def train_with_battle_mode(model, board_size, num_battles, batch_size, use_gpu=False):
    """使用对战模式进行训练，改进版"""
    print(f"开始对战模式训练，共{num_battles}场对战...")
    game = GobangGame(board_size)
    battle_data = []
    battle_results = {1: 0, 2: 0, 0: 0}  # 记录对战结果
    
    # 创建一个临时模型用于对战
    temp_model = GobangModel(board_size)
    temp_model.model.set_weights(model.model.get_weights())
    
    # 随机选择温度参数范围
    initial_temperature = np.random.uniform(0.5, 1.0)
    final_temperature = np.random.uniform(0.1, 0.5)
    
    # 定义两个玩家函数，使用model.py中的新方法
    def player1_move():  # 当前模型
        current_move_count = len(game.move_history)
        # 根据当前步数动态调整温度参数
        current_temp = initial_temperature + (final_temperature - initial_temperature) * min(current_move_count / (board_size * board_size), 1.0)
        # 使用模型的带温度参数的预测方法
        return model.make_move(game.board, game.current_player, temperature=current_temp)
    
    def player2_move():  # 对手模型
        current_move_count = len(game.move_history)
        current_temp = initial_temperature + (final_temperature - initial_temperature) * min(current_move_count / (board_size * board_size), 1.0)
        return temp_model.make_move(game.board, game.current_player, temperature=current_temp)
    
    # 记录每局对战的时间
    total_battle_time = 0
    
    for i in range(num_battles):
        # 随机决定谁先手
        if np.random.random() < 0.5:
            players = [player1_move, player2_move]
            player_names = ["当前模型", "对手模型"]
        else:
            players = [player2_move, player1_move]
            player_names = ["对手模型", "当前模型"]
        
        # 重置游戏
        game.reset()
        game.move_history = []
        game.policy_history = []
        
        # 开始计时
        battle_start_time = time.time()
        
        # 进行对战
        while not game.game_over:
            current_player_idx = 0 if game.current_player == 1 else 1
            move_func = players[current_player_idx]
            
            try:
                move = move_func()
                if game.is_valid_move(*move):
                    # 记录落子前的局面特征
                    board_feature = game.get_board_feature().copy()
                    game.move_history.append((board_feature, game.current_player))
                    
                    # 记录策略分布
                    if current_player_idx == 0:  # 当前模型
                        policy_2d, _ = model.predict(board_feature, temperature=final_temperature)
                    else:  # 对手模型
                        policy_2d, _ = temp_model.predict(board_feature, temperature=final_temperature)
                    policy_flat = policy_2d.flatten()
                    game.policy_history.append(policy_flat)
                    
                    # 落子
                    game.make_move(*move)
                else:
                    # 如果AI选择了无效移动，随机选择一个有效移动
                    valid_moves = game.get_valid_moves()
                    if valid_moves:
                        move = valid_moves[np.random.randint(len(valid_moves))]
                        game.make_move(*move)
            except Exception as e:
                print(f"对战中出错: {e}")
                # 出错时随机选择一个有效移动
                valid_moves = game.get_valid_moves()
                if valid_moves:
                    move = valid_moves[np.random.randint(len(valid_moves))]
                    game.make_move(*move)
        
        # 结束计时
        battle_time = time.time() - battle_start_time
        total_battle_time += battle_time
        
        # 记录对战结果
        winner = game.winner
        battle_results[winner] += 1
        
        # 处理对战数据
        if len(game.move_history) > 0:
            # 为每个局面生成对应的价值标签
            for j, (board_feature, player) in enumerate(game.move_history):
                # 计算该局面的玩家视角的价值
                if winner == 0:  # 平局
                    value = 0
                else:
                    value = 1 if player == winner else -1
                
                # 获取对应的策略标签
                policy = game.policy_history[j] if j < len(game.policy_history) else None
                if policy is None:
                    # 如果没有策略历史，创建一个简化的策略标签
                    policy = np.zeros(board_size * board_size)
                    if j < len(game.move_history) - 1:
                        _, next_move_player = game.move_history[j+1]
                        if next_move_player == player:
                            # 这里需要更多信息来确定实际的移动位置
                            pass
                
                battle_data.append((board_feature, policy, value))
        
        # 定期更新临时模型，避免过拟合
        if (i + 1) % 10 == 0:
            temp_model.model.set_weights(model.model.get_weights())
            print(f"已更新对手模型，当前模型胜率: {battle_results[1] / (i + 1):.2%}")
            
        # 打印进度
        if (i + 1) % max(1, num_battles // 10) == 0:
            avg_time = total_battle_time / (i + 1)
            print(f"已完成{i + 1}/{num_battles}场对战，平均每局耗时: {avg_time:.2f}秒")
    
    # 输出对战统计
    print(f"\n对战模式训练统计:")
    print(f"- 当前模型胜场: {battle_results[1]}")
    print(f"- 对手模型胜场: {battle_results[2]}")
    print(f"- 平局: {battle_results[0]}")
    print(f"- 当前模型胜率: {battle_results[1] / num_battles:.2%}")
    
    if battle_data:
        # 处理对战数据
        x_data = np.array([data[0] for data in battle_data])
        policy_data = np.array([data[1] for data in battle_data])
        value_data = np.array([data[2] for data in battle_data])
        
        # 数据增强：翻转、旋转和噪声
        augmented_x, augmented_policy, augmented_value = [], [], []
        for bx, bp, bv in zip(x_data, policy_data, value_data):
            # 原始数据
            augmented_x.append(bx)
            augmented_policy.append(bp)
            augmented_value.append(bv)
            
            # 水平翻转
            flipped_bx = np.flip(bx, axis=1)
            flipped_bp = np.flip(bp.reshape(board_size, board_size), axis=1).flatten()
            augmented_x.append(flipped_bx)
            augmented_policy.append(flipped_bp)
            augmented_value.append(bv)
            
            # 垂直翻转
            v_flipped_bx = np.flip(bx, axis=0)
            v_flipped_bp = np.flip(bp.reshape(board_size, board_size), axis=0).flatten()
            augmented_x.append(v_flipped_bx)
            augmented_policy.append(v_flipped_bp)
            augmented_value.append(bv)
            
            # 旋转90度
            rotated_bx = np.rot90(bx, k=1, axes=(0, 1))
            rotated_bp = np.rot90(bp.reshape(board_size, board_size), k=1).flatten()
            augmented_x.append(rotated_bx)
            augmented_policy.append(rotated_bp)
            augmented_value.append(bv)
            
            # 旋转180度
            rotated_bx_180 = np.rot90(bx, k=2, axes=(0, 1))
            rotated_bp_180 = np.rot90(bp.reshape(board_size, board_size), k=2).flatten()
            augmented_x.append(rotated_bx_180)
            augmented_policy.append(rotated_bp_180)
            augmented_value.append(bv)
            
            # 添加轻微噪声（不影响游戏规则的情况下）
            noise = np.random.normal(0, 0.01, bx.shape)
            noise[bx != 0] = 0  # 只在空白位置添加噪声
            noisy_bx = bx + noise
            augmented_x.append(noisy_bx)
            augmented_policy.append(bp)
            augmented_value.append(bv)
        
        # 转换为numpy数组
        augmented_x = np.array(augmented_x)
        augmented_policy = np.array(augmented_policy)
        augmented_value = np.array(augmented_value)
        
        # 打乱数据
        indices = np.arange(len(augmented_x))
        np.random.shuffle(indices)
        augmented_x = augmented_x[indices]
        augmented_policy = augmented_policy[indices]
        augmented_value = augmented_value[indices]
        
        # 使用model.py中的新训练方法，带回调和验证
        print(f"使用{len(augmented_x)}条对战数据进行训练...")
        
        # 设置学习率回调
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=2, min_lr=1e-6
        )
        
        # 小规模训练，使用更少的轮数但更大的批次
        model.train(
            augmented_x,
            augmented_policy,
            augmented_value,
            epochs=3,
            batch_size=min(batch_size * 2, 128),  # 增大批次大小
            validation_split=0.1,
            callbacks=[lr_scheduler],
            verbose=1
        )
    
    return len(battle_data) if battle_data else 0

def train_model(args, has_gpu=False):
    """训练模型的主函数"""
    # 加载优化配置
    optimization_config = load_optimization_config()
    print("使用优化配置:", optimization_config)
    
    # 创建保存模型的目录
    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 创建checkpoints目录
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 初始化数据生成器
    data_gen = EnhancedDataGenerator(board_size=args.board_size)
    
    # 初始化模型 - 使用优化配置
    model_config = {
        'learning_rate': optimization_config['learning_rate'],
        'residual_blocks': optimization_config['residual_blocks'],
        'policy_weight': optimization_config['policy_weight'],
        'value_weight': optimization_config['value_weight']
    }
    
    model = GobangModel(
        board_size=args.board_size,
        model_path=args.model_path,
        config=model_config
    )
    
    # 设置数据生成器的模型
    data_gen.set_model(model)
    
    # 准备训练数据
    x_train, policy_train, value_train = None, None, None
    
    # 尝试加载自我对弈数据
    if args.use_self_play_data and args.self_play_file:
        x_train, policy_train, value_train = load_self_play_data(args.self_play_file)
    
    # 如果没有自我对弈数据，生成新数据
    if x_train is None:
        print(f"生成训练数据，共{args.num_games}局...")
        use_model_for_generation = args.model_path is not None
        x_train, policy_train, value_train = data_gen.generate_training_data(
            args.num_games, use_model=use_model_for_generation
        )
    
    # 如果数据不足，生成额外的随机数据
    min_data_size = max(5000, args.num_games * 10)  # 增加最小数据量要求
    if len(x_train) < min_data_size:
        print(f"数据不足，额外生成{min_data_size}局随机数据...")
        additional_x, additional_policy, additional_value = data_gen.generate_training_data(
            min_data_size, use_model=False
        )
        # 合并数据
        x_train = np.concatenate([x_train, additional_x]) if x_train.size > 0 else additional_x
        policy_train = np.concatenate([policy_train, additional_policy]) if policy_train.size > 0 else additional_policy
        value_train = np.concatenate([value_train, additional_value]) if value_train.size > 0 else additional_value
    
    print(f"总训练数据量: {len(x_train)}条")
    
    # 打印模型结构
    print("模型结构:")
    model.model.summary()
    
    # 设置早停策略 - 更激进的早停
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    
    # 设置学习率调度器 - 根据配置选择余弦退火
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6
    )
    
    # 设置模型检查点 - 更频繁地保存
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}_loss_{loss:.4f}.h5"),
        save_best_only=True,
        monitor='loss',
        mode='min',
        save_freq='epoch'
    )
    
    # 导入并使用TensorBoard记录学习率
    # 使用简单的日志目录配置
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir='logs',  # 使用简单的日志目录
        histogram_freq=0,  # 禁用直方图记录
        write_graph=False,  # 禁用图记录
        write_images=False,  # 禁用图像记录
        update_freq='epoch'
    )
    
    # 开始训练
    # 优先使用配置文件中的epochs和batch_size
    epochs = optimization_config.get('epochs', args.epochs)
    batch_size = optimization_config.get('batch_size', args.batch_size)
    
    print(f"开始训练模型，共{epochs}轮，批量大小：{batch_size}...")
    start_time = time.time()
    
    # 分段训练，每训练一定轮数后进行对战模式训练
    segment_size = max(5, epochs // 5)  # 每段训练的轮数
    total_segments = (epochs + segment_size - 1) // segment_size
    history = None
    
    # 使用优化的回调列表
    # 为了解决TensorBoard目录问题，暂时移除TensorBoard回调
    callbacks = [early_stopping, lr_scheduler, checkpoint]
    
    for segment in range(total_segments):
        current_epochs = min(segment_size, epochs - segment * segment_size)
        
        print(f"\n===== 训练段 {segment + 1}/{total_segments} (轮数: {segment * segment_size + 1}-{(segment + 1) * segment_size}) =====")
        
        # 训练当前段
        current_history = model.train(
            x_train, 
            policy_train,
            value_train,
            epochs=current_epochs,
            batch_size=batch_size,
            validation_split=args.validation_split,
            callbacks=callbacks,
            use_cosine_schedule=False  # 禁用余弦退火学习率调度，使用外部定义的学习率调度器
        )
        
        # 合并历史记录
        if history is None:
            history = current_history
        else:
            for key in history.history:
                history.history[key].extend(current_history.history[key])
        
        # 如果启用了对战模式，进行对战训练
        if args.use_battle_mode and segment < total_segments - 1:
            # 使用更大的批量和更多的对战次数
            battle_batch_size = min(batch_size, 128)
            battle_count = max(args.num_battles, 100)
            print(f"开始对战模式训练，批量大小：{battle_batch_size}，对战次数：{battle_count}...")
            battle_data_count = train_with_battle_mode(model, args.board_size, battle_count, battle_batch_size, use_gpu=has_gpu)
            print(f"对战模式训练完成，使用了{battle_data_count}条对战数据")
    
    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"\n训练完成！总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    
    # 保存最终模型
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"gobang_model_{args.epochs}epochs_{timestamp}.h5")
    model.save_model(model_path)
    print(f"模型已保存到: {model_path}")
    
    # 可视化训练过程
    if args.plot_history and history is not None:
        try:
            # 绘制损失曲线
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.plot(history.history['loss'])
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'])
                plt.legend(['Train', 'Validation'], loc='upper right')
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            
            if 'policy_loss' in history.history:
                plt.subplot(1, 3, 2)
                plt.plot(history.history['policy_loss'])
                if 'val_policy_loss' in history.history:
                    plt.plot(history.history['val_policy_loss'])
                    plt.legend(['Train', 'Validation'], loc='upper right')
                plt.title('Policy Loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
            
            if 'value_loss' in history.history:
                plt.subplot(1, 3, 3)
                plt.plot(history.history['value_loss'])
                if 'val_value_loss' in history.history:
                    plt.plot(history.history['val_value_loss'])
                    plt.legend(['Train', 'Validation'], loc='upper right')
                plt.title('Value Loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
            
            plt.tight_layout()
            plot_path = os.path.join(MODEL_DIR, f"training_history_{timestamp}.png")
            plt.savefig(plot_path)
            print(f"训练历史图已保存到 {plot_path}")
        except ImportError:
            print("无法绘制训练历史图，缺少 matplotlib 库")
    
    return model

def main():
    """主函数，解析命令行参数并启动训练"""
    parser = argparse.ArgumentParser(description='训练五子棋AI模型')
    parser.add_argument('--board_size', type=int, default=15, help='棋盘大小')
    parser.add_argument('--num_games', type=int, default=1000, help='用于生成训练数据的游戏局数')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--validation_split', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--model_path', type=str, default=None, help='预训练模型路径')
    parser.add_argument('--plot_history', action='store_true', help='是否绘制训练历史图')
    parser.add_argument('--use_self_play_data', action='store_true', help='是否使用自我对弈数据')
    parser.add_argument('--self_play_file', type=str, default=None, help='自我对弈数据文件路径')
    parser.add_argument('--use_battle_mode', action='store_true', help='是否启用对战模式训练')
    parser.add_argument('--num_battles', type=int, default=50, help='对战模式中的对战次数')
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.8, help='GPU内存使用比例')
    
    args = parser.parse_args()
    
    # 配置GPU
    has_gpu = configure_gpu(args.gpu_memory_fraction)
    
    # 训练模型
    train_model(args, has_gpu=has_gpu)

if __name__ == "__main__":
    main()