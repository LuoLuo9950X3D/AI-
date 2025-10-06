import os
import numpy as np
import tensorflow as tf
import argparse
import os
import time
from datetime import datetime
import threading
from gobang_game import GobangGame
from model import GobangModel

# 设置随机种子
tf.random.set_seed(42)
np.random.seed(42)

# 创建保存数据的目录
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

class MCTSNode:
    """蒙特卡洛树搜索节点"""
    def __init__(self, state, game, model, parent=None, move=None):
        self.state = state  # 当前棋盘状态
        self.game = game    # 游戏实例
        self.model = model  # AI模型
        self.parent = parent  # 父节点
        self.move = move    # 到达此节点的移动
        
        self.children = {}  # 子节点字典
        self.visit_count = 0  # 访问次数
        self.value_sum = 0.0  # 价值总和
        self.prior = 0.0  # 先验概率
        
        # 预先计算合法移动
        self.valid_moves = None
        self.policy_probs = None
        
    def is_leaf(self):
        """判断是否为叶子节点"""
        return len(self.children) == 0
        
    def expand(self):
        """扩展节点"""
        # 复制游戏状态
        game_copy = GobangGame(self.game.board_size)
        game_copy.board = self.state.copy()
        game_copy.current_player = self.game.current_player
        game_copy.winner = self.game.winner
        game_copy.game_over = self.game.game_over
        
        # 获取合法移动
        self.valid_moves = game_copy.get_valid_moves()
        
        if not self.valid_moves or game_copy.game_over:
            return
            
        # 使用模型预测或随机策略
        value = 0.0  # 默认价值为0
        if self.model is not None:
            # 使用模型预测
            board_feature = game_copy.get_board_feature()
            policy_2d, value = self.model.predict(board_feature)
        else:
            # 无模型时使用随机策略
            policy_2d = np.zeros((self.game.board_size, self.game.board_size))
            for move in self.valid_moves:
                policy_2d[move] = 1.0
        
        # 转换为先验概率
        self.policy_probs = {}
        for move in self.valid_moves:
            r, c = move
            self.policy_probs[move] = policy_2d[r, c]
        
        # 归一化概率
        total_prob = sum(self.policy_probs.values())
        if total_prob > 0:
            for move in self.policy_probs:
                self.policy_probs[move] /= total_prob
        
        # 创建子节点
        for move in self.valid_moves:
            # 复制游戏状态用于子节点
            child_game = GobangGame(self.game.board_size)
            child_game.board = self.state.copy()
            child_game.current_player = self.game.current_player
            child_game.winner = self.game.winner
            child_game.game_over = self.game.game_over
            
            # 在子节点游戏状态中落子
            child_game.make_move(*move)
            
            # 创建子节点
            self.children[move] = MCTSNode(
                state=child_game.board.copy(),
                game=child_game,
                model=self.model,
                parent=self,
                move=move
            )
            
            # 设置先验概率
            self.children[move].prior = self.policy_probs.get(move, 0.0)
        
        return value
        
    def select(self, c_puct=1.0):
        """选择子节点"""
        best_score = -float('inf')
        best_child = None
        
        # 计算父节点的总访问次数的平方根
        parent_visit_sqrt = np.sqrt(self.visit_count + 1)
        
        for move, child in self.children.items():
            # 如果子节点还没有被访问过，优先选择先验概率高的
            if child.visit_count == 0:
                u = c_puct * child.prior * parent_visit_sqrt
                score = u
            else:
                # 计算UCB值
                q = child.value_sum / child.visit_count
                u = c_puct * child.prior * parent_visit_sqrt / (1 + child.visit_count)
                score = q + u
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
        
    def update(self, value):
        """更新节点值"""
        self.visit_count += 1
        self.value_sum += value

class SelfPlay:
    def __init__(self, board_size=15, model_path=None, num_simulations=200, c_puct=1.5):
        """初始化自我对弈器"""
        self.board_size = board_size
        self.game = GobangGame(board_size)
        self.model = GobangModel(board_size, model_path) if model_path else None
        self.num_simulations = num_simulations
        self.c_puct = c_puct  # MCTS探索参数
        self.max_game_length = board_size * board_size  # 最大步数
        
    def mcts_search(self, temperature=1.0):
        """蒙特卡洛树搜索"""
        if self.model is None:
            # 如果没有模型，随机选择合法位置
            valid_moves = self.game.get_valid_moves()
            if not valid_moves:
                return None
            return valid_moves[np.random.randint(len(valid_moves))]
        
        # 创建根节点
        root = MCTSNode(
            state=self.game.board.copy(),
            game=self.game,
            model=self.model
        )
        
        # 执行多次模拟
        for _ in range(self.num_simulations):
            node = root
            
            # 选择阶段
            while not node.is_leaf() and not node.game.game_over:
                node = node.select(self.c_puct)
            
            # 扩展阶段和评估阶段
            if not node.game.game_over:
                value = node.expand()
            else:
                # 游戏已结束，直接评估结果
                if node.game.winner == 1:
                    value = 1.0
                elif node.game.winner == 2:
                    value = -1.0
                else:
                    value = 0.0
            
            # 反向传播阶段
            current_player = node.game.current_player
            while node is not None:
                # 根据当前玩家调整价值
                if current_player != node.game.current_player:
                    value = -value
                node.update(value)
                node = node.parent
        
        # 根据访问次数选择移动
        move_visits = {move: child.visit_count for move, child in root.children.items()}
        total_visits = sum(move_visits.values())
        
        if total_visits == 0:
            # 如果没有访问任何子节点，随机选择
            valid_moves = list(root.children.keys())
            return valid_moves[np.random.randint(len(valid_moves))]
        
        # 应用温度参数
        if temperature > 0:
            # 根据温度调整概率
            move_probs = {move: (visit_count ** (1.0 / temperature)) for move, visit_count in move_visits.items()}
            total_probs = sum(move_probs.values())
            move_probs = {move: prob / total_probs for move, prob in move_probs.items()}
            
            # 按概率选择
            moves = list(move_probs.keys())
            probs = list(move_probs.values())
            chosen_move = moves[np.random.choice(len(moves), p=probs)]
        else:
            # 贪婪选择
            chosen_move = max(move_visits, key=move_visits.get)
        
        return chosen_move
    
    def play_game(self, temperature=1.0):
        """进行一局自我对弈"""
        self.game.reset()
        game_history = []
        move_count = 0
        
        start_time = time.time()
        
        while not self.game.game_over and move_count < self.max_game_length:
            # 获取当前局面特征
            board_feature = self.game.get_board_feature().copy()
            current_player = self.game.current_player
            
            # 使用MCTS搜索最佳落子位置
            move = self.mcts_search(temperature)
            
            if move is None:
                break
            
            # 创建根节点获取搜索概率分布
            root = MCTSNode(
                state=self.game.board.copy(),
                game=self.game,
                model=self.model
            )
            
            for _ in range(min(self.num_simulations // 2, 50)):
                node = root
                while not node.is_leaf() and not node.game.game_over:
                    node = node.select(self.c_puct)
                
                if not node.game.game_over:
                    node.expand()
            
            # 计算搜索概率分布
            search_probs = np.zeros(self.board_size * self.board_size)
            if hasattr(root, 'children'):
                total_visits = sum(child.visit_count for child in root.children.values())
                if total_visits > 0:
                    for move_pos, child in root.children.items():
                        idx = move_pos[0] * self.board_size + move_pos[1]
                        search_probs[idx] = child.visit_count / total_visits
            
            # 记录落子前的局面、搜索概率和玩家
            game_history.append((board_feature, search_probs, current_player))
            
            # 落子
            self.game.make_move(*move)
            move_count += 1
        
        end_time = time.time()
        game_duration = end_time - start_time
        
        # 确定游戏结果
        if self.game.winner == 1:
            result = 1
        elif self.game.winner == 2:
            result = -1
        else:
            result = 0
        
        # 为每个局面生成对应的价值标签
        training_data = []
        for board_feature, search_probs, player in game_history:
            # 计算该局面的玩家视角的价值
            value = result if player == 1 else -result
            
            training_data.append((board_feature, search_probs, value))
        
        # 返回训练数据、胜者、步数和游戏时长
        return training_data, self.game.winner, move_count, game_duration
    
    def generate_training_data(self, num_games=100, initial_temperature=1.0, final_temperature=0.1):
        """生成训练数据"""
        all_training_data = []
        results = {1: 0, 2: 0, 0: 0}  # 统计胜负情况
        move_counts = []  # 记录每局步数
        durations = []  # 记录每局时长
        
        print(f"开始自我对弈，共{num_games}局...")
        start_time = time.time()
        
        for i in range(num_games):
            # 线性降低温度参数，使策略从探索逐渐转向利用
            temperature = initial_temperature + (final_temperature - initial_temperature) * (i / (num_games - 1)) if num_games > 1 else initial_temperature
            
            # 进行一局自我对弈
            game_data, winner, move_count, duration = self.play_game(temperature)
            all_training_data.extend(game_data)
            results[winner] += 1
            move_counts.append(move_count)
            durations.append(duration)
            
            # 打印进度
            if (i + 1) % 5 == 0:
                elapsed_time = time.time() - start_time
                avg_move_count = np.mean(move_counts) if move_counts else 0
                avg_duration = np.mean(durations) if durations else 0
                
                print(f"已完成{i + 1}/{num_games}局，" \
                      f"结果统计: 黑棋胜{results[1]}局({results[1]/(i+1)*100:.1f}%), " \
                      f"白棋胜{results[2]}局({results[2]/(i+1)*100:.1f}%), " \
                      f"平局{results[0]}局({results[0]/(i+1)*100:.1f}%) | " \
                      f"平均步数: {avg_move_count:.1f}, " \
                      f"平均时长: {avg_duration:.2f}秒 | " \
                      f"总耗时: {elapsed_time:.1f}秒")
        
        # 打乱数据顺序
        np.random.shuffle(all_training_data)
        
        total_time = time.time() - start_time
        avg_move_count = np.mean(move_counts) if move_counts else 0
        avg_duration = np.mean(durations) if durations else 0
        
        print(f"\n自我对弈完成！")
        print(f"总对局数: {num_games}")
        print(f"黑棋胜: {results[1]}局({results[1]/num_games*100:.1f}%)")
        print(f"白棋胜: {results[2]}局({results[2]/num_games*100:.1f}%)")
        print(f"平局: {results[0]}局({results[0]/num_games*100:.1f}%)")
        print(f"生成训练数据量: {len(all_training_data)}条")
        print(f"平均每局步数: {avg_move_count:.1f}")
        print(f"平均每局时长: {avg_duration:.2f}秒")
        print(f"总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
        
        return all_training_data
    
    def save_training_data(self, training_data, output_file=None):
        """保存训练数据到文件"""
        if not output_file:
            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(DATA_DIR, f"self_play_{timestamp}.npz")
        
        # 将数据分为三个部分
        x_data = []
        policy_data = []
        value_data = []
        
        for x, p, v in training_data:
            x_data.append(x)
            policy_data.append(p)
            value_data.append(v)
        
        # 转换为numpy数组
        x_data = np.array(x_data)
        policy_data = np.array(policy_data)
        value_data = np.array(value_data)
        
        # 添加元数据
        metadata = {
            'board_size': self.board_size,
            'num_samples': len(training_data),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_used': self.model is not None
        }
        
        # 保存数据
        np.savez(output_file, 
                 x=x_data, 
                 policy=policy_data, 
                 value=value_data,
                 metadata=metadata)
        
        print(f"训练数据已保存到: {output_file}")
        print(f"数据形状 - 特征: {x_data.shape}, 策略: {policy_data.shape}, 价值: {value_data.shape}")
        
        return output_file
        
    def load_training_data(self, file_path):
        """加载训练数据"""
        if not os.path.exists(file_path):
            print(f"错误: 文件{file_path}不存在")
            return None
        
        try:
            data = np.load(file_path)
            x_data = data['x']
            policy_data = data['policy']
            value_data = data['value']
            
            print(f"已加载训练数据: {file_path}")
            print(f"数据形状 - 特征: {x_data.shape}, 策略: {policy_data.shape}, 价值: {value_data.shape}")
            
            # 提取元数据（如果有）
            if 'metadata' in data:
                metadata = data['metadata'].item()
                print("元数据:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")
            
            # 转换为训练数据格式
            training_data = list(zip(x_data, policy_data, value_data))
            return training_data
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='通过自我对弈生成五子棋训练数据')
    parser.add_argument('--board_size', type=int, default=15, help='棋盘大小')
    parser.add_argument('--num_games', type=int, default=100, help='自我对弈的局数')
    parser.add_argument('--model_path', type=str, default=None, help='AI模型路径')
    parser.add_argument('--output_dir', type=str, default=DATA_DIR, help='数据保存目录')
    parser.add_argument('--initial_temperature', type=float, default=1.0, help='初始温度参数，控制探索程度')
    parser.add_argument('--final_temperature', type=float, default=0.1, help='最终温度参数，控制探索程度')
    parser.add_argument('--num_simulations', type=int, default=200, help='每步的MCTS模拟次数')
    parser.add_argument('--c_puct', type=float, default=1.5, help='MCTS探索参数')
    parser.add_argument('--save_every', type=int, default=0, help='每多少局保存一次数据，0表示只在最后保存')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化自我对弈器
    self_play = SelfPlay(
        board_size=args.board_size,
        model_path=args.model_path,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct
    )
    
    # 根据是否保存中间结果决定生成方式
    if args.save_every > 0:
        # 分批生成并保存数据
        all_files = []
        remaining_games = args.num_games
        
        for batch_idx in range(0, args.num_games, args.save_every):
            batch_size = min(args.save_every, remaining_games)
            print(f"\n=== 批次 {batch_idx//args.save_every + 1}：生成{batch_size}局对弈 ===")
            
            # 计算当前批次的温度参数范围
            batch_initial_temp = args.initial_temperature + (args.final_temperature - args.initial_temperature) * (batch_idx / args.num_games)
            batch_final_temp = args.initial_temperature + (args.final_temperature - args.initial_temperature) * ((batch_idx + batch_size) / args.num_games)
            
            # 生成训练数据
            batch_data = self_play.generate_training_data(
                num_games=batch_size,
                initial_temperature=batch_initial_temp,
                final_temperature=batch_final_temp
            )
            
            # 保存批次数据
            batch_file = os.path.join(args.output_dir, f"self_play_batch_{batch_idx+1}-{batch_idx+batch_size}.npz")
            saved_file = self_play.save_training_data(batch_data, batch_file)
            all_files.append(saved_file)
            
            remaining_games -= batch_size
            
        print(f"\n所有批次已完成，共保存了{len(all_files)}个数据文件")
        for file in all_files:
            print(f"  - {file}")
    else:
        # 一次性生成所有数据
        training_data = self_play.generate_training_data(
            num_games=args.num_games,
            initial_temperature=args.initial_temperature,
            final_temperature=args.final_temperature
        )
        
        # 保存训练数据
        output_file = os.path.join(args.output_dir, f"self_play_data_{args.num_games}games_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz")
        self_play.save_training_data(training_data, output_file)

if __name__ == "__main__":
    main()