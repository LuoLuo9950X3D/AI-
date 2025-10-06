import numpy as np
import tensorflow as tf
import argparse
import time
from gobang_game import GobangGame
from model import GobangModel

class Evaluator:
    def __init__(self, board_size=15):
        """初始化评估器"""
        self.board_size = board_size
        self.game = GobangGame(board_size)
        
    def random_player(self):
        """随机玩家，随机选择合法位置落子"""
        valid_moves = self.game.get_valid_moves()
        if not valid_moves:
            return None
        return valid_moves[np.random.randint(len(valid_moves))]
    
    def human_player(self):
        """人类玩家，通过命令行输入坐标"""
        while True:
            try:
                move = input("请输入落子坐标 (行 列，0-14): ").strip()
                r, c = map(int, move.split())
                if (r, c) in self.game.get_valid_moves():
                    return (r, c)
                else:
                    print("无效的落子位置，请重新输入。")
            except ValueError:
                print("输入格式错误，请输入两个整数，用空格分隔。")
    
    def model_player(self, model, temperature=0.1):
        """AI玩家，使用模型预测最佳落子位置"""
        valid_moves = self.game.get_valid_moves()
        if not valid_moves:
            return None
        
        # 获取当前局面特征
        board_feature = self.game.get_board_feature()
        
        # 使用模型预测
        policy_2d, _ = model.predict(board_feature)
        
        # 过滤出合法位置的概率
        valid_probs = []
        valid_move_indices = []
        
        for i, (r, c) in enumerate(valid_moves):
            prob = policy_2d[r, c]
            valid_probs.append(prob)
            valid_move_indices.append(i)
        
        # 根据温度参数调整概率分布
        valid_probs = np.array(valid_probs)
        if temperature > 0:
            valid_probs = valid_probs ** (1.0 / temperature)
            valid_probs = valid_probs / np.sum(valid_probs)
        else:
            # 贪婪策略
            valid_probs = np.zeros_like(valid_probs)
            valid_probs[np.argmax(valid_probs)] = 1.0
        
        # 根据概率分布选择落子位置
        chosen_idx = np.random.choice(valid_move_indices, p=valid_probs)
        return valid_moves[chosen_idx]
    
    def play_game(self, player1, player2, verbose=False):
        """进行一局游戏"""
        self.game.reset()
        move_count = 0
        
        # 记录游戏开始时间
        start_time = time.time()
        
        while not self.game.game_over:
            if verbose and move_count % 5 == 0:
                # 每5步打印一次棋盘
                self.game.print_board()
            
            # 选择当前玩家
            current_player = player1 if self.game.current_player == 1 else player2
            
            # 获取落子位置
            move = current_player()
            
            if move is None:
                break
            
            # 落子
            self.game.make_move(*move)
            move_count += 1
            
            # 检查游戏是否结束
            if self.game.game_over:
                if verbose:
                    self.game.print_board()
                    if self.game.winner == 1:
                        print("黑棋获胜！")
                    elif self.game.winner == 2:
                        print("白棋获胜！")
                    else:
                        print("平局！")
                break
        
        # 计算游戏用时
        elapsed_time = time.time() - start_time
        
        return self.game.winner, move_count, elapsed_time
    
    def evaluate_model(self, model_path, num_games=10, opponent_type="random", verbose=False):
        """评估模型性能"""
        # 加载模型
        model = GobangModel(self.board_size, model_path)
        
        # 定义对手类型
        if opponent_type == "random":
            opponent = self.random_player
        elif opponent_type == "human":
            opponent = self.human_player
        else:
            raise ValueError(f"不支持的对手类型: {opponent_type}")
        
        # 记录结果
        results = {1: 0, 2: 0, 0: 0}  # 1:黑棋胜，2:白棋胜，0:平局
        total_moves = 0
        total_time = 0
        
        print(f"开始评估模型: {model_path}")
        print(f"对手类型: {opponent_type}")
        print(f"共进行{num_games}局游戏...")
        
        for i in range(num_games):
            # 交替让模型执黑和执白
            if i % 2 == 0:
                # 模型执黑
                winner, moves, elapsed = self.play_game(
                    lambda: self.model_player(model), 
                    opponent, 
                    verbose
                )
                if verbose:
                    print(f"局 {i+1}: 模型执黑，", end="")
            else:
                # 模型执白
                winner, moves, elapsed = self.play_game(
                    opponent, 
                    lambda: self.model_player(model), 
                    verbose
                )
                if verbose:
                    print(f"局 {i+1}: 模型执白，", end="")
            
            # 记录结果
            results[winner] += 1
            total_moves += moves
            total_time += elapsed
            
            # 打印当前局的结果
            if verbose:
                if winner == 1:
                    print("黑棋获胜")
                elif winner == 2:
                    print("白棋获胜")
                else:
                    print("平局")
                print(f"步数: {moves}, 用时: {elapsed:.2f}秒")
            
            # 打印进度
            if (i + 1) % 5 == 0:
                print(f"已完成{i + 1}/{num_games}局")
        
        # 计算胜率
        total_wins = results[1] if opponent_type == "random" else results[2]
        # 如果模型交替执黑执白，需要计算净胜
        if num_games > 1:
            # 模型执黑获胜次数
            black_wins = (results[1] + (results[1] - results[2])) // 2
            # 模型执白获胜次数
            white_wins = (results[2] + (results[2] - results[1])) // 2
            model_wins = black_wins + white_wins
        else:
            model_wins = results[1] if i % 2 == 0 else results[2]
        
        win_rate = model_wins / num_games if num_games > 0 else 0
        avg_moves = total_moves / num_games if num_games > 0 else 0
        avg_time = total_time / num_games if num_games > 0 else 0
        
        # 打印评估结果
        print("\n评估结果:")
        print(f"总游戏局数: {num_games}")
        print(f"模型获胜: {model_wins}局")
        print(f"胜率: {win_rate:.2%}")
        print(f"平均步数: {avg_moves:.2f}")
        print(f"平均每局用时: {avg_time:.2f}秒")
        print(f"详细结果: 黑棋胜{results[1]}局，白棋胜{results[2]}局，平局{results[0]}局")
        
        # 返回评估指标
        metrics = {
            "win_rate": win_rate,
            "model_wins": model_wins,
            "total_games": num_games,
            "avg_moves": avg_moves,
            "avg_time": avg_time,
            "detailed_results": results
        }
        
        return metrics
    
    def compare_models(self, model1_path, model2_path, num_games=10, verbose=False):
        """比较两个模型的性能"""
        # 加载两个模型
        model1 = GobangModel(self.board_size, model1_path)
        model2 = GobangModel(self.board_size, model2_path)
        
        # 记录结果
        model1_wins = 0
        model2_wins = 0
        draws = 0
        total_moves = 0
        total_time = 0
        
        print(f"开始比较两个模型:")
        print(f"模型1: {model1_path}")
        print(f"模型2: {model2_path}")
        print(f"共进行{num_games}局游戏...")
        
        for i in range(num_games):
            # 交替让两个模型执黑
            if i % 2 == 0:
                # 模型1执黑
                winner, moves, elapsed = self.play_game(
                    lambda: self.model_player(model1), 
                    lambda: self.model_player(model2), 
                    verbose
                )
                if verbose:
                    print(f"局 {i+1}: 模型1执黑，", end="")
            else:
                # 模型2执黑
                winner, moves, elapsed = self.play_game(
                    lambda: self.model_player(model2), 
                    lambda: self.model_player(model1), 
                    verbose
                )
                if verbose:
                    print(f"局 {i+1}: 模型2执黑，", end="")
            
            # 记录结果
            if i % 2 == 0:
                if winner == 1:
                    model1_wins += 1
                elif winner == 2:
                    model2_wins += 1
                else:
                    draws += 1
            else:
                if winner == 1:
                    model2_wins += 1
                elif winner == 2:
                    model1_wins += 1
                else:
                    draws += 1
            
            total_moves += moves
            total_time += elapsed
            
            # 打印当前局的结果
            if verbose:
                if winner == 1:
                    print(f"黑棋获胜")
                elif winner == 2:
                    print(f"白棋获胜")
                else:
                    print(f"平局")
                print(f"步数: {moves}, 用时: {elapsed:.2f}秒")
            
            # 打印进度
            if (i + 1) % 5 == 0:
                print(f"已完成{i + 1}/{num_games}局")
        
        # 计算胜率和平均指标
        model1_win_rate = model1_wins / num_games if num_games > 0 else 0
        model2_win_rate = model2_wins / num_games if num_games > 0 else 0
        avg_moves = total_moves / num_games if num_games > 0 else 0
        avg_time = total_time / num_games if num_games > 0 else 0
        
        # 打印比较结果
        print("\n比较结果:")
        print(f"总游戏局数: {num_games}")
        print(f"模型1获胜: {model1_wins}局 ({model1_win_rate:.2%})")
        print(f"模型2获胜: {model2_wins}局 ({model2_win_rate:.2%})")
        print(f"平局: {draws}局")
        print(f"平均步数: {avg_moves:.2f}")
        print(f"平均每局用时: {avg_time:.2f}秒")
        
        # 确定更优模型
        if model1_wins > model2_wins:
            print(f"结论: 模型1 ({model1_path}) 表现优于模型2 ({model2_path})")
        elif model2_wins > model1_wins:
            print(f"结论: 模型2 ({model2_path}) 表现优于模型1 ({model1_path})")
        else:
            print("结论: 两个模型表现相当")
        
        # 返回比较指标
        metrics = {
            "model1_wins": model1_wins,
            "model2_wins": model2_wins,
            "draws": draws,
            "total_games": num_games,
            "model1_win_rate": model1_win_rate,
            "model2_win_rate": model2_win_rate,
            "avg_moves": avg_moves,
            "avg_time": avg_time
        }
        
        return metrics

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估五子棋AI模型性能')
    parser.add_argument('--board_size', type=int, default=15, help='棋盘大小')
    parser.add_argument('--model_path', type=str, required=True, help='AI模型路径')
    parser.add_argument('--num_games', type=int, default=10, help='评估的游戏局数')
    parser.add_argument('--opponent_type', type=str, default='random', choices=['random', 'human'], help='对手类型')
    parser.add_argument('--compare_model', type=str, default=None, help='用于比较的第二个模型路径')
    parser.add_argument('--verbose', action='store_true', help='打印详细信息')
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = Evaluator(board_size=args.board_size)
    
    if args.compare_model:
        # 比较两个模型
        evaluator.compare_models(
            model1_path=args.model_path,
            model2_path=args.compare_model,
            num_games=args.num_games,
            verbose=args.verbose
        )
    else:
        # 评估单个模型
        evaluator.evaluate_model(
            model_path=args.model_path,
            num_games=args.num_games,
            opponent_type=args.opponent_type,
            verbose=args.verbose
        )

if __name__ == "__main__":
    main()