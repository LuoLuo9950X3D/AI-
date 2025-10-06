import numpy as np
import tensorflow as tf
import argparse
from gobang_game import GobangGame
from model import GobangModel

class GobangPlayer:
    def __init__(self, board_size=15, model_path=None, ai_strength=1.0):
        """初始化五子棋玩家"""
        self.board_size = board_size
        self.game = GobangGame(board_size)
        self.model = GobangModel(board_size, model_path) if model_path else None
        self.ai_strength = ai_strength  # 0.0-1.0，控制AI强度
        
    def get_human_move(self):
        """获取人类玩家的落子"""
        while True:
            try:
                move_input = input("请输入落子位置 (行 列，0-14): ").strip()
                # 支持退出命令
                if move_input.lower() in ['q', 'quit', 'exit']:
                    return None
                
                r, c = map(int, move_input.split())
                
                # 检查输入是否在有效范围内
                if r < 0 or r >= self.board_size or c < 0 or c >= self.board_size:
                    print(f"输入超出范围，请输入0到{self.board_size-1}之间的数字。")
                    continue
                
                # 检查位置是否为空
                if (r, c) in self.game.get_valid_moves():
                    return (r, c)
                else:
                    print("该位置已有棋子，请选择其他位置。")
            except ValueError:
                print("输入格式错误，请输入两个整数，用空格分隔。例如：7 7")
    
    def get_ai_move(self):
        """获取AI的落子"""
        valid_moves = self.game.get_valid_moves()
        
        if not valid_moves:
            return None
        
        # 如果没有加载模型，随机选择一个位置
        if self.model is None:
            return valid_moves[np.random.randint(len(valid_moves))]
        
        # 根据AI强度调整落子策略
        # 高强度：主要使用模型预测
        # 低强度：增加随机因素
        if np.random.random() > self.ai_strength:
            # 随机落子
            return valid_moves[np.random.randint(len(valid_moves))]
        
        # 使用模型预测最佳落子位置
        board_feature = self.game.get_board_feature()
        policy_2d, _ = self.model.predict(board_feature)
        
        # 过滤出合法位置的概率
        valid_probs = []
        valid_move_indices = []
        
        for i, (r, c) in enumerate(valid_moves):
            prob = policy_2d[r, c]
            valid_probs.append(prob)
            valid_move_indices.append(i)
        
        # 调整概率分布，使最强的几个选择有更高的概率被选中
        valid_probs = np.array(valid_probs)
        temperature = 0.1  # 低温度使策略更确定
        valid_probs = valid_probs ** (1.0 / temperature)
        valid_probs = valid_probs / np.sum(valid_probs)
        
        # 根据概率分布选择落子位置
        chosen_idx = np.random.choice(valid_move_indices, p=valid_probs)
        return valid_moves[chosen_idx]
    
    def play_game(self, human_first=True, verbose=True):
        """开始游戏"""
        self.game.reset()
        move_count = 0
        
        print("\n===== 五子棋游戏开始 =====")
        print("提示：输入'q'可以随时退出游戏")
        
        # 打印初始棋盘
        if verbose:
            self.game.print_board()
            print("你执黑棋(●)，AI执白棋(○)")
        
        while not self.game.game_over:
            move_count += 1
            
            if (human_first and self.game.current_player == 1) or (not human_first and self.game.current_player == 2):
                # 人类玩家回合
                print(f"\n第{move_count}回合，你的回合：")
                move = self.get_human_move()
                
                if move is None:
                    print("游戏已退出。")
                    return None
            else:
                # AI回合
                print(f"\n第{move_count}回合，AI思考中...")
                move = self.get_ai_move()
                print(f"AI在位置{move}落子")
            
            # 落子
            self.game.make_move(*move)
            
            # 打印当前棋盘
            if verbose:
                self.game.print_board()
            
            # 检查游戏是否结束
            if self.game.game_over:
                print("\n===== 游戏结束 =====")
                if self.game.winner == 1:
                    print("恭喜你获胜！")
                elif self.game.winner == 2:
                    print("AI获胜！再接再厉！")
                else:
                    print("平局！")
                print(f"共进行了{move_count}回合")
                break
    
    def start_game_loop(self):
        """游戏主循环"""
        while True:
            # 询问玩家是否先手
            while True:
                first_choice = input("\n你想先手执黑棋吗？(y/n): ").strip().lower()
                if first_choice in ['y', 'yes', 'n', 'no']:
                    human_first = first_choice in ['y', 'yes']
                    break
                print("输入无效，请输入'y'或'n'。")
            
            # 询问AI强度
            while True:
                try:
                    strength_input = input("请设置AI强度 (1-10，数字越大越强): ").strip()
                    strength_level = int(strength_input)
                    if 1 <= strength_level <= 10:
                        self.ai_strength = strength_level / 10.0
                        break
                    print("输入无效，请输入1到10之间的整数。")
                except ValueError:
                    print("输入无效，请输入一个整数。")
            
            # 开始游戏
            self.play_game(human_first=human_first)
            
            # 询问是否再玩一局
            while True:
                play_again = input("\n再玩一局？(y/n): ").strip().lower()
                if play_again in ['y', 'yes', 'n', 'no']:
                    if play_again in ['n', 'no']:
                        print("谢谢游玩！再见！")
                        return
                    break
                print("输入无效，请输入'y'或'n'。")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='五子棋游戏 - 与AI对战')
    parser.add_argument('--board_size', type=int, default=15, help='棋盘大小')
    parser.add_argument('--model_path', type=str, default=None, help='AI模型路径')
    parser.add_argument('--ai_strength', type=float, default=0.8, help='AI强度 (0.0-1.0)')
    
    args = parser.parse_args()
    
    # 创建游戏实例
    player = GobangPlayer(
        board_size=args.board_size,
        model_path=args.model_path,
        ai_strength=args.ai_strength
    )
    
    # 打印欢迎信息
    print("\n" + "=" * 50)
    print("欢迎来到五子棋游戏")
    print("=" * 50)
    
    if args.model_path:
        print(f"已加载AI模型: {args.model_path}")
    else:
        print("未加载AI模型，将使用随机策略")
        print("提示：可以使用 --model_path 参数加载训练好的模型")
    
    # 开始游戏循环
    player.start_game_loop()

if __name__ == "__main__":
    main()