import numpy as np
import random

class GobangGame:
    # 常量定义，增加可读性
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    
    def __init__(self, board_size=15):
        """初始化五子棋游戏
        参数:
            board_size: 棋盘大小，默认为15x15
        """
        self.board_size = board_size
        self.reset()
        
    def reset(self):
        """重置游戏状态为初始状态"""
        # 创建空棋盘，0表示空位，1表示黑棋，2表示白棋
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        # 黑棋先行
        self.current_player = self.BLACK
        # 游戏状态标志
        self.game_over = False
        self.winner = self.EMPTY  # 0表示未结束或平局
        # 记录落子历史
        self.history = []
        
    def get_valid_moves(self):
        """获取所有合法的落子位置列表
        返回:
            list: 所有空位的坐标列表，每个坐标是(row, col)元组
        """
        if self.game_over:
            return []
        # 返回所有空位坐标
        return list(zip(*np.where(self.board == self.EMPTY)))
        
    def place_stone(self, row, col, player):
        """在指定位置放置棋子（用于直接控制棋子颜色）
        参数:
            row: 行坐标
            col: 列坐标
            player: 玩家（1或2）
        异常:
            ValueError: 如果位置已有棋子
        """
        if self.board[row][col] != self.EMPTY:
            raise ValueError(f"位置({row}, {col})已有棋子")
        
        self.board[row][col] = player
        self.history.append((row, col, player))
        
    def make_move(self, row, col):
        """玩家在指定位置落子
        参数:
            row: 行坐标
            col: 列坐标
        返回:
            bool: 落子是否成功
        """
        # 检查落子是否合法
        if not self._is_valid_move(row, col):
            return False
        
        # 记录落子
        self.board[row][col] = self.current_player
        self.history.append((row, col, self.current_player))
        
        # 检查是否获胜
        if self._check_win(row, col, self.current_player):
            self.game_over = True
            self.winner = self.current_player
            return True
        
        # 检查是否平局（棋盘已满）
        if not np.any(self.board == self.EMPTY):
            self.game_over = True
            self.winner = self.EMPTY  # 平局
            return True
        
        # 切换玩家
        self.current_player = self.WHITE if self.current_player == self.BLACK else self.BLACK
        return True
        
    def _is_valid_move(self, row, col):
        """检查落子是否合法
        参数:
            row: 行坐标
            col: 列坐标
        返回:
            bool: 落子是否合法
        """
        return (0 <= row < self.board_size and 
                0 <= col < self.board_size and 
                self.board[row, col] == self.EMPTY and 
                not self.game_over)
        
    def undo_move(self):
        """悔棋，撤销最后一步
        异常:
            ValueError: 如果没有可撤销的步骤
        """
        if not self.history:
            raise ValueError("没有可撤销的步骤")
        
        row, col, player = self.history.pop()
        self.board[row][col] = self.EMPTY
        
        # 切换回上一个玩家
        self.current_player = player
        self.game_over = False
        self.winner = self.EMPTY
        
    def _check_win(self, row, col, player):
        """检查指定位置和玩家是否获胜（五子连珠）
        参数:
            row: 行坐标
            col: 列坐标
            player: 玩家（1或2）
        返回:
            bool: 是否获胜
        """
        # 检查四个方向：水平、垂直、两个对角线
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1  # 当前位置已有一个棋子
            
            # 检查正方向
            for i in range(1, 5):
                r, c = row + i * dx, col + i * dy
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break
            
            # 检查反方向
            for i in range(1, 5):
                r, c = row - i * dx, col - i * dy
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break
            
            # 五子连珠获胜
            if count >= 5:
                return True
        
        return False
        
    def is_board_full(self):
        """检查棋盘是否已满
        返回:
            bool: 棋盘是否已满
        """
        return not np.any(self.board == self.EMPTY)
        
    def get_random_move(self):
        """随机选择一个有效的落子位置
        返回:
            tuple: (row, col)坐标，如果没有有效位置则返回None
        """
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return None
        return random.choice(valid_moves)
        
    def get_state(self):
        """获取当前游戏状态快照
        返回:
            dict: 包含棋盘状态、当前玩家、游戏状态等信息
        """
        return {
            'board': self.board.copy(),
            'current_player': self.current_player,
            'game_over': self.game_over,
            'winner': self.winner,
            'history': self.history.copy()
        }
        
    def display_board(self):
        """在控制台显示当前棋盘状态"""
        # 打印列标签
        print("  " + " ".join(f"{i}" for i in range(self.board_size)))
        
        # 打印每一行
        for row in range(self.board_size):
            row_str = f"{row} "
            for col in range(self.board_size):
                if self.board[row, col] == self.EMPTY:
                    row_str += ". "
                elif self.board[row, col] == self.BLACK:
                    row_str += "X "
                else:
                    row_str += "O "
            print(row_str)
        
    def get_board_feature(self, player=None):
        """将棋盘转换为神经网络输入格式
        参数:
            player: 可选，指定玩家视角，默认为当前玩家
        返回:
            numpy.ndarray: 形状为(board_size, board_size, 3)的特征数组
        """
        # 如果没有指定玩家，使用当前玩家
        current_player = player if player is not None else self.current_player
        
        # 创建3个通道：自己的棋子、对手的棋子、当前玩家标识
        feature = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        
        # 第一个通道：当前玩家的棋子
        feature[0] = (self.board == current_player).astype(np.float32)
        
        # 第二个通道：对手的棋子
        opponent_player = self.WHITE if current_player == self.BLACK else self.BLACK
        feature[1] = (self.board == opponent_player).astype(np.float32)
        
        # 第三个通道：当前玩家标识（全1表示黑棋，全0表示白棋）
        feature[2] = np.ones((self.board_size, self.board_size), dtype=np.float32) if current_player == self.BLACK else 0
        
        # 转置为 (board_size, board_size, 3) 格式以兼容model.py
        return feature.transpose(1, 2, 0)
        
    def copy(self):
        """创建游戏状态的深拷贝
        返回:
            GobangGame: 游戏状态的深拷贝
        """
        new_game = GobangGame(self.board_size)
        new_game.board = np.copy(self.board)
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.history = self.history.copy()
        return new_game
        
    def get_winner(self):
        """获取当前游戏的获胜者
        返回:
            int: 获胜者（1或2），-1表示平局，0表示游戏继续
        """
        # 检查所有已落子的位置
        for row, col, player in self.history:
            if self._check_win(row, col, player):
                return player
        
        # 检查是否平局
        if self.is_board_full():
            return -1  # 表示平局
        
        return 0  # 游戏继续
        
    def to_flat_move(self, row, col):
        """将二维坐标转换为一维索引
        参数:
            row: 行坐标
            col: 列坐标
        返回:
            int: 一维索引
        """
        return row * self.board_size + col
        
    def from_flat_move(self, flat_move):
        """将一维索引转换为二维坐标
        参数:
            flat_move: 一维索引
        返回:
            tuple: (row, col)坐标
        """
        row = flat_move // self.board_size
        col = flat_move % self.board_size
        return (row, col)

# 测试代码
if __name__ == "__main__":
    print("=== 五子棋游戏测试 ===")
    
    # 创建游戏实例（使用小棋盘方便测试）
    game = GobangGame(9)
    print("初始空棋盘:")
    game.display_board()
    
    print("\n=== 测试基本功能 ===")
    # 模拟几个落子
    game.place_stone(4, 4, game.BLACK)  # 黑棋
    game.place_stone(4, 5, game.WHITE)  # 白棋
    game.place_stone(3, 4, game.BLACK)  # 黑棋
    game.place_stone(3, 5, game.WHITE)  # 白棋
    
    # 显示棋盘
    print("落子后的棋盘:")
    game.display_board()
    
    # 检查胜负
    winner = game.get_winner()
    print(f"当前获胜者: {winner}")
    
    # 测试棋盘特征生成
    feature = game.get_board_feature(game.BLACK)
    print(f"棋盘特征形状: {feature.shape}")
    
    print("\n=== 测试悔棋功能 ===")
    game.undo_move()
    print("悔棋后:")
    game.display_board()
    
    print("\n=== 测试有效落子 ===")
    valid_moves = game.get_valid_moves()
    print(f"有效落子数量: {len(valid_moves)}")
    
    # 测试随机落子
    random_move = game.get_random_move()
    print(f"随机落子: {random_move}")
    
    print("\n=== 测试完整对局流程 ===")
    # 创建新游戏进行对局测试
    game.reset()
    
    # 模拟一个简单对局（直到游戏结束或达到最大步数）
    max_moves = 20
    move_count = 0
    
    while not game.game_over and move_count < max_moves:
        move = game.get_random_move()
        if move:
            row, col = move
            player = game.current_player
            print(f"玩家{player}在位置({row}, {col})落子")
            game.make_move(row, col)
            move_count += 1
        else:
            break
        
        # 每几步显示一次棋盘
        if move_count % 5 == 0:
            print(f"\n第{move_count}步后棋盘:")
            game.display_board()
    
    # 显示最终结果
    print("\n=== 最终结果 ===")
    game.display_board()
    
    if game.game_over:
        if game.winner == game.EMPTY:
            print("游戏结束，平局！")
        else:
            print(f"游戏结束，玩家{game.winner}获胜！")
    else:
        print(f"测试结束（已进行{move_count}步，未分出胜负）")