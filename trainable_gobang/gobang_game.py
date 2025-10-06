import numpy as np
import random

class GobangGame:
    def __init__(self, board_size=15):
        """初始化五子棋游戏"""
        self.board_size = board_size  # 棋盘大小，默认为15x15
        self.size = board_size  # 保持向后兼容
        self.reset()
        
    def reset(self):
        """重置游戏状态"""
        # 创建空棋盘，0表示空位，1表示黑棋，2表示白棋
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        # 黑棋先行
        self.current_player = 1
        # 记录游戏是否结束
        self.game_over = False
        # 记录获胜方，0表示未结束或平局
        self.winner = 0
        # 记录落子历史
        self.history = []
        self.move_history = []  # 保持与game_ui.py的兼容性
        
    def get_valid_moves(self):
        """获取所有合法的落子位置"""
        if self.game_over:
            return []
        # 返回所有空位坐标
        return list(zip(*np.where(self.board == 0)))
    
    def place_stone(self, x, y, player):
        """在指定位置落子"""
        if self.board[x][y] != 0:
            raise ValueError(f"位置({x}, {y})已有棋子")
        
        self.board[x][y] = player
        self.history.append((x, y, player))
        self.move_history.append((x, y, player))
        
    def make_move(self, row, col):
        """在指定位置落子"""
        # 检查落子是否合法
        if (row < 0 or row >= self.board_size or 
            col < 0 or col >= self.board_size or 
            self.board[row, col] != 0 or 
            self.game_over):
            return False
        
        # 记录落子
        self.board[row, col] = self.current_player
        self.history.append((row, col, self.current_player))
        self.move_history.append((row, col, self.current_player))
        
        # 检查是否获胜
        if self._check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
            return True
        
        # 检查是否平局（棋盘已满）
        if len(self.get_valid_moves()) == 0:
            self.game_over = True
            self.winner = 0  # 平局
            return True
        
        # 切换玩家
        self.current_player = 2 if self.current_player == 1 else 1
        return True
        
    def undo_move(self):
        """悔棋，撤销最后一步"""
        if not self.history:
            raise ValueError("没有可撤销的步骤")
        
        x, y, player = self.history.pop()
        if self.move_history:
            self.move_history.pop()
        self.board[x][y] = 0
        
        # 切换回上一个玩家
        self.current_player = player
        self.game_over = False
        self.winner = 0
        
    def _check_win(self, row, col):
        """检查指定位置落子后是否获胜"""
        player = self.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 水平、垂直、两个对角线方向
        
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
        
    def check_win(self, x, y, player):
        """检查指定位置和玩家是否获胜"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 水平、垂直、右下对角线、左下对角线
        
        for dx, dy in directions:
            count = 1  # 当前位置的棋子
            
            # 向一个方向检查
            for i in range(1, 5):
                nx, ny = x + dx * i, y + dy * i
                if (0 <= nx < self.board_size and 0 <= ny < self.board_size and 
                    self.board[nx][ny] == player):
                    count += 1
                else:
                    break
            
            # 向相反方向检查
            for i in range(1, 5):
                nx, ny = x - dx * i, y - dy * i
                if (0 <= nx < self.board_size and 0 <= ny < self.board_size and 
                    self.board[nx][ny] == player):
                    count += 1
                else:
                    break
            
            # 如果有五个或更多连续的棋子，返回True表示获胜
            if count >= 5:
                return True
        
        return False
        
    def is_board_full(self):
        """检查棋盘是否已满"""
        return not np.any(self.board == 0)
        
    def get_random_move(self):
        """随机选择一个有效的落子位置"""
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return None
        return random.choice(valid_moves)
        
    def get_state(self):
        """获取当前游戏状态"""
        return {
            'board': self.board.copy(),
            'current_player': self.current_player,
            'game_over': self.game_over,
            'winner': self.winner,
            'history': self.history.copy()
        }
        
    def display_board(self):
        """显示棋盘状态"""
        print("  " + " ".join(str(i) for i in range(self.size)))
        for i in range(self.size):
            row_str = f"{i} "
            for j in range(self.size):
                if self.board[i, j] == 0:
                    row_str += ". "
                elif self.board[i, j] == 1:
                    row_str += "X "
                else:
                    row_str += "O "
            print(row_str)
        
    def get_board_feature(self, current_player=None):
        """将棋盘转换为神经网络输入格式"""
        # 如果没有指定当前玩家，使用游戏的当前玩家
        if current_player is None:
            current_player = self.current_player
            
        # 创建3个通道：自己的棋子、对手的棋子、当前玩家标识
        feature = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        
        # 第一个通道：自己的棋子
        feature[0] = (self.board == current_player).astype(np.float32)
        
        # 第二个通道：对手的棋子
        opponent_player = 2 if current_player == 1 else 1
        feature[1] = (self.board == opponent_player).astype(np.float32)
        
        # 第三个通道：当前玩家标识（全1或全0）
        if current_player == 1:
            feature[2] = np.ones((self.board_size, self.board_size), dtype=np.float32)
        else:
            feature[2] = np.zeros((self.board_size, self.board_size), dtype=np.float32)
            
        # 转置为 (board_size, board_size, 3) 格式以兼容model.py
        return feature.transpose(1, 2, 0)
        
    def copy(self):
        """创建游戏状态的深拷贝"""
        new_game = GobangGame(self.board_size)
        new_game.board = np.copy(self.board)
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.history = self.history.copy()
        new_game.move_history = self.move_history.copy()
        return new_game
        
    def get_winner(self):
        """获取当前游戏的获胜者，如果没有获胜者则返回0"""
        # 检查所有已落子的位置
        for x, y, player in self.history:
            if self.check_win(x, y, player):
                return player
        
        # 检查是否平局
        if self.is_board_full():
            return -1  # 表示平局
        
        return 0  # 游戏继续
        
    def to_flat_move(self, x, y):
        """将二维坐标转换为一维索引"""
        return x * self.board_size + y
        
    def from_flat_move(self, flat_move):
        """将一维索引转换为二维坐标"""
        x = flat_move // self.board_size
        y = flat_move % self.board_size
        return (x, y)

# 测试代码
if __name__ == "__main__":
    # 创建游戏实例
    game = GobangGame(15)
    game.display_board()
    
    # 模拟几个落子
    game.place_stone(7, 7, 1)  # 黑棋
    game.place_stone(7, 8, 2)  # 白棋
    game.place_stone(6, 7, 1)  # 黑棋
    game.place_stone(6, 8, 2)  # 白棋
    
    # 显示棋盘
    print("初始棋盘:")
    game.display_board()
    
    # 检查胜负
    winner = game.get_winner()
    print(f"当前获胜者: {winner}")
    
    # 测试棋盘特征生成
    feature = game.get_board_feature(1)
    print(f"棋盘特征形状: {feature.shape}")
    
    # 测试悔棋
    game.undo_move()
    print("悔棋后:")
    game.display_board()
    
    # 测试有效落子
    valid_moves = game.get_valid_moves()
    print(f"有效落子数量: {len(valid_moves)}")
    
    # 测试随机落子
    random_move = game.get_random_move()
    print(f"随机落子: {random_move}")
    
    # 简单的测试对局
    moves = [(8, 8), (8, 9), (6, 6), (6, 9), (9, 9), (5, 5)]
    for i, (r, c) in enumerate(moves):
        print(f"玩家{game.current_player}在位置({r}, {c})落子")
        game.make_move(r, c)
        game.display_board()
        if game.game_over:
            if game.winner == 0:
                print("游戏结束，平局！")
            else:
                print(f"游戏结束，玩家{game.winner}获胜！")
            break