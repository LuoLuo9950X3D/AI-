import numpy as np
import tensorflow as tf
from math import *
import time
import random

# 棋盘参数
BOARD_SIZE = 15
GRID_WIDTH = 40

# 搜索参数
DEFAULT_DEPTH = 4  # 默认搜索深度
MAX_DEPTH = 6      # 最大搜索深度
MIN_DEPTH = 3      # 最小搜索深度

# 进攻系数（大于1进攻型，小于1防守型）
ATTACK_RATIO = 1.2

# 棋型评分表（价值越大表示棋型越好）
SHAPE_SCORES = [
    (50, (0, 1, 1, 0, 0)),           # 冲四-半开放
    (50, (0, 0, 1, 1, 0)),           # 冲四-半开放
    (200, (1, 1, 0, 1, 0)),          # 冲四-连接型
    (500, (0, 0, 1, 1, 1)),          # 活三
    (500, (1, 1, 1, 0, 0)),          # 活三
    (5000, (0, 1, 1, 1, 0)),         # 冲四-活三进阶
    (5000, (0, 1, 0, 1, 1, 0)),      # 冲四-活三连接
    (5000, (0, 1, 1, 0, 1, 0)),      # 冲四-活三连接
    (5000, (1, 1, 1, 0, 1)),         # 冲四-活三延伸
    (5000, (1, 1, 0, 1, 1)),         # 冲四-活三延伸
    (5000, (1, 0, 1, 1, 1)),         # 冲四-活三延伸
    (5000, (1, 1, 1, 1, 0)),         # 冲四-即将成五
    (5000, (0, 1, 1, 1, 1)),         # 冲四-即将成五
    (50000, (0, 1, 1, 1, 1, 0)),     # 活四
    (99999999, (1, 1, 1, 1, 1))      # 五连子（胜利）
]

class DeepSeekGobangAI:
    def __init__(self, board_size=BOARD_SIZE, use_mla=True, use_moe=True):
        """初始化DeepSeek风格的五子棋AI"""
        self.board_size = board_size
        self.reset_game()
        
        # DeepSeek架构特性
        self.use_mla = use_mla  # 使用多头潜在注意力机制
        self.use_moe = use_moe  # 使用混合专家架构
        
        # 统计信息
        self.search_count = 0
        self.cut_count = 0
        self.time_spent = 0
        
        # 专家模块 - 不同的评估策略
        self.experts = {
            "pattern_based": self._expert_pattern_based,
            "mobility": self._expert_mobility,
            "influence": self._expert_influence,
            "history": self._expert_history
        }
        
        # 历史数据记录 - 用于历史启发式搜索
        self.history_table = {}
        self.history_weight = 1.0
        
    def reset_game(self):
        """重置游戏状态"""
        self.ai_moves = []        # AI落子记录
        self.human_moves = []     # 人类落子记录
        self.all_moves = []       # 所有落子记录
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)  # 0:空，1:AI，2:人类
        self.next_point = [0, 0]  # AI下一步最应该下的位置
        
    def ai_move(self, depth=None, time_limit=3.0):
        """AI选择下一步的位置"""
        self.search_count = 0
        self.cut_count = 0
        start_time = time.time()
        
        # 根据局面复杂度动态调整搜索深度
        if depth is None:
            move_count = len(self.all_moves)
            # 初期搜索浅一些，中盘搜索深一些，尾盘根据情况调整
            if move_count < 10:
                depth = DEFAULT_DEPTH - 1
            elif move_count < 20:
                depth = DEFAULT_DEPTH
            elif move_count < 30:
                depth = min(DEFAULT_DEPTH + 1, MAX_DEPTH)
            else:
                depth = min(DEFAULT_DEPTH + 1, MAX_DEPTH)
        
        # 使用负极大值算法搜索最佳位置
        self._negamax(True, depth, -99999999, 99999999, time_limit, start_time)
        
        self.time_spent = time.time() - start_time
        print(f"搜索深度: {depth}, 搜索次数: {self.search_count}, 剪枝次数: {self.cut_count}, 用时: {self.time_spent:.3f}秒")
        
        # 记录AI落子
        ai_x, ai_y = self.next_point
        self.ai_moves.append((ai_x, ai_y))
        self.all_moves.append((ai_x, ai_y))
        self.board[ai_x, ai_y] = 1
        
        return ai_x, ai_y
    
    def human_move(self, x, y):
        """记录人类玩家的落子"""
        if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x, y] == 0:
            self.human_moves.append((x, y))
            self.all_moves.append((x, y))
            self.board[x, y] = 2
            return True
        return False
    
    def _negamax(self, is_ai, depth, alpha, beta, time_limit, start_time, expert_weights=None):
        """负极大值算法搜索 + alpha-beta剪枝 + DeepSeek架构优化"""
        # 检查游戏是否结束、搜索深度是否达到边界或超时
        current_time = time.time()
        if current_time - start_time > time_limit:
            return self._evaluate_position(is_ai, expert_weights)
            
        if self._check_win(1 if is_ai else 2) or depth == 0:
            return self._evaluate_position(is_ai, expert_weights)
        
        # 生成候选移动列表并排序
        candidates = self._generate_candidates()
        
        # 遍历每个候选步
        for move in candidates:
            self.search_count += 1
            
            x, y = move
            # 如果没有相邻的子，跳过评估以减少计算量
            if not self._has_neighbor(x, y):
                continue
            
            # 落子
            player = 1 if is_ai else 2
            self.board[x, y] = player
            self.all_moves.append((x, y))
            
            # 根据当前局面动态调整专家权重
            if self.use_moe and expert_weights is None:
                dynamic_weights = self._adjust_expert_weights()
            else:
                dynamic_weights = expert_weights
            
            # 递归搜索
            value = -self._negamax(not is_ai, depth - 1, -beta, -alpha, time_limit, start_time, dynamic_weights)
            
            # 撤销落子
            self.board[x, y] = 0
            self.all_moves.pop()
            
            # alpha-beta剪枝
            if value > alpha:
                if depth == DEFAULT_DEPTH or depth == MAX_DEPTH:
                    self.next_point = [x, y]
                    # 更新历史表
                    if self.use_mla:
                        self._update_history_table(x, y, value)
                
                if value >= beta:
                    self.cut_count += 1
                    return beta
                
                alpha = value
        
        return alpha
    
    def _generate_candidates(self):
        """生成候选移动列表并排序（优先考虑有价值的位置）"""
        candidates = []
        
        # 检查所有空位
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    # 只有有邻居的位置才被考虑
                    if self._has_neighbor(i, j):
                        candidates.append((i, j))
        
        # 使用历史启发式对候选位置进行排序
        if self.use_mla and len(self.history_table) > 0:
            candidates.sort(key=lambda pos: self.history_table.get(pos, 0), reverse=True)
        
        # 没有历史数据时，优先考虑最近落子的周围
        elif len(self.all_moves) > 0:
            last_x, last_y = self.all_moves[-1]
            ordered = []
            others = []
            
            # 检查最近落子周围的位置
            for pos in candidates:
                x, y = pos
                if abs(x - last_x) <= 2 and abs(y - last_y) <= 2:
                    ordered.append(pos)
                else:
                    others.append(pos)
            
            # 先搜索最近落子周围的位置
            candidates = ordered + others
        
        return candidates
    
    def _has_neighbor(self, x, y):
        """检查指定位置是否有相邻的棋子"""
        for i in range(-2, 3):
            for j in range(-2, 3):
                if i == 0 and j == 0:
                    continue
                nx, ny = x + i, y + j
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if self.board[nx, ny] != 0:
                        return True
        return False
    
    def _evaluate_position(self, is_ai, expert_weights=None):
        """评估当前局面价值（基于混合专家架构）"""
        # 默认专家权重
        if expert_weights is None:
            expert_weights = {
                "pattern_based": 0.5,
                "mobility": 0.2,
                "influence": 0.2,
                "history": 0.1
            }
        
        # 汇总各专家的评估结果
        total_score = 0
        for expert_name, weight in expert_weights.items():
            if self.use_moe and expert_name in self.experts:
                expert_score = self.experts[expert_name](is_ai)
                total_score += expert_score * weight
        
        # 如果没有使用混合专家架构，只使用基于棋型的评估
        if not self.use_moe:
            total_score = self._expert_pattern_based(is_ai)
        
        return total_score
    
    def _adjust_expert_weights(self):
        """根据当前局面动态调整专家权重"""
        move_count = len(self.all_moves)
        weights = {
            "pattern_based": 0.5,
            "mobility": 0.2,
            "influence": 0.2,
            "history": 0.1
        }
        
        # 开局阶段更重视机动性
        if move_count < 10:
            weights["mobility"] += 0.1
            weights["pattern_based"] -= 0.1
        # 中盘阶段更重视棋型和影响力
        elif move_count < 30:
            weights["pattern_based"] += 0.1
            weights["influence"] += 0.1
            weights["mobility"] -= 0.2
        # 尾盘阶段更重视棋型
        else:
            weights["pattern_based"] += 0.2
            weights["mobility"] -= 0.1
            weights["history"] -= 0.1
        
        # 检查是否有即将形成的五连子，增加棋型专家权重
        if self._check_imminent_win():
            weights["pattern_based"] += 0.2
            
        return weights
    
    def _check_imminent_win(self):
        """检查是否有即将形成的五连子"""
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] != 0:
                    # 检查四个方向
                    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
                    for dx, dy in directions:
                        # 检查是否有连续四个相同的棋子
                        count = 1
                        for step in range(1, 4):
                            ni, nj = i + step*dx, j + step*dy
                            if 0 <= ni < self.board_size and 0 <= nj < self.board_size:
                                if self.board[ni, nj] == self.board[i, j]:
                                    count += 1
                                else:
                                    break
                            else:
                                break
                        
                        # 如果有连续四个相同的棋子，说明即将形成五连子
                        if count >= 4:
                            return True
        return False
    
    def _update_history_table(self, x, y, value):
        """更新历史表（多头潜在注意力机制的体现）"""
        # 增强当前位置的权重
        current_value = self.history_table.get((x, y), 0)
        self.history_table[(x, y)] = current_value + value * self.history_weight
        
        # 衰减历史记录的值
        for pos in list(self.history_table.keys()):
            if pos != (x, y):
                self.history_table[pos] *= 0.95
                # 移除过小的值以节省内存
                if self.history_table[pos] < 0.1:
                    del self.history_table[pos]
    
    def _expert_pattern_based(self, is_ai):
        """基于棋型的专家评估"""
        my_color = 1 if is_ai else 2
        enemy_color = 2 if is_ai else 1
        
        # 计算我方得分
        my_score = self._calculate_pattern_score(my_color)
        # 计算敌方得分并根据进攻系数调整
        enemy_score = self._calculate_pattern_score(enemy_color) * (1/ATTACK_RATIO)
        
        return my_score - enemy_score
    
    def _expert_mobility(self, is_ai):
        """基于机动性的专家评估"""
        # 机动性评分：可用的好点数量
        my_color = 1 if is_ai else 2
        enemy_color = 2 if is_ai else 1
        
        my_mobility = 0
        enemy_mobility = 0
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    # 评估该点对双方的价值
                    my_value = self._evaluate_point(i, j, my_color)
                    enemy_value = self._evaluate_point(i, j, enemy_color)
                    
                    if my_value > 100:
                        my_mobility += 1
                    if enemy_value > 100:
                        enemy_mobility += 1
        
        return my_mobility - enemy_mobility * 0.8
    
    def _expert_influence(self, is_ai):
        """基于影响力的专家评估"""
        # 影响力评分：棋子对周围的控制范围
        my_color = 1 if is_ai else 2
        my_influence = 0
        
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x, y] == my_color:
                    # 计算该棋子的影响力范围
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                                # 距离越近，影响力越大
                                distance = max(1, abs(dx) + abs(dy))
                                my_influence += 1.0 / distance
        
        return my_influence * 10
    
    def _expert_history(self, is_ai):
        """基于历史数据的专家评估"""
        # 历史评分：基于之前的搜索结果
        history_score = 0
        
        # 检查当前所有已落子的历史价值
        for x, y in self.all_moves:
            if self.board[x, y] == (1 if is_ai else 2):
                history_score += self.history_table.get((x, y), 0) * 0.1
        
        return history_score
    
    def _calculate_pattern_score(self, color):
        """计算指定颜色的棋型得分"""
        score = 0
        score_positions = []  # 已计算得分的位置，避免重复计算
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == color:
                    # 检查四个方向
                    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
                    for dx, dy in directions:
                        # 如果该位置和方向已经计算过得分，则跳过
                        if (i, j, dx, dy) in score_positions:
                            continue
                        
                        # 计算该方向上的得分
                        line_score = self._calculate_line_score(i, j, dx, dy, color)
                        score += line_score
                        
                        # 记录已计算得分的位置和方向
                        for step in range(5):
                            ni, nj = i + step*dx, j + step*dy
                            if 0 <= ni < self.board_size and 0 <= nj < self.board_size:
                                score_positions.append((ni, nj, dx, dy))
        
        return score
    
    def _calculate_line_score(self, x, y, dx, dy, color):
        """计算指定方向上的棋型得分"""
        # 收集该方向上的棋子分布
        line = []
        for step in range(-5, 6):
            nx, ny = x + step*dx, y + step*dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                line.append(self.board[nx, ny])
            else:
                line.append(-1)  # 边界
        
        # 检查所有可能的棋型
        max_score = 0
        for i in range(len(line) - 5):
            window = line[i:i+5]
            for score, pattern in SHAPE_SCORES:
                if self._match_pattern(window, pattern, color):
                    max_score = max(max_score, score)
        
        return max_score
    
    def _match_pattern(self, window, pattern, color):
        """匹配棋型模式"""
        for w, p in zip(window, pattern):
            if p == 0 and w != 0 and w != -1:  # -1表示边界
                return False
            if p == 1 and w != color:
                return False
        return True
    
    def _evaluate_point(self, x, y, color):
        """评估单个点的价值"""
        # 临时落子以评估价值
        self.board[x, y] = color
        score = self._calculate_pattern_score(color)
        self.board[x, y] = 0  # 撤销落子
        return score
    
    def _check_win(self, color):
        """检查指定颜色是否获胜"""
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == color:
                    # 检查水平方向
                    if j + 4 < self.board_size:
                        win = True
                        for k in range(5):
                            if self.board[i, j+k] != color:
                                win = False
                                break
                        if win:
                            return True
                    
                    # 检查垂直方向
                    if i + 4 < self.board_size:
                        win = True
                        for k in range(5):
                            if self.board[i+k, j] != color:
                                win = False
                                break
                        if win:
                            return True
                    
                    # 检查对角线方向
                    if i + 4 < self.board_size and j + 4 < self.board_size:
                        win = True
                        for k in range(5):
                            if self.board[i+k, j+k] != color:
                                win = False
                                break
                        if win:
                            return True
                    
                    # 检查反对角线方向
                    if i + 4 < self.board_size and j - 4 >= 0:
                        win = True
                        for k in range(5):
                            if self.board[i+k, j-k] != color:
                                win = False
                                break
                        if win:
                            return True
        return False

# 演示如何使用这个优化的AI
if __name__ == "__main__":
    # 创建AI实例
    ai = DeepSeekGobangAI(use_mla=True, use_moe=True)
    
    # 简单的游戏循环示例
    print("DeepSeek风格五子棋AI演示")
    print("输入格式: x y (0-14，空格分隔)")
    
    game_over = False
    current_player = 2  # 1:AI, 2:人类
    
    while not game_over:
        if current_player == 1:
            # AI回合
            print("AI思考中...")
            ai_x, ai_y = ai.ai_move()
            print(f"AI落子在: {ai_x}, {ai_y}")
            
            # 检查AI是否获胜
            if ai._check_win(1):
                print("AI获胜！")
                game_over = True
            
            current_player = 2
        else:
            # 人类回合
            try:
                user_input = input("请输入你的落子位置 (x y): ").strip()
                # 支持退出命令
                if user_input.lower() in ['q', 'quit', 'exit']:
                    print("游戏已退出。")
                    game_over = True
                    continue
                
                x, y = map(int, user_input.split())
                
                if ai.human_move(x, y):
                    print(f"你落子在: {x}, {y}")
                    
                    # 检查人类是否获胜
                    if ai._check_win(2):
                        print("恭喜你获胜！")
                        game_over = True
                    
                    current_player = 1
                else:
                    print("无效的落子位置，请重新输入。")
            except ValueError:
                print("输入格式错误，请输入两个整数，用空格分隔。例如：7 7")