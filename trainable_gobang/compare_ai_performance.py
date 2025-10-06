import numpy as np
import time
import random
from optimized_gobang_ai import DeepSeekGobangAI

class OriginalGobangAI:
    """原始五子棋AI的简化版本，用于对比测试"""
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.reset_game()
        self.search_count = 0
        self.cut_count = 0
        
    def reset_game(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)  # 0:空，1:AI，2:人类
        self.all_moves = []
        self.next_point = [0, 0]
        
    def ai_move(self, depth=4):
        self.search_count = 0
        self.cut_count = 0
        start_time = time.time()
        
        # 简化版的negamax算法
        self._negamax_simplified(True, depth, -99999999, 99999999)
        
        self.time_spent = time.time() - start_time
        
        ai_x, ai_y = self.next_point
        self.board[ai_x, ai_y] = 1
        self.all_moves.append((ai_x, ai_y))
        
        return ai_x, ai_y
        
    def _negamax_simplified(self, is_ai, depth, alpha, beta):
        """简化版的negamax算法"""
        self.search_count += 1
        
        # 检查胜利条件
        if self._check_win(1 if is_ai else 2):
            return 99999999 if is_ai else -99999999
            
        if depth == 0:
            return self._evaluate_simplified(is_ai)
        
        # 生成候选移动
        candidates = self._generate_candidates_simplified()
        
        for move in candidates:
            x, y = move
            
            # 落子
            player = 1 if is_ai else 2
            self.board[x, y] = player
            self.all_moves.append((x, y))
            
            # 递归搜索
            value = -self._negamax_simplified(not is_ai, depth - 1, -beta, -alpha)
            
            # 撤销落子
            self.board[x, y] = 0
            self.all_moves.pop()
            
            # 剪枝
            if value > alpha:
                if depth == 4:
                    self.next_point = [x, y]
                
                if value >= beta:
                    self.cut_count += 1
                    return beta
                
                alpha = value
        
        return alpha
        
    def _generate_candidates_simplified(self):
        """简化版的候选移动生成"""
        candidates = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    if self._has_neighbor(i, j):
                        candidates.append((i, j))
        
        # 简单排序：优先考虑中心位置
        candidates.sort(key=lambda pos: abs(7 - pos[0]) + abs(7 - pos[1]))
        
        return candidates
        
    def _has_neighbor(self, x, y):
        """检查是否有相邻的棋子"""
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                nx, ny = x + i, y + j
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if self.board[nx, ny] != 0:
                        return True
        return False
        
    def _evaluate_simplified(self, is_ai):
        """简化版的评估函数"""
        my_color = 1 if is_ai else 2
        enemy_color = 2 if is_ai else 1
        
        my_score = self._calculate_score_simplified(my_color)
        enemy_score = self._calculate_score_simplified(enemy_color)
        
        return my_score - enemy_score
        
    def _calculate_score_simplified(self, color):
        """简化版的得分计算"""
        score = 0
        
        # 简单的棋型识别
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == color:
                    # 检查水平、垂直、对角线方向
                    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
                    for dx, dy in directions:
                        # 计算连续相同颜色的棋子数
                        count = 1
                        for step in range(1, 5):
                            nx, ny = i + step*dx, j + step*dy
                            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                                if self.board[nx, ny] == color:
                                    count += 1
                                else:
                                    break
                            else:
                                break
                        
                        # 简单评分
                        if count == 5:
                            score += 99999999
                        elif count == 4:
                            score += 5000
                        elif count == 3:
                            score += 500
                        elif count == 2:
                            score += 50
                        elif count == 1:
                            score += 5
        
        return score
        
    def _check_win(self, color):
        """检查是否获胜"""
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

class AIPerformanceTester:
    """AI性能测试器"""
    def __init__(self):
        self.board_size = 15
        
    def compare_ai_performance(self, num_games=5):
        """比较两个AI的性能"""
        print("开始比较DeepSeek优化版AI与原始AI的性能...")
        print(f"将进行{num_games}局比赛\n")
        
        optimized_wins = 0
        original_wins = 0
        draws = 0
        
        total_optimized_search_count = 0
        total_original_search_count = 0
        total_optimized_time = 0
        total_original_time = 0
        
        for game_idx in range(1, num_games + 1):
            print(f"=== 游戏 {game_idx}/{num_games} ===")
            
            # 交替先手
            optimized_first = (game_idx % 2 == 1)
            print(f"DeepSeek优化版AI {'先手' if optimized_first else '后手'}")
            
            # 创建AI实例
            optimized_ai = DeepSeekGobangAI()
            original_ai = OriginalGobangAI()
            
            # 记录单局统计信息
            game_optimized_search_count = 0
            game_original_search_count = 0
            game_optimized_time = 0
            game_original_time = 0
            
            # 游戏主循环
            game_over = False
            current_player = 1 if optimized_first else 2  # 1:优化版，2:原始版
            move_count = 0
            
            # 复制棋盘状态的函数
            def copy_board(source, target):
                target.board = np.copy(source.board)
                target.all_moves = source.all_moves.copy()
            
            while not game_over and move_count < self.board_size * self.board_size:
                if current_player == 1:
                    # 优化版AI回合
                    start_time = time.time()
                    ai_x, ai_y = optimized_ai.ai_move()
                    move_time = time.time() - start_time
                    
                    game_optimized_search_count += optimized_ai.search_count
                    game_optimized_time += move_time
                    
                    print(f"优化版AI: 落子({ai_x},{ai_y}), 搜索次数: {optimized_ai.search_count}, 用时: {move_time:.3f}秒")
                    
                    # 同步棋盘状态
                    copy_board(optimized_ai, original_ai)
                    
                    # 检查优化版AI是否获胜
                    if optimized_ai._check_win(1):
                        print("DeepSeek优化版AI获胜！")
                        optimized_wins += 1
                        game_over = True
                    
                    current_player = 2
                else:
                    # 原始版AI回合
                    start_time = time.time()
                    ai_x, ai_y = original_ai.ai_move()
                    move_time = time.time() - start_time
                    
                    game_original_search_count += original_ai.search_count
                    game_original_time += move_time
                    
                    print(f"原始版AI: 落子({ai_x},{ai_y}), 搜索次数: {original_ai.search_count}, 用时: {move_time:.3f}秒")
                    
                    # 同步棋盘状态
                    copy_board(original_ai, optimized_ai)
                    
                    # 检查原始版AI是否获胜
                    if original_ai._check_win(1):
                        print("原始版AI获胜！")
                        original_wins += 1
                        game_over = True
                    
                    current_player = 1
                
                move_count += 1
            
            if not game_over:
                print("游戏平局！")
                draws += 1
            
            # 更新总统计信息
            total_optimized_search_count += game_optimized_search_count
            total_original_search_count += game_original_search_count
            total_optimized_time += game_optimized_time
            total_original_time += game_original_time
            
            print(f"本局优化版搜索总数: {game_optimized_search_count}, 总用时: {game_optimized_time:.3f}秒")
            print(f"本局原始版搜索总数: {game_original_search_count}, 总用时: {game_original_time:.3f}秒")
            print()
        
        # 输出最终统计结果
        print("=== 性能比较结果汇总 ===")
        print(f"DeepSeek优化版AI胜场: {optimized_wins}")
        print(f"原始版AI胜场: {original_wins}")
        print(f"平局: {draws}")
        
        avg_optimized_search = total_optimized_search_count / num_games if num_games > 0 else 0
        avg_original_search = total_original_search_count / num_games if num_games > 0 else 0
        avg_optimized_time = total_optimized_time / num_games if num_games > 0 else 0
        avg_original_time = total_original_time / num_games if num_games > 0 else 0
        
        print(f"平均每局优化版搜索次数: {avg_optimized_search:.0f}")
        print(f"平均每局原始版搜索次数: {avg_original_search:.0f}")
        print(f"平均每局优化版用时: {avg_optimized_time:.3f}秒")
        print(f"平均每局原始版用时: {avg_original_time:.3f}秒")
        
        search_reduction = ((avg_original_search - avg_optimized_search) / avg_original_search * 100) if avg_original_search > 0 else 0
        time_reduction = ((avg_original_time - avg_optimized_time) / avg_original_time * 100) if avg_original_time > 0 else 0
        
        print(f"搜索次数减少: {search_reduction:.2f}%")
        print(f"用时减少: {time_reduction:.2f}%")
        
        # 计算胜率
        total_games_without_draws = optimized_wins + original_wins
        if total_games_without_draws > 0:
            optimized_win_rate = (optimized_wins / total_games_without_draws) * 100
            print(f"DeepSeek优化版AI胜率: {optimized_win_rate:.2f}%")

    def benchmark_single_ai(self, ai_type="optimized", num_tests=3, depth=4):
        """基准测试单个AI的性能"""
        print(f"开始基准测试{ai_type}版AI...")
        
        total_search_count = 0
        total_time = 0
        total_move_count = 0
        
        for test_idx in range(1, num_tests + 1):
            print(f"=== 测试 {test_idx}/{num_tests} ===")
            
            # 创建AI实例
            if ai_type == "optimized":
                ai = DeepSeekGobangAI()
            else:
                ai = OriginalGobangAI()
            
            # 随机生成一些初始棋子以模拟不同局面
            self._generate_random_position(ai, random.randint(5, 15))
            
            # 运行AI进行决策
            start_time = time.time()
            ai.ai_move(depth=depth)
            move_time = time.time() - start_time
            
            print(f"搜索深度: {depth}, 搜索次数: {ai.search_count}, 剪枝次数: {ai.cut_count}, 用时: {move_time:.3f}秒")
            
            # 更新统计信息
            total_search_count += ai.search_count
            total_time += move_time
            total_move_count += 1
        
        # 输出基准测试结果
        print("\n=== 基准测试结果汇总 ===")
        print(f"平均搜索次数: {total_search_count / total_move_count:.0f}")
        print(f"平均剪枝次数: {'N/A' if ai_type != 'optimized' else total_cut_count / total_move_count:.0f}")
        print(f"平均用时: {total_time / total_move_count:.3f}秒")
        print(f"每秒搜索次数: {total_search_count / total_time:.0f} 次/秒")
        
    def _generate_random_position(self, ai, num_pieces):
        """生成随机的棋盘位置"""
        positions = set()
        colors = [1, 2]  # 1:AI，2:人类
        color_idx = 0
        
        while len(positions) < num_pieces:
            x = random.randint(0, self.board_size - 1)
            y = random.randint(0, self.board_size - 1)
            
            if (x, y) not in positions:
                positions.add((x, y))
                ai.board[x, y] = colors[color_idx % 2]
                ai.all_moves.append((x, y))
                color_idx += 1

    def demonstrate_optimization_features(self):
        """演示优化版AI的关键特性"""
        print("=== 演示DeepSeek风格五子棋AI的关键优化特性 ===")
        print("\n1. 混合专家(MoE)架构")
        print("   - 结合了多种专家评估策略: 棋型评估、机动性评估、影响力评估和历史数据评估")
        print("   - 根据不同的游戏阶段动态调整各专家的权重")
        print("   - 开局阶段更重视机动性，中盘阶段更重视棋型和影响力，尾盘阶段更重视棋型")
        
        print("\n2. 多头潜在注意力机制(MLA)")
        print("   - 使用历史表记录搜索过程中的有价值位置")
        print("   - 根据历史记录对候选移动进行排序，提高剪枝效率")
        print("   - 动态更新和衰减历史记录的值")
        
        print("\n3. 高级剪枝优化")
        print("   - 基于深度和时间的自适应搜索")
        print("   - 只考虑有邻居的位置，减少搜索空间")
        print("   - 优先搜索最近落子周围的位置，提高剪枝效率")
        
        print("\n4. 智能的动态深度调整")
        print("   - 根据棋局的复杂度自动调整搜索深度")
        print("   - 初期搜索浅一些，中盘搜索深一些")
        
        print("\n5. 进攻与防守平衡")
        print("   - 通过进攻系数调整AI的进攻性")
        print("   - 综合考虑我方和敌方的威胁")
        
        print("\n6. 实时调用分析优化")
        print("   - 支持时间限制，防止搜索时间过长")
        print("   - 实时统计搜索次数和剪枝次数")
        print("   - 提供性能指标反馈")

    def play_with_optimized_ai(self):
        """与优化版AI进行对战"""
        print("=== 与DeepSeek优化版五子棋AI对战 ===")
        
        # 创建优化版AI
        ai = DeepSeekGobangAI(use_mla=True, use_moe=True)
        
        # 决定谁先手
        human_first = input("你想先手吗？(y/n): ").lower().strip() == 'y'
        
        game_over = False
        current_player = 2 if human_first else 1  # 1:AI, 2:人类
        
        # 打印棋盘的函数
        def print_board():
            print("   " + " ".join([str(i).rjust(2) for i in range(ai.board_size)]))
            for i in range(ai.board_size):
                row = [str(i).rjust(2)]
                for j in range(ai.board_size):
                    if ai.board[i, j] == 0:
                        row.append(" .")
                    elif ai.board[i, j] == 1:
                        row.append(" O")
                    else:
                        row.append(" X")
                print(" ".join(row))
        
        # 游戏主循环
        while not game_over:
            print_board()
            
            if current_player == 1:
                # AI回合
                print("AI思考中...")
                ai_x, ai_y = ai.ai_move()
                print(f"AI落子在: {ai_x}, {ai_y}")
                print(f"搜索次数: {ai.search_count}, 剪枝次数: {ai.cut_count}, 用时: {ai.time_spent:.3f}秒")
                
                # 检查AI是否获胜
                if ai._check_win(1):
                    print_board()
                    print("AI获胜！")
                    game_over = True
                
                current_player = 2
            else:
                # 人类回合
                try:
                    user_input = input("请输入你的落子位置 (x y)，或输入'q'退出: ").strip()
                    if user_input.lower() in ['q', 'quit', 'exit']:
                        print("游戏已退出。")
                        game_over = True
                        continue
                    
                    x, y = map(int, user_input.split())
                    
                    if ai.human_move(x, y):
                        print(f"你落子在: {x}, {y}")
                        
                        # 检查人类是否获胜
                        if ai._check_win(2):
                            print_board()
                            print("恭喜你获胜！")
                            game_over = True
                        
                        current_player = 1
                    else:
                        print("无效的落子位置，请重新输入。")
                except ValueError:
                    print("输入格式错误，请输入两个整数，用空格分隔。例如：7 7")

# 主函数
if __name__ == "__main__":
    tester = AIPerformanceTester()
    
    while True:
        print("\n=== DeepSeek风格五子棋AI性能测试和演示 ===")
        print("1. 比较优化版AI与原始AI的性能")
        print("2. 基准测试优化版AI")
        print("3. 基准测试原始版AI")
        print("4. 演示优化特性")
        print("5. 与优化版AI对战")
        print("0. 退出")
        
        choice = input("请选择一个选项 (0-5): ").strip()
        
        if choice == "1":
            try:
                num_games = int(input("请输入要进行的游戏局数: ").strip())
                if num_games <= 0:
                    raise ValueError
                tester.compare_ai_performance(num_games=num_games)
            except ValueError:
                print("请输入有效的正整数。")
        elif choice == "2":
            tester.benchmark_single_ai(ai_type="optimized")
        elif choice == "3":
            tester.benchmark_single_ai(ai_type="original")
        elif choice == "4":
            tester.demonstrate_optimization_features()
        elif choice == "5":
            tester.play_with_optimized_ai()
        elif choice == "0":
            print("感谢使用DeepSeek风格五子棋AI性能测试和演示程序！")
            break
        else:
            print("无效的选项，请重新选择。")