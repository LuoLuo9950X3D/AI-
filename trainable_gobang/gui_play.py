import tkinter as tk
import numpy as np
import tensorflow as tf
from tkinter import messagebox, ttk
from gobang_game import GobangGame
from model import GobangModel
import time
import threading

class GobangGUI:
    def __init__(self, root, board_size=15):
        """初始化五子棋GUI界面"""
        self.root = root
        self.root.title("五子棋AI")
        self.root.geometry("800x700")
        self.root.resizable(False, False)
        
        self.board_size = board_size
        self.cell_size = 40  # 每个格子的大小
        self.margin = 30      # 边距
        
        # 初始化游戏和AI模型
        self.game = GobangGame(board_size)
        self.model = None
        self.ai_strength = 1.0  # AI强度
        self.current_player = 1  # 1:黑棋(先手), 2:白棋
        self.is_ai_thinking = False
        self.is_game_started = False
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        """创建GUI组件"""
        # 创建顶部控制栏
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10, fill=tk.X, padx=10)
        
        # 加载模型按钮
        self.load_model_btn = tk.Button(control_frame, text="加载AI模型", command=self.load_model)
        self.load_model_btn.pack(side=tk.LEFT, padx=5)
        
        # AI强度滑块
        strength_frame = tk.Frame(control_frame)
        strength_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(strength_frame, text="AI强度: ").pack(side=tk.LEFT)
        self.strength_scale = tk.Scale(strength_frame, from_=0.1, to=1.0, resolution=0.1,
                                      orient=tk.HORIZONTAL, length=150, command=self.update_ai_strength)
        self.strength_scale.set(self.ai_strength)
        self.strength_scale.pack(side=tk.LEFT)
        
        # 游戏状态标签
        self.status_var = tk.StringVar()
        self.status_var.set("游戏未开始")
        self.status_label = tk.Label(control_frame, textvariable=self.status_var, font=("Arial", 12))
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # 重置按钮
        self.reset_btn = tk.Button(control_frame, text="重新开始", command=self.reset_game)
        self.reset_btn.pack(side=tk.RIGHT, padx=5)
        
        # 创建画布用于绘制棋盘
        canvas_width = self.board_size * self.cell_size + 2 * self.margin
        canvas_height = self.board_size * self.cell_size + 2 * self.margin
        self.canvas = tk.Canvas(self.root, width=canvas_width, height=canvas_height,
                               bg="#DEB887", highlightthickness=0)
        self.canvas.pack(pady=10)
        
        # 绑定鼠标点击事件
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # 绘制空棋盘
        self.draw_board()
        
        # 创建状态栏显示统计信息
        self.stats_var = tk.StringVar()
        self.stats_var.set("落子数: 0 | 思考时间: 0.0s")
        stats_label = tk.Label(self.root, textvariable=self.stats_var, anchor="w")
        stats_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
    
    def draw_board(self):
        """绘制棋盘"""
        self.canvas.delete("all")
        
        # 绘制棋盘线
        for i in range(self.board_size):
            # 横线
            self.canvas.create_line(self.margin, self.margin + i * self.cell_size,
                                   self.margin + (self.board_size - 1) * self.cell_size, self.margin + i * self.cell_size,
                                   width=2)
            # 竖线
            self.canvas.create_line(self.margin + i * self.cell_size, self.margin,
                                   self.margin + i * self.cell_size, self.margin + (self.board_size - 1) * self.cell_size,
                                   width=2)
        
        # 绘制天元和星位
        star_positions = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
        for r, c in star_positions:
            self.canvas.create_oval(self.margin + c * self.cell_size - 5, self.margin + r * self.cell_size - 5,
                                   self.margin + c * self.cell_size + 5, self.margin + r * self.cell_size + 5,
                                   fill="black")
    
    def draw_piece(self, row, col, player):
        """绘制棋子"""
        color = "black" if player == 1 else "white"
        outline_color = "black" if player == 2 else "#333333"
        
        # 绘制棋子
        self.canvas.create_oval(self.margin + col * self.cell_size - self.cell_size // 2 + 2,
                               self.margin + row * self.cell_size - self.cell_size // 2 + 2,
                               self.margin + col * self.cell_size + self.cell_size // 2 - 2,
                               self.margin + row * self.cell_size + self.cell_size // 2 - 2,
                               fill=color, outline=outline_color, width=1)
        
        # 为最后一步添加标记
        if player == self.game.current_player:
            self.canvas.create_text(self.margin + col * self.cell_size,
                                  self.margin + row * self.cell_size,
                                  text="●", fill="red", font=("Arial", 8))
    
    def on_canvas_click(self, event):
        """处理鼠标点击事件"""
        # 检查游戏是否开始或AI是否正在思考
        if not self.is_game_started or self.is_ai_thinking:
            return
        
        # 计算落子位置
        col = round((event.x - self.margin) / self.cell_size)
        row = round((event.y - self.margin) / self.cell_size)
        
        # 检查位置是否有效
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            if (row, col) in self.game.get_valid_moves():
                # 玩家落子
                self.make_move(row, col)
                
                # 检查游戏是否结束
                if not self.game.game_over and self.model is not None:
                    # AI开始思考
                    self.is_ai_thinking = True
                    self.status_var.set("AI正在思考...")
                    self.root.update_idletasks()
                    
                    # 在新线程中运行AI思考逻辑
                    threading.Thread(target=self.ai_move).start()
    
    def make_move(self, row, col):
        """执行落子"""
        # 记录落子前的时间
        move_start_time = time.time()
        
        # 落子
        self.game.make_move(row, col)
        
        # 绘制棋子
        self.draw_piece(row, col, self.game.current_player % 2 + 1)  # 当前玩家是刚落子的玩家
        
        # 更新统计信息
        move_count = np.sum(self.game.board != 0)
        self.stats_var.set(f"落子数: {move_count} | 思考时间: {(time.time() - move_start_time):.2f}s")
        
        # 检查游戏是否结束
        if self.game.game_over:
            self.is_game_started = False
            if self.game.winner == 1:
                messagebox.showinfo("游戏结束", "黑棋获胜！")
                self.status_var.set("游戏结束: 黑棋获胜")
            elif self.game.winner == 2:
                messagebox.showinfo("游戏结束", "白棋获胜！")
                self.status_var.set("游戏结束: 白棋获胜")
            else:
                messagebox.showinfo("游戏结束", "平局！")
                self.status_var.set("游戏结束: 平局")
        else:
            # 更新状态
            current_player = "黑棋" if self.game.current_player == 1 else "白棋"
            self.status_var.set(f"当前回合: {current_player}")
    
    def ai_move(self):
        """AI落子逻辑"""
        # 记录思考开始时间
        think_start_time = time.time()
        
        # 使用模型预测最佳落子位置
        valid_moves = self.game.get_valid_moves()
        
        if valid_moves:
            # 获取当前局面特征
            board_feature = self.game.get_board_feature()
            
            # 使用模型预测
            policy_2d, _ = self.model.predict(board_feature)
            
            # 过滤出合法位置的概率
            valid_probs = []
            valid_move_indices = []
            
            for i, (r, c) in enumerate(valid_moves):
                prob = policy_2d[r, c]
                valid_probs.append(prob)
                valid_move_indices.append(i)
            
            # 根据AI强度调整概率分布
            temperature = 1.0 - self.ai_strength + 0.1  # 强度越高，temperature越小，策略越确定
            valid_probs = np.array(valid_probs)
            valid_probs = valid_probs ** (1.0 / temperature)
            valid_probs = valid_probs / np.sum(valid_probs)
            
            # 根据概率分布选择落子位置
            chosen_idx = np.random.choice(valid_move_indices, p=valid_probs)
            row, col = valid_moves[chosen_idx]
            
            # 在主线程中更新UI
            self.root.after(0, self._make_ai_move, row, col, time.time() - think_start_time)
    
    def _make_ai_move(self, row, col, think_time):
        """在主线程中执行AI落子并更新UI"""
        # 执行AI落子
        self.make_move(row, col)
        
        # 更新思考时间
        move_count = np.sum(self.game.board != 0)
        self.stats_var.set(f"落子数: {move_count} | AI思考时间: {think_time:.2f}s")
        
        # 重置AI思考状态
        self.is_ai_thinking = False
    
    def load_model(self):
        """加载AI模型"""
        from tkinter import filedialog
        
        # 打开文件选择对话框
        file_path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("模型文件", "*.h5"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                # 加载模型
                self.model = GobangModel(self.board_size, file_path)
                messagebox.showinfo("成功", f"模型加载成功: {file_path}")
                
                # 如果游戏未开始，开始新游戏
                if not self.is_game_started:
                    self.reset_game()
                
            except Exception as e:
                messagebox.showerror("错误", f"模型加载失败: {str(e)}")
    
    def update_ai_strength(self, value):
        """更新AI强度"""
        self.ai_strength = float(value)
    
    def reset_game(self):
        """重置游戏"""
        # 重置游戏状态
        self.game.reset()
        self.is_game_started = True
        
        # 重新绘制棋盘
        self.draw_board()
        
        # 更新状态
        self.status_var.set("当前回合: 黑棋")
        self.stats_var.set("落子数: 0 | 思考时间: 0.0s")
        
        # 重置AI思考状态
        self.is_ai_thinking = False

if __name__ == "__main__":
    # 设置中文字体支持
    root = tk.Tk()
    
    # 创建五子棋GUI
    gobang_gui = GobangGUI(root)
    
    # 启动主循环
    root.mainloop()