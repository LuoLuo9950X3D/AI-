import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import os
import sys
import time
from datetime import datetime
import argparse

# 确保中文显示正常
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入我们的模型和游戏逻辑
from trainable_gobang.model import GobangModel
from trainable_gobang.gobang_game import GobangGame

class GobangUI:
    def __init__(self, root, model_path=None, ai_strength=None):
        self.root = root
        self.root.title("AI五子棋")
        self.root.resizable(False, False)
        
        # 基本配置
        self.board_size = 15
        self.cell_size = 40
        self.margin = 40
        
        # 游戏状态
        self.game = GobangGame(self.board_size)
        self.current_player = 1  # 1表示黑棋(玩家), 2表示白棋(AI)
        self.game_over = False
        
        # AI配置
        self.ai_thinking = False
        self.temperature = 1.0
        self.c_puct = 5.0
        self.num_simulations = 200
        # 使用传入的模型路径，如果没有则使用默认路径
        self.model_path = model_path if model_path else "models/best_model.h5"
        
        # 如果提供了AI强度参数，调整温度参数
        if ai_strength:
            try:
                strength = float(ai_strength)
                # AI强度和温度成反比关系：强度越高，温度越低
                self.temperature = max(0.1, min(2.0, 2.1 - strength*2))
                print(f"AI强度设置为: {strength}, 温度参数调整为: {self.temperature}")
            except ValueError:
                print(f"无效的AI强度值: {ai_strength}")
        
        # 创建模型
        self.model = None
        self.load_model()
        
        # 创建界面
        self.create_widgets()
        
        # 训练数据收集配置
        self.collect_training_data = False
        self.training_history = []
        self.data_collection_delay = 0.5  # 收集数据的延迟时间(秒)
        
    def create_widgets(self):
        # 创建主框架
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建左侧控制面板
        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 创建棋盘画布
        canvas_width = self.board_size * self.cell_size + 2 * self.margin
        canvas_height = self.board_size * self.cell_size + 2 * self.margin
        self.canvas = tk.Canvas(
            main_frame, 
            width=canvas_width, 
            height=canvas_height, 
            bg="#DEB887"
        )
        self.canvas.pack(side=tk.LEFT)
        
        # 绘制棋盘
        self.draw_board()
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # 添加控制按钮
        tk.Label(control_frame, text="AI五子棋", font=("SimHei", 16, "bold")).pack(pady=10)
        
        # 游戏控制
        tk.Button(control_frame, text="新游戏", command=self.start_new_game, width=15).pack(pady=5)
        tk.Button(control_frame, text="悔棋", command=self.undo_move, width=15).pack(pady=5)
        
        # AI设置
        tk.Label(control_frame, text="AI 设置", font=("SimHei", 12, "bold"), pady=10).pack()
        
        # 温度参数滑块
        tk.Label(control_frame, text="温度参数: {:.2f}".format(self.temperature)).pack()
        temp_scale = ttk.Scale(
            control_frame, 
            from_=0.1, 
            to=2.0, 
            orient="horizontal", 
            value=self.temperature,
            length=150,
            command=lambda val: self.update_temperature(val)
        )
        temp_scale.pack()
        
        # C_PUCT参数滑块
        tk.Label(control_frame, text="探索参数: {:.2f}".format(self.c_puct)).pack()
        c_puct_scale = ttk.Scale(
            control_frame, 
            from_=1.0, 
            to=10.0, 
            orient="horizontal", 
            value=self.c_puct,
            length=150,
            command=lambda val: self.update_c_puct(val)
        )
        c_puct_scale.pack()
        
        # 模拟次数选择
        tk.Label(control_frame, text="模拟次数: {}".format(self.num_simulations)).pack()
        sim_scale = ttk.Scale(
            control_frame, 
            from_=50, 
            to=500, 
            orient="horizontal", 
            value=self.num_simulations,
            length=150,
            command=lambda val: self.update_simulations(val)
        )
        sim_scale.pack()
        
        # 训练数据收集
        tk.Label(control_frame, text="训练数据", font=("SimHei", 12, "bold"), pady=10).pack()
        self.train_data_var = tk.BooleanVar(value=False)
        tk.Checkbutton(control_frame, text="收集训练数据", variable=self.train_data_var, command=self.toggle_data_collection).pack(pady=5)
        
        # 模型状态
        model_status = "模型已加载" if self.model else "未加载模型"
        self.model_status_label = tk.Label(control_frame, text=model_status, font=("SimHei", 10), fg="green" if self.model else "red")
        self.model_status_label.pack(pady=20)
        
        # 游戏状态显示
        self.game_status_label = tk.Label(control_frame, text="轮到你下棋", font=("SimHei", 12), fg="black")
        self.game_status_label.pack(pady=10)
        
    def draw_board(self):
        # 清空画布
        self.canvas.delete("all")
        
        # 绘制棋盘网格
        for i in range(self.board_size):
            # 横线
            self.canvas.create_line(
                self.margin, 
                self.margin + i * self.cell_size, 
                self.margin + (self.board_size - 1) * self.cell_size, 
                self.margin + i * self.cell_size,
                width=2
            )
            # 竖线
            self.canvas.create_line(
                self.margin + i * self.cell_size, 
                self.margin, 
                self.margin + i * self.cell_size, 
                self.margin + (self.board_size - 1) * self.cell_size,
                width=2
            )
        
        # 绘制棋盘上的五个点
        star_points = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
        for x, y in star_points:
            self.canvas.create_oval(
                self.margin + x * self.cell_size - 5, 
                self.margin + y * self.cell_size - 5, 
                self.margin + x * self.cell_size + 5, 
                self.margin + y * self.cell_size + 5,
                fill="black"
            )
        
        # 绘制已下的棋子
        for i in range(self.board_size):
            for j in range(self.board_size):
                piece = self.game.board[i][j]
                if piece != 0:
                    color = "black" if piece == 1 else "white"
                    outline = "#8B4513" if piece == 1 else "#D2B48C"
                    self.canvas.create_oval(
                        self.margin + i * self.cell_size - 18, 
                        self.margin + j * self.cell_size - 18, 
                        self.margin + i * self.cell_size + 18, 
                        self.margin + j * self.cell_size + 18,
                        fill=color, 
                        outline=outline,
                        width=2
                    )
        
    def on_canvas_click(self, event):
        if self.game_over or self.ai_thinking:
            return
        
        # 计算点击位置对应的棋盘坐标
        x = round((event.x - self.margin) / self.cell_size)
        y = round((event.y - self.margin) / self.cell_size)
        
        # 检查坐标是否有效
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return
        
        # 检查该位置是否为空
        if self.game.board[x][y] != 0:
            return
        
        # 玩家落子
        self.make_move(x, y)
        
    def make_move(self, x, y):
        # 记录落子前的棋盘状态用于训练数据收集
        if self.collect_training_data:
            board_state = np.copy(self.game.board)
            player = self.current_player
            
        # 执行落子
        self.game.place_stone(x, y, self.current_player)
        self.draw_board()
        
        # 检查游戏是否结束
        if self.game.check_win(x, y, self.current_player):
            self.game_over = True
            winner = "你赢了！" if self.current_player == 1 else "AI赢了！"
            self.game_status_label.config(text=winner, fg="red")
            messagebox.showinfo("游戏结束", winner)
            return
        
        # 检查是否平局
        if self.game.is_board_full():
            self.game_over = True
            self.game_status_label.config(text="平局！", fg="blue")
            messagebox.showinfo("游戏结束", "平局！")
            return
        
        # 收集训练数据（如果启用）
        if self.collect_training_data:
            # 创建棋盘特征
            board_feature = self.game.get_board_feature(self.current_player)
            # 记录训练数据
            self.training_history.append({
                'board': board_state,
                'player': player,
                'move': (x, y),
                'result': 0  # 游戏继续
            })
            
        # 切换玩家
        self.current_player = 2 if self.current_player == 1 else 1
        
        # 如果是AI回合，执行AI落子
        if self.current_player == 2 and not self.game_over:
            self.game_status_label.config(text="AI思考中...", fg="blue")
            self.ai_thinking = True
            self.root.update()
            
            # 在新线程中执行AI思考，避免界面卡顿
            self.root.after(100, self.ai_move)
        else:
            self.game_status_label.config(text="轮到你下棋", fg="black")
            
    def ai_move(self):
        if self.model is None:
            # 如果没有加载模型，使用随机落子
            move = self.game.get_random_move()
            time.sleep(0.5)  # 添加延迟，模拟思考过程
        else:
            # 获取棋盘特征
            board_feature = self.game.get_board_feature(self.current_player)
            
            # 使用模型预测最佳落子位置
            # 记录开始时间，用于训练数据收集时的思考时间
            start_time = time.time()
            
            # 执行预测
            best_move = self.model.make_move(board_feature, temperature=self.temperature)
            
            # 模拟MCTS搜索的延迟
            elapsed = time.time() - start_time
            if elapsed < self.data_collection_delay:
                time.sleep(self.data_collection_delay - elapsed)
            
            move = best_move
        
        # 执行AI落子
        if move is not None:
            self.make_move(*move)
            
        # 重置AI思考状态
        self.ai_thinking = False
        
    def start_new_game(self):
        # 处理上一局的训练数据
        if self.collect_training_data and self.training_history:
            self.save_training_data()
            
        # 重置游戏状态
        self.game = GobangGame(self.board_size)
        self.current_player = 1
        self.game_over = False
        self.ai_thinking = False
        self.training_history = []
        
        # 重绘画布
        self.draw_board()
        self.game_status_label.config(text="轮到你下棋", fg="black")
        
    def undo_move(self):
        if self.game_over or self.ai_thinking:
            return
        
        # 尝试悔棋两步（玩家和AI各一步）
        if len(self.game.move_history) >= 2:
            self.game.undo_move()
            self.game.undo_move()
            self.current_player = 1
            self.draw_board()
            self.game_status_label.config(text="轮到你下棋", fg="black")
        elif len(self.game.move_history) == 1:
            # 只悔棋一步（只有玩家下了一步）
            self.game.undo_move()
            self.current_player = 1
            self.draw_board()
            self.game_status_label.config(text="轮到你下棋", fg="black")
        else:
            messagebox.showinfo("提示", "没有可悔的棋")
            
    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                self.model = GobangModel(board_size=self.board_size)
                self.model.load_model(self.model_path)
                self.model_status_label.config(text="模型已加载", fg="green")
                return True
            else:
                self.model_status_label.config(text="未找到模型", fg="orange")
                return False
        except Exception as e:
            self.model_status_label.config(text=f"加载模型失败: {str(e)}", fg="red")
            return False
            
    def update_temperature(self, val):
        self.temperature = float(val)
        # 更新标签显示
        for child in self.root.winfo_children():
            if isinstance(child, tk.Frame):
                for grandchild in child.winfo_children():
                    if isinstance(grandchild, tk.Frame):
                        for great_grandchild in grandchild.winfo_children():
                            if isinstance(great_grandchild, tk.Label) and "温度参数" in great_grandchild.cget("text"):
                                great_grandchild.config(text="温度参数: {:.2f}".format(self.temperature))
                                return
    
    def update_c_puct(self, val):
        self.c_puct = float(val)
        # 更新标签显示
        for child in self.root.winfo_children():
            if isinstance(child, tk.Frame):
                for grandchild in child.winfo_children():
                    if isinstance(grandchild, tk.Frame):
                        for great_grandchild in grandchild.winfo_children():
                            if isinstance(great_grandchild, tk.Label) and "探索参数" in great_grandchild.cget("text"):
                                great_grandchild.config(text="探索参数: {:.2f}".format(self.c_puct))
                                return
    
    def update_simulations(self, val):
        self.num_simulations = int(float(val))
        # 更新标签显示
        for child in self.root.winfo_children():
            if isinstance(child, tk.Frame):
                for grandchild in child.winfo_children():
                    if isinstance(grandchild, tk.Frame):
                        for great_grandchild in grandchild.winfo_children():
                            if isinstance(great_grandchild, tk.Label) and "模拟次数" in great_grandchild.cget("text"):
                                great_grandchild.config(text="模拟次数: {}".format(self.num_simulations))
                                return
    
    def toggle_data_collection(self):
        self.collect_training_data = self.train_data_var.get()
        if not self.collect_training_data and self.training_history:
            # 如果关闭数据收集且有未保存的数据，提示保存
            if messagebox.askyesno("保存训练数据", "是否保存当前训练数据？"):
                self.save_training_data()
                
    def save_training_data(self):
        if not self.training_history:
            return
            
        try:
            # 确保数据目录存在
            data_dir = "training_data/human_vs_ai"
            os.makedirs(data_dir, exist_ok=True)
            
            # 生成唯一的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(data_dir, f"human_vs_ai_{timestamp}.npz")
            
            # 准备数据
            boards = []
            moves = []
            players = []
            
            for record in self.training_history:
                boards.append(record['board'])
                moves.append(record['move'])
                players.append(record['player'])
            
            # 保存数据
            np.savez_compressed(
                file_path,
                boards=np.array(boards),
                moves=np.array(moves),
                players=np.array(players),
                metadata={
                    'board_size': self.board_size,
                    'temperature': self.temperature,
                    'c_puct': self.c_puct,
                    'num_simulations': self.num_simulations,
                    'timestamp': timestamp,
                    'model_path': self.model_path
                }
            )
            
            messagebox.showinfo("保存成功", f"训练数据已保存到:\n{file_path}")
            self.training_history = []
            
        except Exception as e:
            messagebox.showerror("保存失败", f"保存训练数据时出错:\n{str(e)}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='AI五子棋游戏界面')
    parser.add_argument('--model_path', type=str, default=None, help='AI模型路径')
    parser.add_argument('--ai_strength', type=str, default=None, help='AI强度(0.0-1.0)')
    args = parser.parse_args()
    
    # 创建主窗口
    root = tk.Tk()
    
    # 创建游戏界面实例，传入解析的参数
    app = GobangUI(root, model_path=args.model_path, ai_strength=args.ai_strength)
    
    # 运行主循环
    root.mainloop()