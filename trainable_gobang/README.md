# 可训练的AI五子棋项目

这是一个基于深度学习的五子棋AI项目，与传统基于规则和Alpha-Beta剪枝的五子棋AI不同，本项目中的AI需要通过训练来提高棋力。

## 项目特点

- 使用深度学习技术（卷积神经网络）实现五子棋AI
- 支持自我对弈生成训练数据
- 实现了基于策略梯度和价值函数的强化学习算法
- 提供完整的训练、评估和对战功能

## 项目结构

- `gobang_game.py`: 五子棋游戏核心逻辑
- `model.py`: 神经网络模型定义
- `train.py`: 训练脚本
- `play.py`: 人机对战界面
- `self_play.py`: 自我对弈生成训练数据
- `evaluate.py`: 评估模型性能
- `requirements.txt`: 项目依赖

## 环境配置

1. 安装Python 3.6或更高版本
2. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

## 使用说明

### 1. 训练模型

```
python train.py --epochs 100 --batch_size 64
```

### 2. 自我对弈生成数据

```
python self_play.py --num_games 1000 --output_dir data
```

### 3. 人机对战

```
python play.py --model_path models/best_model.h5
```

### 4. 评估模型

```
python evaluate.py --model1 models/model_v1.h5 --model2 models/model_v2.h5 --num_games 100
```

## 技术原理

本项目基于深度强化学习技术，主要包括以下几个部分：

1. **神经网络模型**：使用卷积神经网络来评估棋盘局面和预测最佳落子位置
2. **策略梯度算法**：通过自我对弈的结果来更新模型参数
3. **蒙特卡洛树搜索**：结合神经网络模型进行高效的搜索

通过不断的训练和自我对弈，AI可以逐步提高其棋力水平。

## 注意事项

- 训练过程可能需要较长时间，建议在GPU环境下进行
- 初始模型棋力较弱，需要经过足够的训练才能达到较好的水平
- 自我对弈生成的数据量会影响训练效果，建议生成足够多的训练样本