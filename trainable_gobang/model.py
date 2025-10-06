import numpy as np
import tensorflow as tf
# 配置GPU使用
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # 设置内存增长，避免一次性占用全部GPU内存
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"成功配置GPU: {[gpu.name for gpu in physical_devices]}")
    except RuntimeError as e:
        print(f"GPU配置错误: {e}")
else:
    print("未检测到可用GPU，将使用CPU进行计算")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    Flatten, Dense, Dropout, Add, Multiply, GlobalAveragePooling2D,
    LayerNormalization, Attention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from datetime import datetime
import os

# Swish激活函数定义
def swish(x):
    return x * tf.sigmoid(x)

# 确保中文显示正常
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

class GobangModel:
    def __init__(self, board_size=15, model_path=None, config=None):
        """初始化五子棋AI模型"""
        self.board_size = board_size
        
        # 默认配置 - 优化版本
        self.config = {
            'filters': 128,  # 特征通道数
            'residual_blocks': 8,  # 优化残差块数量，平衡复杂度和训练效率
            'attention_heads': 4,  # 注意力头数量
            'dropout_rate': 0.2,  # 降低dropout率以保留更多信息
            'learning_rate': 0.001,  # 提高初始学习率促进更快收敛
            'policy_temperature': 0.8,  # 策略温度参数
            'use_layer_norm': True,  # 使用LayerNormalization
            'use_attention': True,  # 使用注意力机制
            'use_swish': True,  # 使用Swish激活函数
            'use_position_encoding': True,  # 使用位置编码
            'value_scale': 1.0,         # 价值头输出缩放
            'policy_weight': 1.0,       # 策略头损失权重
            'value_weight': 1.0         # 价值头损失权重
        }
        
        # 更新配置
        if config:
            self.config.update(config)
            print(f"已更新模型配置: {config}")
        
        self.model = self._build_model()
        
        # 如果提供了模型路径，则加载已有模型
        if model_path:
            self.load_model(model_path)
            
        # 记录模型创建时间
        self.creation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"模型已初始化 (棋盘大小: {board_size}x{board_size}, 创建时间: {self.creation_time})")
            
    def _swish_activation(self, x):
        """Swish激活函数，比ReLU在某些情况下效果更好"""
        return x * tf.sigmoid(x)
        
    def _get_activation(self):
        """获取配置的激活函数"""
        if self.config['use_swish']:
            return lambda x: self._swish_activation(x)
        else:
            return 'relu'
            
    def _get_normalization(self):
        """获取配置的归一化层"""
        if self.config['use_layer_norm']:
            return LayerNormalization
        else:
            return BatchNormalization
    
    def _add_position_encoding(self, x):
        """添加位置编码，帮助模型理解棋盘位置关系"""
        if not self.config['use_position_encoding']:
            return x
        
        # 创建位置编码
        height, width = self.board_size, self.board_size
        channels = x.shape[-1]
        
        # 创建位置索引
        row_pos = tf.tile(tf.range(height)[:, tf.newaxis], [1, width])
        col_pos = tf.tile(tf.range(width)[tf.newaxis, :], [height, 1])
        
        # 将位置索引转换为特征
        row_embedding = tf.cast(row_pos, tf.float32) / (height - 1) * 2 - 1
        col_embedding = tf.cast(col_pos, tf.float32) / (width - 1) * 2 - 1
        
        # 扩展维度以匹配输入形状，但不使用批次维度（使用broadcast自动匹配）
        row_embedding = row_embedding[tf.newaxis, :, :, tf.newaxis]
        col_embedding = col_embedding[tf.newaxis, :, :, tf.newaxis]
        
        # 确保通道数匹配
        row_embedding = tf.tile(row_embedding, [1, 1, 1, channels])
        col_embedding = tf.tile(col_embedding, [1, 1, 1, channels])
        
        # 添加位置编码（通过broadcast自动匹配批次维度）
        x = x + row_embedding + col_embedding
        return x
        
    def _multi_head_attention(self, x, filters):
        """实现多头自注意力机制"""
        if not self.config['use_attention']:
            return x
        
        # 保存原始输入用于残差连接
        shortcut = x
        
        # 重塑为序列格式 (batch, seq_len, features)，避免显式使用批次维度
        seq_len = self.board_size * self.board_size
        features = filters
        
        # 使用 -1 自动计算批次维度
        x_reshaped = tf.keras.layers.Reshape((seq_len, features))(x)
        
        # 应用注意力机制
        attention_output = tf.keras.layers.Attention()([x_reshaped, x_reshaped])
        
        # 重塑回原始形状，使用 -1 自动计算批次维度
        attention_output = tf.keras.layers.Reshape((self.board_size, self.board_size, features))(attention_output)
        
        # 残差连接和归一化
        x = tf.keras.layers.Add()([shortcut, attention_output])
        x = self._get_normalization()()(x)
        
        return x
    
    def _residual_block(self, x, filters):
        """创建改进的残差块"""
        shortcut = x
        activation = self._get_activation()
        norm_layer = self._get_normalization()
        
        # 第一层卷积
        x = norm_layer()(x)
        x = Activation(activation)(x)
        x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        
        # 第二层卷积
        x = norm_layer()(x)
        x = Activation(activation)(x)
        x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        
        # 可选的注意力机制
        x = self._multi_head_attention(x, filters)
        
        # 残差连接
        x = Add()([x, shortcut])
        
        return x
        
    def _build_model(self):
        """构建增强的卷积神经网络模型"""
        # 输入形状：(board_size, board_size, 3)，3个通道分别表示黑棋、白棋和当前玩家
        inputs = Input(shape=(self.board_size, self.board_size, 3))
        
        # 获取配置参数
        filters = self.config['filters']
        residual_blocks = self.config['residual_blocks']
        activation = self._get_activation()
        norm_layer = self._get_normalization()
        
        # 初始卷积层 - 提取基础特征
        x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(inputs)
        
        # 添加位置编码
        x = self._add_position_encoding(x)
        
        # 初始归一化和激活
        x = norm_layer()(x)
        x = Activation(activation)(x)
        
        # 主体特征提取 - 增加更多残差块
        for i in range(residual_blocks):
            # 每4个残差块后添加一次注意力机制
            if i % 4 == 0:
                x = self._multi_head_attention(x, filters)
            x = self._residual_block(x, filters)
            
            # 在特定残差块后增加通道数
            if (i + 1) % 8 == 0 and filters < 256:
                # 使用1x1卷积进行升维
                shortcut = Conv2D(filters * 2, (1, 1), padding='same')(x)
                shortcut = norm_layer()(shortcut)
                
                # 更新主路径
                x = Conv2D(filters * 2, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
                x = Add()([x, shortcut])
                filters = filters * 2
        
        # 全局注意力模块 - 增强对整个棋盘的理解
        if self.config['use_attention']:
            # 全局特征池化
            global_features = GlobalAveragePooling2D()(x)
            global_features = Dense(filters // 4, activation=activation)(global_features)
            global_features = Dropout(self.config['dropout_rate'])(global_features)
            global_features = Dense(filters, activation='sigmoid')(global_features)
            
            # 将全局特征应用到空间特征上
            x = Multiply()([x, global_features[:, tf.newaxis, tf.newaxis, :]])
        
        # 策略头：预测落子概率分布 - 改进版本
        policy = Conv2D(4, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        policy = norm_layer()(policy)
        policy = Activation(activation)(policy)
        
        # 添加棋盘边缘感知模块，保持通道数一致
        edge_aware = Conv2D(4, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(policy)
        edge_aware = norm_layer()(edge_aware)
        edge_aware = Activation(activation)(edge_aware)
        
        policy = Add()([policy, edge_aware])
        policy = Flatten()(policy)
        policy = Dense(self.board_size * self.board_size, activation='softmax', name='policy')(policy)
        
        # 价值头：评估局面价值 - 改进版本
        value = Conv2D(2, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        value = norm_layer()(value)
        value = Activation(activation)(value)
        value = Flatten()(value)
        value = Dense(128, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(1e-5))(value)
        value = Dropout(self.config['dropout_rate'])(value)
        value = Dense(64, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(1e-5))(value)
        value = Dropout(self.config['dropout_rate'])(value)
        value = Dense(1, activation='tanh', name='value')(value)
        
        # 创建模型
        model = Model(inputs=inputs, outputs=[policy, value])
        
        # 编译模型 - 优化版本
        # 使用带权重衰减的AdamW优化器
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'],
            beta_1=0.95,  # 调整beta参数提高稳定性
            beta_2=0.999,
            epsilon=1e-8,
            clipnorm=1.0  # 梯度裁剪防止梯度爆炸
        )
        
        model.compile(
            optimizer=optimizer,
            loss={
                'policy': 'categorical_crossentropy',
                'value': 'mse'
            },
            loss_weights={
                'policy': self.config['policy_weight'],
                'value': self.config['value_weight']
            },
            metrics={
                'policy': 'accuracy',
                'value': 'mae'
            }
        )
        
        return model
    
    def predict(self, board_feature, temperature=None, return_probabilities=True):
        """预测给定局面的最佳落子位置和局面价值"""
        # 使用配置的温度参数或默认值
        temp = temperature if temperature is not None else self.config['policy_temperature']
        
        # 确保输入形状正确
        if len(board_feature.shape) == 3:
            board_feature = np.expand_dims(board_feature, axis=0)
        
        # 获取预测结果
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            policy_prob, value = self.model.predict(board_feature, verbose=0)
        
        # 将策略概率转换为二维数组
        policy_2d = policy_prob[0].reshape(self.board_size, self.board_size)
        
        # 应用温度参数调整概率分布
        if temp > 0 and temp != 1.0:
            policy_flat = policy_prob[0].copy()
            policy_flat = policy_flat ** (1.0 / temp)
            policy_flat = policy_flat / np.sum(policy_flat)
            policy_2d = policy_flat.reshape(self.board_size, self.board_size)
        
        if return_probabilities:
            return policy_2d, value[0][0]
        else:
            # 返回最佳位置
            best_move = np.unravel_index(np.argmax(policy_2d), policy_2d.shape)
            return best_move, value[0][0]
            
    def predict_multiple(self, board_features, batch_size=32):
        """批量预测多个局面"""
        # 确保输入形状正确
        if len(board_features.shape) == 3:
            board_features = np.expand_dims(board_features, axis=0)
        
        # 批量预测
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            policy_probs, values = self.model.predict(board_features, batch_size=batch_size, verbose=0)
        
        # 转换格式
        policy_2ds = []
        for policy_prob in policy_probs:
            policy_2d = policy_prob.reshape(self.board_size, self.board_size)
            policy_2ds.append(policy_2d)
        
        return np.array(policy_2ds), values.flatten()
        
    def save_model(self, path, include_config=True):
        """保存模型和配置"""
        # 保存模型权重和架构
        self.model.save(path)
        
        # 保存配置文件
        if include_config:
            config_path = path.replace('.h5', '_config.json')
            import json
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"模型配置已保存到: {config_path}")
        
        print(f"模型已保存到: {path}")
        
    def load_model(self, path, load_config=True):
        """加载模型和配置"""
        # 加载模型
        self.model = tf.keras.models.load_model(path)
        
        # 加载配置（如果存在）
        if load_config:
            config_path = path.replace('.h5', '_config.json')
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.config.update(config)
                print(f"已加载模型配置: {config_path}")
        
        # 确保模型结构与棋盘大小匹配
        input_shape = self.model.input_shape
        if input_shape[1] != self.board_size or input_shape[2] != self.board_size:
            print(f"警告：模型输入形状({input_shape[1]}x{input_shape[2]})与指定的棋盘大小({self.board_size}x{self.board_size})不匹配")
            self.board_size = input_shape[1]
        
        return self
    
    def train(self, x_train, policy_train, value_train, epochs=10, batch_size=128, 
              validation_split=0.1, callbacks=None, shuffle=True, use_cosine_schedule=False):
        """训练模型 - 优化版本"""
        # 创建保存检查点的目录
        os.makedirs('checkpoints', exist_ok=True)
        
        # 设置默认回调 - 优化版本
        default_callbacks = []
        
        # 添加模型检查点
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join('checkpoints', 'model_epoch_{epoch:02d}_loss_{loss:.4f}.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        default_callbacks.append(checkpoint)
        
        # 添加学习率调度器
        if use_cosine_schedule:
            # 使用余弦退火学习率调度器
            steps_per_epoch = max(1, len(x_train) // batch_size)
            total_steps = steps_per_epoch * epochs
            
            # 设置带预热的余弦退火学习率
            lr_schedule = CosineDecayRestarts(
                initial_learning_rate=self.config['learning_rate'],
                first_decay_steps=total_steps // 3,  # 第一个周期长度
                t_mul=1.5,  # 周期乘法因子
                m_mul=0.8,  # 学习率乘法因子
                alpha=0.01  # 最小学习率比例
            )
            
            lr_callback = tf.keras.callbacks.LearningRateScheduler(
                lambda epoch, lr: lr_schedule(epoch * steps_per_epoch)
            )
            default_callbacks.append(lr_callback)
        else:
            # 使用改进的学习率衰减策略
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # 降低因子
                patience=3,  # 更少的耐心，更快响应
                min_lr=1e-7,
                verbose=1
            )
            default_callbacks.append(lr_scheduler)
        
        # 添加早停策略
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,  # 更少的耐心，避免无效训练
            restore_best_weights=True,
            verbose=1
        )
        default_callbacks.append(early_stopping)
        
        # 添加学习率记录回调
        # 为了解决TensorBoard目录问题，暂时禁用TensorBoard回调
        # 注释掉TensorBoard回调以确保训练能够进行
        # 如果需要监控训练进度，可以在训练完成后启用
        
        # 如果提供了回调，则合并默认回调
        if callbacks:
            callbacks = default_callbacks + callbacks
        else:
            callbacks = default_callbacks
        
        # 训练模型
        history = self.model.fit(
            x_train,
            {'policy': policy_train, 'value': value_train},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            shuffle=shuffle,
            verbose=1
        )
        
        # 记录最终训练指标
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else None
        
        print(f"\n训练完成！最终训练损失: {final_loss:.4f}")
        if final_val_loss is not None:
            print(f"最终验证损失: {final_val_loss:.4f}")
        
        return history
        
    def evaluate_model(self, x_test, policy_test, value_test, batch_size=64):
        """评估模型性能"""
        # 在测试集上评估模型
        results = self.model.evaluate(
            x_test,
            {'policy': policy_test, 'value': value_test},
            batch_size=batch_size,
            verbose=1
        )
        
        # 提取评估指标
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = results[i]
        
        # 打印评估结果
        print("\n模型评估结果:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        
        return metrics
        
    def freeze_layers(self, num_layers_to_freeze=0):
        """冻结指定数量的层，用于迁移学习或微调"""
        # 获取所有可训练层
        trainable_layers = [layer for layer in self.model.layers if len(layer.trainable_weights) > 0]
        total_trainable = len(trainable_layers)
        
        # 计算要冻结的层数
        layers_to_freeze = min(num_layers_to_freeze, total_trainable)
        
        # 冻结层
        for i in range(layers_to_freeze):
            trainable_layers[i].trainable = False
        
        # 重新编译模型 - 使用更新的优化器配置
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'],
            beta_1=0.95,
            beta_2=0.999,
            epsilon=1e-8,
            clipnorm=1.0
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss={
                'policy': 'categorical_crossentropy',
                'value': 'mse'
            },
            loss_weights={
                'policy': self.config['policy_weight'],
                'value': self.config['value_weight']
            },
            metrics={
                'policy': 'accuracy',
                'value': 'mae'
            }
        )
        
        print(f"已冻结前{layers_to_freeze}/{total_trainable}个可训练层")
        
        return self
        
    def unfreeze_all(self):
        """解冻所有层"""
        # 解冻所有层
        for layer in self.model.layers:
            layer.trainable = True
        
        # 重新编译模型 - 使用更新的优化器配置
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'],
            beta_1=0.95,
            beta_2=0.999,
            epsilon=1e-8,
            clipnorm=1.0
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss={
                'policy': 'categorical_crossentropy',
                'value': 'mse'
            },
            loss_weights={
                'policy': self.config['policy_weight'],
                'value': self.config['value_weight']
            },
            metrics={
                'policy': 'accuracy',
                'value': 'mae'
            }
        )
        
        print("所有层已解冻")
        
        return self

    def make_move(self, board_feature, temperature=None):
        """根据当前棋盘状态和温度参数，使用模型预测选择最佳落子位置"""
        # 使用predict方法获取最佳位置
        best_move, value = self.predict(board_feature, temperature=temperature, return_probabilities=False)
        return best_move

# 测试代码
if __name__ == "__main__":
    # 创建模型实例
    model = GobangModel()
    
    # 打印模型结构
    model.model.summary()
    
    # 创建随机测试数据
    test_input = np.random.random((1, 15, 15, 3)).astype(np.float32)
    
    # 进行预测测试
    policy_2d, value = model.predict(test_input)
    print(f"Policy shape: {policy_2d.shape}")
    print(f"Value: {value}")
    
    # 测试make_move方法
    best_move = model.make_move(test_input)
    print(f"Best move: {best_move}")
    
    # 保存模型
    model.save_model("test_model.h5")
    print("Model saved successfully.")