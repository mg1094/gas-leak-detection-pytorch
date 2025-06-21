"""
燃气泄漏检测AI模型
PyTorch LSTM + SVM 混合模型实现
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import logging
import os
from datetime import datetime
from config import Config

# 尝试导入 PyTorch，如果失败则使用替代方案
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
    
    class GasLeakLSTM(nn.Module):
        """PyTorch LSTM模型用于燃气泄漏检测特征提取"""
        
        def __init__(self, input_size, hidden_size, num_layers, output_size=8):
            super(GasLeakLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            # LSTM层
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=0.2)
            
            # 全连接层
            self.fc1 = nn.Linear(hidden_size, 32)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, output_size)
            
            # Dropout层
            self.dropout1 = nn.Dropout(0.3)
            self.dropout2 = nn.Dropout(0.2)
            
            # 激活函数
            self.relu = nn.ReLU()
            
        def forward(self, x):
            # 初始化隐藏状态
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            
            # LSTM前向传播
            out, _ = self.lstm(x, (h0, c0))
            
            # 取最后一个时间步的输出
            out = out[:, -1, :]
            
            # 全连接层
            out = self.relu(self.fc1(out))
            out = self.dropout1(out)
            out = self.relu(self.fc2(out))
            out = self.dropout2(out)
            out = self.relu(self.fc3(out))
            
            return out
    
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️ PyTorch 不可用，将使用纯 SVM 模型")
    
    # 当PyTorch不可用时，定义一个占位符类
    class GasLeakLSTM:
        def __init__(self, *args, **kwargs):
            pass

class GasLeakDetectionModel:
    """燃气泄漏检测混合模型（PyTorch LSTM + SVM）"""
    
    def __init__(self):
        self.config = Config()
        self.lstm_model = None
        self.svm_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.pytorch_available = PYTORCH_AVAILABLE
        
        # 模型参数
        self.sequence_length = self.config.MODEL_CONFIG['sequence_length']
        self.features_count = self.config.MODEL_CONFIG['features_count']
        self.lstm_units = self.config.MODEL_CONFIG['lstm_units']
        self.lstm_layers = self.config.MODEL_CONFIG['lstm_layers']
        
        # PyTorch设备选择
        if self.pytorch_available:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if not self.pytorch_available:
            self.logger.warning("PyTorch 不可用，将使用纯 SVM 模型进行检测")
        elif self.pytorch_available:
            self.logger.info(f"PyTorch 可用，使用设备: {self.device}")
        
    def build_lstm_model(self):
        """构建PyTorch LSTM模型"""
        if not self.pytorch_available:
            self.logger.warning("PyTorch 不可用，跳过 LSTM 模型构建")
            return None
            
        # 创建PyTorch LSTM模型
        self.lstm_model = GasLeakLSTM(
            input_size=self.features_count,
            hidden_size=self.lstm_units,
            num_layers=self.lstm_layers,
            output_size=8
        ).to(self.device)
        
        # 定义优化器和损失函数
        self.optimizer = optim.Adam(
            self.lstm_model.parameters(), 
            lr=self.config.MODEL_CONFIG['learning_rate']
        )
        self.criterion = nn.MSELoss()
        
        self.logger.info("PyTorch LSTM模型构建完成")
        return self.lstm_model
        
    def build_svm_model(self):
        """构建SVM模型"""
        self.svm_model = SVC(
            kernel=self.config.SVM_CONFIG['kernel'],
            C=self.config.SVM_CONFIG['C'],
            gamma=self.config.SVM_CONFIG['gamma'],
            probability=True  # 启用概率预测
        )
        self.logger.info("SVM模型构建完成")
        return self.svm_model
        
    def predict(self, sequence_data):
        """预测燃气泄漏"""
        if not self.is_trained:
            self.logger.warning("模型尚未训练，无法进行预测")
            return None, None
            
        try:
            # 确保输入数据格式正确
            if len(sequence_data.shape) == 2:
                sequence_data = sequence_data.reshape(1, *sequence_data.shape)
                
            if self.pytorch_available and self.lstm_model is not None:
                # 使用 PyTorch LSTM + SVM 混合模型预测
                # 转换为PyTorch张量
                sequence_tensor = torch.FloatTensor(sequence_data).to(self.device)
                
                # 切换到评估模式
                self.lstm_model.eval()
                
                with torch.no_grad():
                    # LSTM特征提取
                    features = self.lstm_model(sequence_tensor).cpu().numpy()
                
                # 特征标准化
                features_scaled = self.scaler.transform(features)
            else:
                # 使用纯 SVM 模型预测
                # 将序列数据展平
                sequence_flat = sequence_data.reshape(sequence_data.shape[0], -1)
                
                # 特征标准化
                features_scaled = self.scaler.transform(sequence_flat)
            
            # SVM预测
            prediction = self.svm_model.predict(features_scaled)[0]
            probability = self.svm_model.predict_proba(features_scaled)[0]
            
            return prediction, probability[1]  # 返回泄漏概率
            
        except Exception as e:
            self.logger.error(f"预测失败: {e}")
            return None, None
    
    def get_model_info(self):
        """获取模型信息"""
        info = {
            'pytorch_available': self.pytorch_available,
            'is_trained': self.is_trained,
            'model_type': 'PyTorch LSTM+SVM' if self.pytorch_available and self.lstm_model else 'Pure SVM',
            'sequence_length': self.sequence_length,
            'features_count': self.features_count
        }
        
        if self.pytorch_available:
            info['device'] = str(self.device)
            
        return info

# 示例使用
if __name__ == '__main__':
    # 创建模型实例
    model = GasLeakDetectionModel()
    
    # 构建模型
    lstm_model = model.build_lstm_model()
    svm_model = model.build_svm_model()
    
    print("燃气泄漏检测模型构建完成！")
    if lstm_model and model.pytorch_available:
        print(f"PyTorch LSTM模型结构:")
        print(lstm_model)
        print(f"设备: {model.device}")
    if svm_model:
        print(f"SVM模型参数: {svm_model.get_params()}")