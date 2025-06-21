"""
燃气泄漏检测系统 - 训练配置优化器
Training Configuration Optimizer for Gas Leak Detection System
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import argparse

class TrainingConfigOptimizer:
    """训练配置优化器"""
    
    def __init__(self):
        # 基准配置
        self.base_config = {
            'lstm_units': 50,
            'lstm_layers': 2,
            'sequence_length': 60,
            'features_count': 3,
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001
        }
        
        # 模型复杂度
        self.model_params = self._calculate_model_params()
        
    def _calculate_model_params(self):
        """计算模型参数量"""
        config = self.base_config
        
        # LSTM参数量计算
        # 公式: 4 * (input_size + hidden_size + 1) * hidden_size
        lstm_params = 0
        
        # 第一层LSTM
        first_layer = 4 * (config['features_count'] + config['lstm_units'] + 1) * config['lstm_units']
        lstm_params += first_layer
        
        # 后续LSTM层
        for i in range(1, config['lstm_layers']):
            layer_params = 4 * (config['lstm_units'] + config['lstm_units'] + 1) * config['lstm_units']
            lstm_params += layer_params
        
        # 全连接层参数 (简化估算)
        fc_params = config['lstm_units'] * 32 + 32 * 16 + 16 * 8
        
        total_params = lstm_params + fc_params
        
        return {
            'lstm_params': lstm_params,
            'fc_params': fc_params,
            'total_params': total_params
        }
        
    def analyze_data(self, data_path=None):
        """分析训练数据"""
        if data_path is None:
            # 分析测试用例数据
            data_info = self._analyze_test_cases()
        else:
            # 分析指定数据文件
            data_info = self._analyze_data_file(data_path)
            
        return data_info
        
    def _analyze_test_cases(self):
        """分析测试用例数据"""
        test_cases_dir = 'data/test_cases'
        
        if not os.path.exists(test_cases_dir):
            return None
            
        total_samples = 0
        total_leak_samples = 0
        total_normal_samples = 0
        data_files = []
        
        for filename in os.listdir(test_cases_dir):
            if filename.endswith('.csv') and filename != 'test_summary.json':
                file_path = os.path.join(test_cases_dir, filename)
                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    
                    # 计算序列样本数量
                    sequence_samples = max(0, len(df) - self.base_config['sequence_length'] + 1)
                    leak_samples = df['is_leak'].sum() if 'is_leak' in df.columns else 0
                    normal_samples = sequence_samples - leak_samples
                    
                    total_samples += sequence_samples
                    total_leak_samples += leak_samples
                    total_normal_samples += normal_samples
                    
                    data_files.append({
                        'filename': filename,
                        'total_points': len(df),
                        'sequence_samples': sequence_samples,
                        'leak_samples': leak_samples,
                        'normal_samples': normal_samples,
                        'time_span': (df.index[-1] - df.index[0]).total_seconds() / 3600  # 小时
                    })
                    
                except Exception as e:
                    print(f"⚠️ 分析文件 {filename} 时出错: {e}")
                    
        return {
            'total_sequence_samples': total_samples,
            'total_leak_samples': total_leak_samples,
            'total_normal_samples': total_normal_samples,
            'leak_ratio': total_leak_samples / total_samples if total_samples > 0 else 0,
            'data_files': data_files,
            'estimated_time_span': sum(f['time_span'] for f in data_files)
        }

    def recommend_config(self, data_info):
        """根据数据量推荐训练配置"""
        if data_info is None:
            print("❌ 无法获取数据信息，使用默认配置")
            return self.base_config.copy()
            
        total_samples = data_info['total_sequence_samples']
        leak_ratio = data_info['leak_ratio']
        time_span = data_info['estimated_time_span']
        
        print(f"📊 数据分析结果:")
        print(f"   总序列样本数: {total_samples:,}")
        print(f"   泄漏样本数: {data_info['total_leak_samples']:,}")
        print(f"   正常样本数: {data_info['total_normal_samples']:,}")
        print(f"   泄漏比例: {leak_ratio:.1%}")
        print(f"   数据时间跨度: {time_span:.1f} 小时")
        
        # 根据数据量级别确定配置
        if total_samples < 100_000:
            level = "小数据集"
            config = self._get_small_dataset_config()
        elif total_samples < 1_000_000:
            level = "中等数据集"
            config = self._get_medium_dataset_config()
        elif total_samples < 10_000_000:
            level = "大数据集"
            config = self._get_large_dataset_config()
        else:
            level = "超大数据集"
            config = self._get_xlarge_dataset_config()
            
        print(f"\n🎯 推荐配置级别: {level}")
        
        return config

    def _get_small_dataset_config(self):
        """小数据集配置"""
        config = self.base_config.copy()
        config.update({
            'epochs': 150,
            'batch_size': 64,
            'learning_rate': 0.01,
            'early_stopping_patience': 20,
            'lr_decay_patience': 10,
            'training_mode': '快速验证'
        })
        return config

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='燃气泄漏检测系统训练配置优化器')
    parser.add_argument('--data-path', type=str, help='训练数据文件路径')
    parser.add_argument('--output', type=str, default='optimized_config.json', 
                       help='输出配置文件路径')
    parser.add_argument('--analyze-only', action='store_true', 
                       help='仅分析数据，不生成配置')
    
    args = parser.parse_args()
    
    print("🔧 燃气泄漏检测系统 - 训练配置优化器")
    print("=" * 60)
    
    # 创建优化器
    optimizer = TrainingConfigOptimizer()
    
    # 分析数据
    print("🔍 分析训练数据...")
    data_info = optimizer.analyze_data(args.data_path)
    
    if data_info is None:
        print("❌ 无法分析数据，请检查数据文件路径")
        return
        
    if args.analyze_only:
        print("\n✅ 数据分析完成")
        return
        
    # 推荐配置
    print("\n🎯 生成训练配置建议...")
    config = optimizer.recommend_config(data_info)
    
    print(f"\n✅ 配置优化完成！")

if __name__ == '__main__':
    main()