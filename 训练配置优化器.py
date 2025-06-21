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
        
    def _analyze_data_file(self, data_path):
        """分析单个数据文件"""
        try:
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            
            # 计算序列样本数量
            sequence_samples = max(0, len(df) - self.base_config['sequence_length'] + 1)
            leak_samples = df['is_leak'].sum() if 'is_leak' in df.columns else 0
            normal_samples = sequence_samples - leak_samples
            
            time_span = (df.index[-1] - df.index[0]).total_seconds() / 3600  # 小时
            
            return {
                'total_sequence_samples': sequence_samples,
                'total_leak_samples': leak_samples,
                'total_normal_samples': normal_samples,
                'leak_ratio': leak_samples / sequence_samples if sequence_samples > 0 else 0,
                'estimated_time_span': time_span,
                'data_files': [{
                    'filename': os.path.basename(data_path),
                    'total_points': len(df),
                    'sequence_samples': sequence_samples,
                    'leak_samples': leak_samples,
                    'normal_samples': normal_samples,
                    'time_span': time_span
                }]
            }
            
        except Exception as e:
            print(f"❌ 分析数据文件失败: {e}")
            return None
            
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
            
        # 根据数据不平衡调整配置
        config = self._adjust_for_imbalance(config, leak_ratio)
        
        # 根据时间跨度调整配置
        config = self._adjust_for_timespan(config, time_span)
        
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
        
    def _get_medium_dataset_config(self):
        """中等数据集配置"""
        config = self.base_config.copy()
        config.update({
            'epochs': 120,
            'batch_size': 32,
            'learning_rate': 0.001,
            'early_stopping_patience': 15,
            'lr_decay_patience': 8,
            'training_mode': '标准训练'
        })
        return config
        
    def _get_large_dataset_config(self):
        """大数据集配置"""
        config = self.base_config.copy()
        config.update({
            'epochs': 80,
            'batch_size': 16,
            'learning_rate': 0.0005,
            'early_stopping_patience': 12,
            'lr_decay_patience': 6,
            'training_mode': '高效训练'
        })
        return config
        
    def _get_xlarge_dataset_config(self):
        """超大数据集配置"""
        config = self.base_config.copy()
        config.update({
            'epochs': 50,
            'batch_size': 8,
            'learning_rate': 0.0001,
            'early_stopping_patience': 10,
            'lr_decay_patience': 5,
            'training_mode': '分布式训练'
        })
        return config
        
    def _adjust_for_imbalance(self, config, leak_ratio):
        """根据数据不平衡调整配置"""
        if leak_ratio < 0.05:  # 泄漏样本少于5%
            config['class_weight'] = 'balanced'
            config['epochs'] = int(config['epochs'] * 1.2)  # 增加训练轮数
            print("   ⚠️ 检测到数据不平衡，启用类别权重平衡")
        elif leak_ratio > 0.3:  # 泄漏样本超过30%
            config['epochs'] = int(config['epochs'] * 0.8)  # 减少训练轮数
            print("   ℹ️ 泄漏样本较多，适当减少训练轮数")
            
        return config
        
    def _adjust_for_timespan(self, config, time_span):
        """根据时间跨度调整配置"""
        if time_span < 24:  # 少于1天
            config['validation_split'] = 0.3  # 增加验证集比例
            print("   ⚠️ 数据时间跨度较短，增加验证集比例")
        elif time_span > 720:  # 超过30天
            config['validation_split'] = 0.15  # 减少验证集比例
            config['shuffle'] = True  # 启用数据打乱
            print("   ✅ 数据时间跨度充足，启用数据打乱")
        else:
            config['validation_split'] = 0.2  # 标准验证集比例
            
        return config
        
    def estimate_training_time(self, config, data_info):
        """估算训练时间"""
        if data_info is None:
            return "无法估算"
            
        total_samples = data_info['total_sequence_samples']
        epochs = config['epochs']
        batch_size = config['batch_size']
        
        # 基于经验的时间估算（秒）
        # 假设每个样本处理时间约为0.001秒（CPU）或0.0001秒（GPU）
        time_per_sample = 0.001  # CPU时间
        
        # 计算每个epoch的时间
        batches_per_epoch = np.ceil(total_samples / batch_size)
        time_per_epoch = batches_per_epoch * batch_size * time_per_sample
        
        # 总训练时间
        total_time_seconds = time_per_epoch * epochs
        
        # 转换为小时
        total_time_hours = total_time_seconds / 3600
        
        if total_time_hours < 1:
            return f"{total_time_seconds/60:.0f} 分钟"
        elif total_time_hours < 24:
            return f"{total_time_hours:.1f} 小时"
        else:
            return f"{total_time_hours/24:.1f} 天"
            
    def generate_config_file(self, config, output_path='optimized_config.json'):
        """生成配置文件"""
        config_with_meta = {
            'generation_time': datetime.now().isoformat(),
            'model_params': self.model_params,
            'training_config': config,
            'usage_instructions': {
                'description': '燃气泄漏检测系统优化训练配置',
                'how_to_use': [
                    '将此配置文件导入到训练脚本中',
                    '根据实际硬件资源调整batch_size',
                    '监控训练过程中的损失变化',
                    '使用早停机制避免过拟合'
                ]
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_with_meta, f, indent=2, ensure_ascii=False)
            
        print(f"📄 配置文件已保存到: {output_path}")
        
    def print_recommendations(self, config, data_info):
        """打印训练建议"""
        print("\n🎯 训练配置建议:")
        print("=" * 50)
        
        # 基础配置
        print("📋 基础参数:")
        print(f"   训练轮数 (epochs): {config['epochs']}")
        print(f"   批次大小 (batch_size): {config['batch_size']}")
        print(f"   学习率 (learning_rate): {config['learning_rate']}")
        print(f"   训练模式: {config.get('training_mode', '标准训练')}")
        
        # 高级配置
        print(f"\n⚙️ 高级参数:")
        print(f"   早停耐心值: {config.get('early_stopping_patience', 15)}")
        print(f"   学习率衰减耐心值: {config.get('lr_decay_patience', 8)}")
        print(f"   验证集比例: {config.get('validation_split', 0.2)}")
        
        if 'class_weight' in config:
            print(f"   类别权重: {config['class_weight']}")
            
        # 时间估算
        estimated_time = self.estimate_training_time(config, data_info)
        print(f"\n⏱️ 预估训练时间: {estimated_time}")
        
        # 硬件建议
        print(f"\n💻 硬件建议:")
        if data_info and data_info['total_sequence_samples'] > 1_000_000:
            print("   - 推荐使用GPU加速训练")
            print("   - 内存需求: 16GB+")
            print("   - 存储空间: 10GB+")
        else:
            print("   - CPU训练即可满足需求")
            print("   - 内存需求: 8GB+")
            print("   - 存储空间: 5GB+")
            
        # 监控建议
        print(f"\n📊 训练监控建议:")
        print("   - 监控训练/验证损失曲线")
        print("   - 关注准确率、精确率、召回率变化")
        print("   - 使用TensorBoard可视化训练过程")
        print("   - 定期保存模型检查点")

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
    
    # 打印建议
    optimizer.print_recommendations(config, data_info)
    
    # 生成配置文件
    print(f"\n💾 保存配置文件...")
    optimizer.generate_config_file(config, args.output)
    
    print(f"\n✅ 配置优化完成！")
    print(f"📁 配置文件: {args.output}")
    print(f"💡 使用方法: 将配置导入到训练脚本中")

if __name__ == '__main__':
    main()