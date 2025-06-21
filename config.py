"""
燃气泄漏检测系统配置文件
Gas Leak Detection System Configuration
"""
import os

class Config:
    """系统配置类"""
    
    # 数据采集配置
    SENSOR_CONFIG = {
        'gas_sensor_pin': 0,        # 燃气传感器引脚
        'temp_sensor_pin': 1,       # 温度传感器引脚
        'humidity_sensor_pin': 2,   # 湿度传感器引脚
        'sampling_rate': 1,         # 采样频率（秒）
        'data_buffer_size': 60      # 数据缓冲区大小（1分钟数据）
    }
    
    # 模型参数配置
    MODEL_CONFIG = {
        'lstm_units': 50,           # LSTM单元数量
        'lstm_layers': 2,           # LSTM层数
        'sequence_length': 60,      # 时间序列长度（60秒）
        'features_count': 3,        # 特征数量（燃气浓度、温度、湿度）
        'batch_size': 32,           # 训练批次大小
        'epochs': 100,              # 训练轮数
        'learning_rate': 0.001      # 学习率
    }
    
    # SVM配置
    SVM_CONFIG = {
        'kernel': 'rbf',            # SVM核函数
        'C': 1.0,                   # 正则化参数
        'gamma': 'scale'            # 核函数参数
    }
    
    # 阈值设置
    THRESHOLD_CONFIG = {
        'gas_concentration_danger': 1000,  # 燃气浓度危险阈值(ppm)
        'gas_concentration_warning': 500,  # 燃气浓度警告阈值(ppm)
        'prediction_threshold': 0.7,       # 模型预测阈值
        'consecutive_alerts': 3             # 连续预警次数才触发警报
    }
    
    # 预警配置
    ALERT_CONFIG = {
        'local_alarm_pin': 8,       # 本地报警器引脚
        'app_push_url': 'http://api.push.server.com/send',  # APP推送API
        'emergency_contacts': [      # 紧急联系人
            {'name': '用户', 'phone': '13800138000'},
            {'name': '紧急联系人', 'phone': '13900139000'}
        ],
        'alert_sound_duration': 10   # 警报声持续时间（秒）
    }
    
    # 数据存储配置
    DATABASE_CONFIG = {
        'data_file_path': 'data/sensor_data.csv',      # 传感器数据存储路径
        'model_save_path': 'models/',                   # 模型保存路径
        'log_file_path': 'logs/system.log',            # 日志文件路径
        'backup_interval': 24 * 60 * 60                # 数据备份间隔（秒）
    }
    
    # 云端服务配置
    CLOUD_CONFIG = {
        'server_url': 'http://localhost:5000',         # 云端服务器地址
        'api_key': 'your_api_key_here',                # API密钥
        'upload_interval': 60,                         # 数据上传间隔（秒）
        'max_retry_times': 3                           # 最大重试次数
    }
    
    # MQTT配置（物联网通信）
    MQTT_CONFIG = {
        'broker_host': 'localhost',                    # MQTT代理地址
        'broker_port': 1883,                           # MQTT端口
        'topic_sensor_data': 'gas_detection/sensor',   # 传感器数据主题
        'topic_alerts': 'gas_detection/alerts',        # 警报主题
        'client_id': 'gas_detector_001'                # 客户端ID
    }

# 创建必要的目录结构
def create_directories():
    """创建项目所需的目录结构"""
    directories = [
        'data',
        'models', 
        'logs',
        'utils',
        'sensors',
        'ml_models',
        'alert_system',
        'cloud_service'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"创建目录: {directory}")

if __name__ == '__main__':
    create_directories()
    print("配置文件加载完成！")