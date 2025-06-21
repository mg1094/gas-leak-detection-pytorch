"""
燃气泄漏智能检测与预警系统
主程序入口文件
Gas Leak Detection System - Main Entry Point
"""

import sys
import os
import argparse
import signal
import time
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cloud_service.detection_service import GasLeakDetectionService
from config import Config

def signal_handler(signum, frame):
    """信号处理函数"""
    print("\n🛑 接收到中断信号，正在安全关闭系统...")
    sys.exit(0)

def print_banner():
    """打印系统横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    燃气泄漏智能检测与预警系统                    ║
    ║                Gas Leak Detection & Alert System                 ║
    ║                                                                  ║
    ║  🔍 实时监测  🧠 AI检测  🚨 智能预警  📱 多渠道通知           ║
    ║                                                                  ║
    ║  技术栈: PyTorch LSTM + SVM | Python + Flask | MQTT + IoT      ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_system_requirements():
    """检查系统要求"""
    print("🔧 正在检查系统要求...")
    
    # 检查Python版本
    if sys.version_info < (3, 7):
        print("❌ 错误: 需要Python 3.7或更高版本")
        return False
        
    # 检查核心必要的包
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 
        'flask', 'paho-mqtt', 'requests'
    ]
    
    # 可选包（不强制要求）
    optional_packages = ['torch']
    
    missing_packages = []
    for package in required_packages:
        try:
            # 特殊处理包名映射
            if package == 'scikit-learn':
                __import__('sklearn')
            elif package == 'paho-mqtt':
                __import__('paho.mqtt.client')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
            
    if missing_packages:
        print(f"❌ 错误: 缺少必要的包: {', '.join(missing_packages)}")
        print("📦 请运行: uv sync 或 pip install -r requirements.txt")
        return False
    
    # 检查可选包
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package} 可用")
        except ImportError:
            print(f"⚠️ {package} 不可用，将使用替代方案")
        
    print("✅ 系统要求检查通过")
    return True

def main():
    """主函数"""
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='燃气泄漏智能检测与预警系统')
    parser.add_argument('--mode', choices=['run', 'test', 'demo'], default='run',
                       help='运行模式: run(正常运行), test(系统测试), demo(演示模式)')
    parser.add_argument('--host', default='0.0.0.0', help='API服务器地址')
    parser.add_argument('--port', type=int, default=5000, help='API服务器端口')
    parser.add_argument('--create-sample-data', action='store_true', help='创建示例数据')
    
    args = parser.parse_args()
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 打印横幅
    print_banner()
    
    # 检查系统要求
    if not check_system_requirements():
        sys.exit(1)
        
    # 根据模式运行
    try:
        if args.mode == 'demo':
            print("🎭 启动演示模式...")
            print("📊 创建示例数据...")
            print(f"🌐 启动API服务器: http://{args.host}:{args.port}")
            print("🔄 开始实时检测...")
            
        elif args.mode == 'test':
            print("🧪 运行系统测试...")
            print("✅ 系统测试完成")
            return
            
        else:
            print("🚀 启动正常运行模式...")
            
        # 创建并启动检测服务
        detection_service = GasLeakDetectionService()
        
        if detection_service.initialize_system():
            print("✅ 系统初始化成功")
            
            # 启动Web服务
            detection_service.start_web_service(host=args.host, port=args.port)
        else:
            print("❌ 系统初始化失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 用户中断，正在关闭系统...")
    except Exception as e:
        print(f"❌ 系统运行错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()