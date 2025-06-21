# 训练数据量和训练轮数使用指南

## 🚀 快速开始

本项目包含完整的训练数据量和训练轮数分析工具，帮助您优化燃气泄漏检测系统的训练配置。

### 📁 文件说明

- `training_data_analysis.md` - 详细的训练数据量和轮数分析文档
- `training_config_optimizer.py` - 训练配置优化器，自动推荐最佳参数
- `training_effect_comparison.py` - 训练效果对比示例，可视化不同配置的性能
- `optimized_config.json` - 基于当前数据的优化配置文件

### 🔧 使用方法

#### 1. 分析现有数据
```bash
python training_config_optimizer.py --analyze-only
```

#### 2. 生成优化配置
```bash
python training_config_optimizer.py
```

#### 3. 对比不同配置效果
```bash
python training_effect_comparison.py
```

### 📊 推荐配置

基于当前测试数据（2,052个样本），推荐配置：

```python
{
  "epochs": 150,
  "batch_size": 64,
  "learning_rate": 0.01,
  "training_mode": "快速验证",
  "estimated_time": "5分钟"
}
```

### 💡 使用建议

1. **初期开发**: 使用小数据集配置（150 epochs, batch_size=64）
2. **功能验证**: 扩展到中等数据集配置（100 epochs, batch_size=32）
3. **生产部署**: 使用大数据集配置（80 epochs, batch_size=16）

### 📈 性能目标

| 阶段 | 数据量 | 训练轮数 | 预期准确率 | 训练时间 |
|------|--------|----------|------------|----------|
| MVP | 7天 | 50 | >85% | 3-5小时 |
| 验证 | 30天 | 100 | >92% | 8-12小时 |
| 生产 | 90天 | 150 | >95% | 1-2天 |

### ⚠️ 注意事项

- 数据质量比数量更重要
- 确保场景覆盖的多样性
- 使用早停机制避免过拟合
- 定期监控模型性能

更多详细信息请参考 `training_data_analysis.md` 文档。