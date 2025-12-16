import torch.nn as nn
import torch.nn.functional as F
import torch

class ImprovedCNN(nn.Module):
    """改进的CNN模型，添加批量归一化 - 适用于STL-10数据集"""
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()
        
        # 第一组卷积层 + 批量归一化
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 第二组卷积层 + 批量归一化
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 第三组卷积层 + 批量归一化
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 池化层和Dropout
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # 全连接层
        # 输入: 96x96 -> 经过3次池化: 96->48->24->12
        self.fc1 = nn.Linear(128 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 第一层: 卷积 -> 批量归一化 -> ReLU -> 池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 96x96 -> 48x48
        
        # 第二层: 卷积 -> 批量归一化 -> ReLU -> 池化
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 48x48 -> 24x24
        
        # 第三层: 卷积 -> 批量归一化 -> ReLU -> 池化
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # 24x24 -> 12x12
        
        # 展平并连接全连接层
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)  # 展平尺寸: 128 * 12 * 12
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

    def get_feature_maps(self, x):
        """获取中间特征图（用于可视化）"""
        features = []
        
        # 第一层特征
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1_pool = self.pool(x1)
        features.append(('conv1', x1.detach()))
        features.append(('pool1', x1_pool.detach()))
        
        # 第二层特征
        x2 = self.conv2(x1_pool)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2_pool = self.pool(x2)
        features.append(('conv2', x2.detach()))
        features.append(('pool2', x2_pool.detach()))
        
        # 第三层特征
        x3 = self.conv3(x2_pool)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x3_pool = self.pool(x3)
        features.append(('conv3', x3.detach()))
        features.append(('pool3', x3_pool.detach()))
        
        return features


def test_model():
    """测试模型结构 - CPU版本"""
    model = ImprovedCNN()
    print("=" * 60)
    print("改进的CNN模型结构（带批量归一化）")
    print("=" * 60)
    print(model)
    
    # 测试随机输入
    x = torch.randn(2, 3, 96, 96)
    output = model(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n参数统计:")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 各层参数分布
    print(f"\n各层参数分布:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:30} | {str(param.shape):20} | {param.numel():8,} 参数")
    
    # 测试特征图提取
    print(f"\n特征图测试:")
    features = model.get_feature_maps(x)
    for name, feat in features:
        print(f"{name:10} | 形状: {feat.shape}")
    
    return model


def compare_with_original():
    """与原始模型比较"""
    class OriginalCNN(nn.Module):
        """原始CNN模型（用于比较）"""
        def __init__(self, num_classes=10):
            super(OriginalCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(128 * 12 * 12, 512)
            self.fc2 = nn.Linear(512, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.dropout1(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            return x
    
    print("\n" + "=" * 60)
    print("模型对比: 原始 vs 改进")
    print("=" * 60)
    
    original_model = OriginalCNN()
    improved_model = ImprovedCNN()
    
    original_params = sum(p.numel() for p in original_model.parameters())
    improved_params = sum(p.numel() for p in improved_model.parameters())
    
    print(f"原始模型参数数量: {original_params:,}")
    print(f"改进模型参数数量: {improved_params:,}")
    print(f"参数增加: {improved_params - original_params:,} (+{(improved_params/original_params-1)*100:.2f}%)")
    print("\n改进点:")
    print("✓ 添加了批量归一化层 (BatchNorm)")
    print("✓ 添加了权重初始化")
    print("✓ 添加了特征图提取功能")
    print("✓ 训练更稳定，收敛更快")


if __name__ == "__main__":
    # 测试改进模型
    model = test_model()
    
    # 与原始模型比较
    compare_with_original()
    
    print("\n" + "=" * 60)
    print("模型测试完成！")
    print("=" * 60)