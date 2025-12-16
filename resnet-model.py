# model.py
import torch
import torch.nn as nn
import torchvision.models as models

class STL10_ResNet18(nn.Module):
    """基于ResNet18的STL-10分类模型"""
    
    def __init__(self, pretrained=True, feature_extract=False, num_classes=10):
        super(STL10_ResNet18, self).__init__()
        
        # 加载预训练的ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        if feature_extract:
            # 特征提取模式：冻结所有层
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # 替换最后的全连接层以适应STL-10的10个类别
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        # 打印模型信息
        if pretrained:
            print("使用预训练的ResNet18模型")
        else:
            print("使用随机初始化的ResNet18模型")
            
        if feature_extract:
            print("特征提取模式：冻结卷积层")
        else:
            print("微调模式：所有层都可训练")
    
    def forward(self, x):
        return self.resnet(x)

class SimpleCNN(nn.Module):
    """简单的CNN模型用于STL-10分类"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # 输入: 3x96x96
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 12 * 12, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 测试代码
if __name__ == "__main__":
    # 测试ResNet18模型
    model_resnet = STL10_ResNet18(pretrained=True, feature_extract=False)
    print(f"ResNet18参数数量: {count_parameters(model_resnet):,}")
    
    # 测试简单CNN模型
    model_cnn = SimpleCNN()
    print(f"SimpleCNN参数数量: {count_parameters(model_cnn):,}")
    
    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)
    output = model_resnet(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")