import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_stl10_dataloaders(batch_size=32, image_size=96, validation_split=0.2):
    """
    创建STL-10数据加载器，包含训练集、验证集和测试集
    
    Args:
        batch_size: 批次大小
        image_size: 图像尺寸（STL-10原始尺寸是96x96）
        validation_split: 验证集比例（从训练集中划分）
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    # 训练数据预处理 - 包含数据增强
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomCrop(image_size, padding=4),  # 随机裁剪
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 验证和测试数据预处理 - 不包含数据增强
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 下载训练集
    full_train_dataset = datasets.STL10(
        root='./data', 
        split='train',
        download=True, 
        transform=train_transform
    )
    
    # 划分训练集和验证集
    train_size = int((1 - validation_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # 为验证集应用测试时的预处理
    val_dataset.dataset.transform = test_transform

    
    # 下载测试集
    test_dataset = datasets.STL10(
        root='./data', 
        split='test',
        download=True, 
        transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # 打印数据集信息
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"训练/验证/测试比例: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
    print(f"总样本数: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    print(f"图像形状: {train_dataset[0][0].shape}")
    
    return train_loader, val_loader, test_loader

def get_stl10_class_names():
    """返回STL-10数据集的类别名称"""  
    return ['airplane', 'bird', 'car', 'cat', 'deer', 
            'dog', 'horse', 'monkey', 'ship', 'truck']

# 测试代码
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_stl10_dataloaders(batch_size=32)
    class_names = get_stl10_class_names()
    
    print("\nSTL-10数据加载器创建成功！")
    print(f"类别数量: {len(class_names)}")
    print(f"类别名称: {class_names}")
    
    # 测试一个批次
    for images, labels in train_loader:
        print(f"\n训练集 - 图像批次形状: {images.shape}")  # torch.Size([32, 3, 96, 96])
        print(f"训练集 - 标签批次形状: {labels.shape}")    # torch.Size([32])
        print(f"标签值范围: {labels.min()} ~ {labels.max()}")
        break
    
    for images, labels in val_loader:
        print(f"\n验证集 - 图像批次形状: {images.shape}")
        print(f"验证集 - 标签批次形状: {labels.shape}")
        break
    
    for images, labels in test_loader:
        print(f"\n测试集 - 图像批次形状: {images.shape}")
        print(f"测试集 - 标签批次形状: {labels.shape}")
        break