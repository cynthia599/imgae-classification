import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import platform
import os
import numpy as np

# 自定义数据集类，用于 Mean Teacher 的无标签数据
class MeanTeacherUnlabeledDataset(Dataset):
    """
    Mean Teacher 专用的无标签数据集
    为每个样本提供两种不同增强：弱增强（教师）和强增强（学生）
    """
    
    def __init__(self, stl10_dataset, weak_transform, strong_transform):
        """
        参数:
            stl10_dataset: STL-10 无标签数据集
            weak_transform: 弱增强变换（用于教师模型）
            strong_transform: 强增强变换（用于学生模型）
        """
        self.dataset = stl10_dataset
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        
        # 检查数据集是否已经应用了转换
        self._check_transform()
    
    def _check_transform(self):
        """检查数据集是否已经应用了转换"""
        sample, _ = self.dataset[0]
        if isinstance(sample, torch.Tensor):
            print("⚠️  警告: 无标签数据集已经返回Tensor，可能需要修改变换")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # STL-10 无标签数据返回 (image, -1)
        image, _ = self.dataset[idx]
        
        # 确保图像是PIL Image，而不是Tensor
        if isinstance(image, torch.Tensor):
            # 如果是Tensor，先转换为PIL Image
            from torchvision.transforms.functional import to_pil_image
            image = to_pil_image(image)
        
        # 应用两种增强
        weak_aug = self.weak_transform(image)
        strong_aug = self.strong_transform(image)
        
        # 返回两种增强版本
        return (weak_aug, strong_aug), -1  # -1 表示无标签
    
class STL10DataLoader:
    """
    STL-10 数据加载器类，支持 Mean Teacher 半监督学习
    """
    
    def __init__(self, data_dir='./data', batch_size=32, validation_split=0.1):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        # 自动设置 num_workers
        if platform.system() == 'Windows':
            self.num_workers = 0
        else:
            self.num_workers = 4
        
        # 设置环境变量
        os.environ['OMP_NUM_THREADS'] = '1'
        
        # 定义变换
        self._define_transforms()
    
    def _define_transforms(self):
        """定义各种数据变换"""
        
        # 1. 基础变换（测试集使用）
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet要求224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 2. 弱增强变换（用于教师模型和有标签训练）
        self.weak_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 3. 强增强变换（用于学生模型）
        self.strong_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, 
                                 saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1))
        ])
        
        # 4. 有标签训练增强（中等强度）
        self.labeled_train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(224, padding=8, padding_mode='reflect'),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                 saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _check_dataset_sizes(self):
        """检查数据集大小是否正确"""
        print("\n📊 STL-10 数据集信息:")
        print("-" * 40)
        
        # 检查有标签训练集
        try:
            train_dataset_temp = datasets.STL10(
                root=self.data_dir,
                split='train',
                download=False,
                transform=None
            )
            print(f"有标签训练集: {len(train_dataset_temp)} 张图片")
        except:
            print("有标签训练集: 未下载或路径错误")
        
        # 检查无标签数据
        try:
            unlabeled_dataset_temp = datasets.STL10(
                root=self.data_dir,
                split='unlabeled',
                download=False,
                transform=None
            )
            print(f"无标签数据集: {len(unlabeled_dataset_temp)} 张图片")
        except:
            print("无标签数据集: 未下载或路径错误")
        
        # 检查测试集
        try:
            test_dataset_temp = datasets.STL10(
                root=self.data_dir,
                split='test',
                download=False,
                transform=None
            )
            print(f"测试集: {len(test_dataset_temp)} 张图片")
        except:
            print("测试集: 未下载或路径错误")
        
        print("-" * 40)
    
    def get_mean_teacher_dataloaders(self, include_unlabeled=True):
        """
        为 Mean Teacher 训练获取数据加载器
        
        返回:
            labeled_train_loader: 有标签训练数据
            val_loader: 验证数据
            test_loader: 测试数据
            unlabeled_loader: 无标签数据（Mean Teacher 格式）
        """
        
        self._check_dataset_sizes()
        
        print("\n🚀 准备 Mean Teacher 数据加载器...")
        
        # 1. 有标签训练集（使用弱增强）
        print("加载有标签训练集...")
        labeled_train_dataset = datasets.STL10(
            root=self.data_dir,
            split='train',
            download=True,
            transform=self.labeled_train_transform
        )
        
        # 验证集从有标签数据中划分
        val_size = int(self.validation_split * len(labeled_train_dataset))
        train_size = len(labeled_train_dataset) - val_size
        
        print(f"有标签数据划分: {train_size} 训练, {val_size} 验证")
        
        # 随机划分
        indices = torch.randperm(len(labeled_train_dataset)).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        
        train_subset = Subset(labeled_train_dataset, train_indices)
        val_subset = Subset(labeled_train_dataset, val_indices)
        
        # 2. 测试集（使用基础变换）
        print("加载测试集...")
        test_dataset = datasets.STL10(
            root=self.data_dir,
            split='test',
            download=True,
            transform=self.base_transform
        )
        
        # 3. 无标签数据集（Mean Teacher 格式）
        unlabeled_loader = None
        if include_unlabeled:
            print("加载无标签数据集（Mean Teacher 格式）...")
            
            # 加载原始无标签数据
            raw_unlabeled_dataset = datasets.STL10(
                root=self.data_dir,
                split='unlabeled',
                download=True,
                transform=transforms.ToTensor()  # 只转换为tensor，后面应用增强
            )
            
            # 创建 Mean Teacher 专用数据集
            mt_unlabeled_dataset = MeanTeacherUnlabeledDataset(
                raw_unlabeled_dataset,
                self.weak_transform,
                self.strong_transform
            )
            
            print(f"无标签数据: {len(mt_unlabeled_dataset)} 张图片")
            print("  每个样本提供: 弱增强（教师） + 强增强（学生）")
        
        # 创建数据加载器
        print("\n创建数据加载器...")
        labeled_train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True  # Mean Teacher 中建议 drop_last
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        if include_unlabeled:
            unlabeled_loader = DataLoader(
                mt_unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True
            )
        
        # 打印统计信息
        print("\n✅ 数据加载器创建完成:")
        print(f"   有标签训练集: {len(labeled_train_loader)} 批次 × {self.batch_size}")
        print(f"   验证集: {len(val_loader)} 批次")
        print(f"   测试集: {len(test_loader)} 批次")
        if unlabeled_loader:
            print(f"   无标签数据: {len(unlabeled_loader)} 批次 × {self.batch_size}")
        
        return labeled_train_loader, val_loader, test_loader, unlabeled_loader
    
    def get_class_distribution(self):
        """获取每个类别的样本数量分布"""
        
        # 加载有标签训练集
        train_dataset = datasets.STL10(
            root=self.data_dir,
            split='train',
            download=True,
            transform=None
        )
        
        # STL-10 类别名称
        class_names = ['airplane', 'bird', 'car', 'cat', 'deer',
                      'dog', 'horse', 'monkey', 'ship', 'truck']
        
        # 统计每个类别的数量
        class_counts = {name: 0 for name in class_names}
        
        for _, label in train_dataset:
            class_counts[class_names[label]] += 1
        
        print("\n📈 类别分布统计:")
        print("-" * 40)
        for class_name, count in class_counts.items():
            percentage = (count / len(train_dataset)) * 100
            print(f"{class_name:10s}: {count:4d} 张 ({percentage:5.1f}%)")
        
        return class_counts


def visualize_augmentations(data_loader, num_samples=3):
    """
    可视化数据增强效果
    """
    import matplotlib.pyplot as plt
    
    # 获取一个批次
    for batch in data_loader:
        if isinstance(batch, tuple) and len(batch) == 2:
            (weak_augs, strong_augs), _ = batch
            break
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 4))
    
    for i in range(num_samples):
        # 转换为可显示的格式
        weak_img = weak_augs[i].cpu().numpy().transpose(1, 2, 0)
        weak_img = weak_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        weak_img = np.clip(weak_img, 0, 1)
        
        strong_img = strong_augs[i].cpu().numpy().transpose(1, 2, 0)
        strong_img = strong_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        strong_img = np.clip(strong_img, 0, 1)
        
        # 显示图像
        axes[i, 0].imshow(weak_img)
        axes[i, 0].set_title(f"样本 {i+1}: 弱增强")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(strong_img)
        axes[i, 1].set_title(f"样本 {i+1}: 强增强")
        axes[i, 1].axis('off')
    
    plt.suptitle("Mean Teacher 数据增强对比", fontsize=16)
    plt.tight_layout()
    plt.savefig('mean_teacher_augmentations.png', dpi=150, bbox_inches='tight')
    plt.show()


# 向后兼容的函数
def get_stl10_dataloaders(batch_size=32, use_resnet_preprocessing=True, 
                         validation_split=0.1, include_unlabeled=True, 
                         num_workers=None):
    """
    向后兼容的函数，使用新的数据加载器
    """
    # 如果指定了num_workers，则使用指定的值
    if num_workers is not None:
        import warnings
        warnings.warn("num_workers参数将在新版本中被忽略，请使用STL10DataLoader类")
    
    # 创建数据加载器实例
    loader = STL10DataLoader(
        data_dir='./data',
        batch_size=batch_size,
        validation_split=validation_split
    )
    
    return loader.get_mean_teacher_dataloaders(include_unlabeled=include_unlabeled)


# 测试代码
if __name__ == "__main__":
    print("🧪 测试 Mean Teacher 数据加载器...")
    
    # 创建数据加载器
    loader = STL10DataLoader(
        data_dir='./data',
        batch_size=16,  # 使用小批次便于测试
        validation_split=0.1
    )
    
    # 获取类别分布
    class_dist = loader.get_class_distribution()
    
    # 获取数据加载器
    train_loader, val_loader, test_loader, unlabeled_loader = loader.get_mean_teacher_dataloaders()
    
    # 检查数据形状
    print("\n🔍 检查数据形状:")
    
    # 有标签数据
    for images, labels in train_loader:
        print(f"有标签训练数据形状: {images.shape}")
        print(f"有标签训练标签形状: {labels.shape}")
        print(f"标签值示例: {labels[:5].numpy()}")
        break
    
    # 无标签数据
    if unlabeled_loader:
        for (weak_augs, strong_augs), _ in unlabeled_loader:
            print(f"无标签弱增强形状: {weak_augs.shape}")
            print(f"无标签强增强形状: {strong_augs.shape}")
            break
    
    # 验证数据
    for images, labels in val_loader:
        print(f"验证数据形状: {images.shape}")
        print(f"验证标签形状: {labels.shape}")
        break
    
    print("\n✅ 数据加载器测试完成!")
    
    # 可视化增强效果（可选）
    # if unlabeled_loader:
    #     visualize_augmentations(unlabeled_loader)