# train_optimized_gpu.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from data_loader import get_stl10_dataloaders
from model import STL10_ResNet18
from utils import evaluate_model, plot_optimizer_comparison, print_experiment_summary
import pickle
import time

def calculate_loss(model, data_loader, criterion, device):
    """计算模型在给定数据加载器上的损失"""
    model.eval()
    running_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
    
    avg_loss = running_loss / total_samples
    return avg_loss

def setup_device():
    """设置设备并优化GPU性能"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        
        # GPU性能优化设置
        torch.backends.cudnn.benchmark = True  # 加速卷积运算
        torch.backends.cudnn.deterministic = False  # 为了速度牺牲可重复性
        
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"✅ 使用GPU: {gpu_props.name}")
        print(f"   GPU内存: {gpu_props.total_memory / 1024**3:.1f} GB")
        print(f"   CUDA核心: {gpu_props.multi_processor_count}")
        
    else:
        device = torch.device('cpu')
        print("⚠️ 使用CPU (GPU不可用)")
    
    return device

def train_single_experiment(optimizer_name='adam', learning_rate=0.001, batch_size=64, epochs=15, 
                           use_amp=True, model_type='resnet18', validation_split=0.1):
    """优化的GPU训练实验"""
    
    # 设置设备
    device = setup_device()
    
    # 获取数据 - 包含训练集、验证集和测试集
    train_loader, val_loader, test_loader = get_stl10_dataloaders(
        batch_size=batch_size, 
        use_resnet_preprocessing=True,
        validation_split=validation_split
    )
    
    # 选择模型
    if model_type == 'resnet18':
        model = STL10_ResNet18(pretrained=True, feature_extract=False)
        print("使用ResNet18模型 (完整微调)")
    else:
        from model import SimpleCNN
        model = SimpleCNN()
        print("使用简单CNN模型")
    
    # 将模型移动到设备
    model = model.to(device)
    print(f"模型已移动到: {device}")
    
    # 设置混合精度训练 (AMP) - 大幅提升训练速度并减少显存使用
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
   # 优化器配置 - 增加权重衰减（更强的L2正则化）
    weight_decay = 1e-3
    
    if optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(), 
            lr=learning_rate, 
            alpha=0.99,
            momentum=0.9,
            weight_decay=weight_decay,
            eps=1e-8
        )
    
    # 使用余弦退火学习率调度器 - 更适合长训练
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    
    print(f"\n开始训练 {optimizer_name} 优化器...")
    print(f"模型: {model_type}")
    print(f"混合精度训练: {use_amp}")
    print(f"设备: {device}")
    print(f"批次大小: {batch_size}")
    print(f"训练轮数: {epochs}")
    print(f"学习率: {learning_rate}")
    print(f"验证集比例: {validation_split}")
    print(f"数据集: STL-10")
    
    # 记录训练过程
    train_losses = []
    val_accuracies = []
    val_losses = []  # 新增：验证集loss
    test_losses = []  # 新增：测试集loss（定期计算）
    training_times = []
    
    # 早停参数
    # best_val_accuracy = 0
    # patience = 5
    # patience_counter = 0
    # best_model_state = None

    # 在训练循环开始前添加热身
    warmup_epochs = 3
    for epoch in range(epochs):
        # 学习率热身
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * lr_scale
        
    # 训练循环
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0

        # ========== Warmup学习率调整 ==========
        if epoch < warmup_epochs:
            # 线性增加学习率: 从0.1*lr增加到目标lr
            warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # 将数据移动到设备
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # 混合精度训练
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
            # 每50个batch打印一次进度
            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | LR: {current_lr:.6f}')
        
        # 计算平均损失
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 更新学习率
        if epoch >= warmup_epochs:
            scheduler.step()
        
        # 每个epoch结束后在验证集上评估
        accuracy, _ = evaluate_model(model, val_loader, device)
        val_accuracies.append(accuracy)

        # 新增：计算验证集loss
        val_loss = calculate_loss(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # 每5个epoch计算一次测试集loss
        if epoch % 5 == 0 or epoch == epochs - 1:
            test_loss = calculate_loss(model, test_loader, criterion, device)
            test_losses.append({'epoch': epoch, 'loss': test_loss})
        
        epoch_time = time.time() - epoch_start_time
        training_times.append(epoch_time)
        
        warmup_status = "[Warmup]" if epoch < warmup_epochs else ""
        print(f'Epoch [{epoch+1}/{epochs}] | 时间: {epoch_time:.1f}s | '
              f'平均损失: {avg_loss:.4f} | 验证准确率: {accuracy:.2f}% {warmup_status}')
        # # 早停逻辑
        # if accuracy > best_val_accuracy:
        #     best_val_accuracy = accuracy
        #     patience_counter = 0
        #     # 保存最佳模型状态
        #     best_model_state = model.state_dict().copy()
        #     print(f"✅ 保存最佳模型，验证准确率: {accuracy:.2f}%")
        # else:
        #     patience_counter += 1
        #     print(f"⚠️  验证准确率未提升，耐心计数: {patience_counter}/{patience}")
            
        # if patience_counter >= patience:
        #     print(f"🛑 早停在第 {epoch+1} 轮")
        #     # 恢复最佳模型
        #     if best_model_state is not None:
        #         model.load_state_dict(best_model_state)
        #     break
    
    # 最终在测试集上评估
    final_accuracy, class_accuracy = evaluate_model(model, test_loader, device)
    final_test_loss = calculate_loss(model, test_loader, criterion, device)
    avg_epoch_time = sum(training_times) / len(training_times)
    print(f"平均每轮训练时间: {avg_epoch_time:.1f}秒")
    print(f"最终测试集准确率: {final_accuracy:.2f}%")
    print(f"最终测试集损失: {final_test_loss:.4f}")

    return {
        'model': model,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'val_losses': val_losses,  # 新增
        'test_losses': test_losses,  # 新增
        'final_accuracy': final_accuracy,
        'class_accuracy': class_accuracy,
        'training_times': training_times,
        'final_test_loss': final_test_loss  # 新增
        # 'best_val_accuracy': best_val_accuracy
    }

def main():
    """主函数，运行完整的30个epoch训练"""
    
    print("STL-10 图像分类实验 - 100个epoch完整训练")
    print("Python 3.9 + PyTorch 2.7.1 + CUDA 12.8")
    print("="*60)
    
    # 实验配置 - 30个epoch，更强的正则化
    experiments = [
    {'name': 'ResNet18_Adam', 'optimizer': 'adam', 'lr': 0.0001, 'batch_size': 128, 'epochs': 30, 'val_split': 0.1},
    {'name': 'ResNet18_AdamW', 'optimizer': 'adamw', 'lr': 0.0001, 'batch_size': 128, 'epochs': 30, 'val_split': 0.1},
    {'name': 'ResNet18_SGD', 'optimizer': 'sgd', 'lr': 0.001, 'batch_size': 128, 'epochs': 30, 'val_split': 0.1},  # 从0.01降到0.001
    {'name': 'RMSprop_VeryLowLR_NoMomentum', 'optimizer': 'rmsprop', 'lr': 0.00002, 'batch_size': 128, 'epochs': 20, 'val_split': 0.1, 'momentum': 0.0},
    {'name': 'RMSprop_LowLR_NoMomentum', 'optimizer': 'rmsprop', 'lr': 0.00005, 'batch_size': 128, 'epochs': 20, 'val_split': 0.1, 'momentum': 0.0},
]
    
    results = {}
    
    for exp_config in experiments:
        print(f"\n{'='*50}")
        print(f"实验: {exp_config['name']}")
        print(f"{'='*50}")
        
        result = train_single_experiment(
            optimizer_name=exp_config['optimizer'],
            learning_rate=exp_config['lr'],
            batch_size=exp_config['batch_size'],
            epochs=exp_config['epochs'],
            use_amp=True,  # 启用混合精度训练
            model_type='resnet18',
            validation_split=exp_config['val_split']
        )
        
        results[exp_config['name']] = result
        
        # 打印最终结果
        print(f"\n📊 {exp_config['name']} 最终测试准确率: {result['final_accuracy']:.2f}%")
        # print(f"🏆 {exp_config['name']} 最佳验证准确率: {result['best_val_accuracy']:.2f}%")
    
    # 绘制训练历史
    plot_optimizer_comparison(results)
    
    # 比较不同配置的性能
    print_experiment_summary(results)
    
    # 保存结果
    with open('optimized_gpu_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("训练结果已保存到 optimized_gpu_results.pkl")
    
    # 性能统计
    total_training_time = sum(sum(result['training_times']) for result in results.values())
    print(f"\n总训练时间: {total_training_time:.1f}秒 ({total_training_time/60:.1f}分钟)")
    
    # 找出最佳模型
    best_exp = max(results.items(), key=lambda x: x[1]['final_accuracy'])
    print(f"\n🎉 最佳模型: {best_exp[0]} - 测试准确率: {best_exp[1]['final_accuracy']:.2f}%")
    
    return results

if __name__ == "__main__":
    results = main()