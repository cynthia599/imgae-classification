"""
STL-10 Mean Teacher 半监督学习实验
包含完整的类别分析和可视化功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import platform
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_loader import get_stl10_dataloaders
from model import STL10_ResNet18
from utils import evaluate_model, plot_mean_teacher_results, print_experiment_summary, check_overfitting, calculate_loss
import pickle
import time
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from PIL import Image

# ========== 新增分析函数 ==========

def analyze_class_performance(model, test_loader, device, class_names=None):
    """
    分析模型在每个类别上的性能
    """
    model.eval()
    
    # 如果没有提供类别名称，使用STL-10默认
    if class_names is None:
        class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 
                      'dog', 'horse', 'monkey', 'ship', 'truck']
    
    num_classes = len(class_names)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # 统计每个类别的正确预测数
            c = (predicted == labels).squeeze()
            
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # 计算每个类别的准确率
    class_accuracies = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            class_accuracies[class_names[i]] = accuracy
    
    # 找出最佳和最差类别
    sorted_classes = sorted(class_accuracies.items(), 
                           key=lambda x: x[1], 
                           reverse=True)
    
    best_class = sorted_classes[0] if sorted_classes else ("None", 0)
    worst_class = sorted_classes[-1] if sorted_classes else ("None", 0)
    
    return class_accuracies, best_class, worst_class, None

def analyze_unlabeled_data_performance(teacher_model, unlabeled_loader, device, 
                                      confidence_threshold=0.8, num_samples=1000):
    """
    分析模型在无标签数据上的预测置信度
    """
    teacher_model.eval()
    
    confidence_scores = []
    predicted_classes = []
    sample_predictions = []
    
    # 只分析部分样本，避免内存问题
    num_analyzed = 0
    
    with torch.no_grad():
        for batch in unlabeled_loader:
            if num_analyzed >= num_samples:
                break
            
            # 根据您的数据结构，batch是列表：[images_list, labels]
            if isinstance(batch, list) and len(batch) == 2:
                images_list = batch[0]
                if isinstance(images_list, list) and len(images_list) >= 1:
                    # 使用弱增强图像进行预测
                    images = images_list[0].to(device)
                    outputs = teacher_model(images)
                    probabilities = torch.softmax(outputs, dim=1)
                    
                    # 获取每个样本的最大概率和预测类别
                    max_probs, preds = torch.max(probabilities, dim=1)
                    
                    confidence_scores.extend(max_probs.cpu().numpy())
                    predicted_classes.extend(preds.cpu().numpy())
                    
                    # 存储高置信度样本的预测
                    for i in range(len(images)):
                        if num_analyzed >= num_samples:
                            break
                        
                        sample_predictions.append({
                            'predicted_class': preds[i].item(),
                            'confidence': max_probs[i].item(),
                            'probabilities': probabilities[i].cpu().numpy()
                        })
                        num_analyzed += 1
    
    # 统计每个类别的预测次数和平均置信度
    class_stats = {}
    class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 
                  'dog', 'horse', 'monkey', 'ship', 'truck']
    
    for class_idx in range(10):
        # 获取属于该类别的所有预测
        class_mask = [p['predicted_class'] == class_idx for p in sample_predictions]
        
        if any(class_mask):
            class_confidences = [p['confidence'] for p, mask in zip(sample_predictions, class_mask) if mask]
            class_stats[class_names[class_idx]] = {
                'count': sum(class_mask),
                'avg_confidence': np.mean(class_confidences) if class_confidences else 0,
                'std_confidence': np.std(class_confidences) if class_confidences else 0,
                'high_confidence_count': sum(c >= confidence_threshold for c in class_confidences)
            }
        else:
            class_stats[class_names[class_idx]] = {
                'count': 0,
                'avg_confidence': 0,
                'std_confidence': 0,
                'high_confidence_count': 0
            }
    
    # 找出模型最"自信"和最"不自信"的类别
    if class_stats:
        sorted_by_confidence = sorted(class_stats.items(), 
                                     key=lambda x: x[1]['avg_confidence'], 
                                     reverse=True)
        most_confident = sorted_by_confidence[0][0] if sorted_by_confidence else "None"
        least_confident = sorted_by_confidence[-1][0] if sorted_by_confidence else "None"
        
        sorted_by_count = sorted(class_stats.items(), 
                                key=lambda x: x[1]['count'], 
                                reverse=True)
        most_predicted = sorted_by_count[0][0] if sorted_by_count else "None"
        least_predicted = sorted_by_count[-1][0] if sorted_by_count else "None"
    else:
        most_confident = least_confident = most_predicted = least_predicted = "None"
    
    return {
        'class_stats': class_stats,
        'most_confident_class': most_confident,
        'least_confident_class': least_confident,
        'most_predicted_class': most_predicted,
        'least_predicted_class': least_predicted,
        'overall_avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
        'high_confidence_ratio': sum(c >= confidence_threshold for c in confidence_scores) / len(confidence_scores) if confidence_scores else 0
    }

def plot_confusion_matrix_and_analysis(teacher_model, test_loader, device, 
                                      experiment_name="MeanTeacher"):
    """
    绘制混淆矩阵并进行详细分析
    """
    teacher_model.eval()
    
    all_preds = []
    all_labels = []
    class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 
                  'dog', 'horse', 'monkey', 'ship', 'truck']
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = teacher_model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    
    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {experiment_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{experiment_name}.png', dpi=300)
    plt.close()
    
    # 分析混淆矩阵
    analysis_results = {}
    
    for i in range(10):
        # 计算每个类别的精度和召回率
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        analysis_results[class_names[i]] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positive': tp,
            'false_positive': fp,
            'false_negative': fn
        }
    
    # 找出最容易混淆的类别对
    confusion_pairs = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0.05:  # 误分类率大于5%
                confusion_pairs.append({
                    'from': class_names[i],
                    'to': class_names[j],
                    'rate': cm[i, j]
                })
    
    confusion_pairs.sort(key=lambda x: x['rate'], reverse=True)
    
    # 找出最佳和最差F1分数
    if analysis_results:
        best_class = max(analysis_results.items(), key=lambda x: x[1]['f1_score'])[0]
        worst_class = min(analysis_results.items(), key=lambda x: x[1]['f1_score'])[0]
    else:
        best_class = worst_class = "None"
    
    return {
        'confusion_matrix': cm,
        'class_analysis': analysis_results,
        'top_confusion_pairs': confusion_pairs[:5] if confusion_pairs else [],
        'best_class': best_class,
        'worst_class': worst_class
    }

# ========== 原有函数 ==========

def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def log_gradient_info(model, optimizer, epoch, batch_idx, log_freq=50):
    """记录梯度信息"""
    if batch_idx % log_freq != 0:
        return
    
    total_norm = 0.0
    max_grad = -float('inf')
    min_grad = float('inf')
    num_params = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            
            param_max = param.grad.data.max().item()
            param_min = param.grad.data.min().item()
            max_grad = max(max_grad, param_max)
            min_grad = min(min_grad, param_min)
            
            num_params += 1
    
    if num_params > 0:
        total_norm = total_norm ** 0.5
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}, Batch {batch_idx}: "
              f"梯度范数={total_norm:.4f}, LR={current_lr:.6f}, "
              f"最大梯度={max_grad:.6f}, 最小梯度={min_grad:.6f}")
        
        # 梯度异常警告
        if total_norm > 100:
            print("⚠️  警告: 梯度可能爆炸 (范数 > 100)")
        elif total_norm < 0.0001:
            print("⚠️  警告: 梯度可能消失 (范数 < 0.0001)")

def train_mean_teacher(optimizer_name='adam', learning_rate=0.001, batch_size=64, epochs=50,
                      use_amp=True, model_type='resnet18', validation_split=0.1,
                      consistency_weight=10.0, ema_decay=0.999, warmup_epochs=10,
                      use_kl_loss=False, max_grad_norm=1.0, save_best=True,
                      experiment_name="MeanTeacher", num_workers=None):
    """Mean Teacher半监督训练"""
    
    # 设置随机种子
    set_seed(42)
    
    device = setup_device()
    
    # 获取数据（包含无标签数据），明确指定num_workers
    train_loader, val_loader, test_loader, unlabeled_loader = get_stl10_dataloaders(
        batch_size=batch_size,
        use_resnet_preprocessing=True,
        validation_split=validation_split,
        include_unlabeled=True,
        num_workers=num_workers  # 传递参数
    )
    
    # 初始化学生模型和教师模型
    print("初始化学生模型和教师模型...")
    student_model = STL10_ResNet18(pretrained=True, feature_extract=False)
    teacher_model = STL10_ResNet18(pretrained=True, feature_extract=False)
    
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    
    # 初始化教师模型权重为学生模型
    teacher_model.load_state_dict(student_model.state_dict())
    
    # 确保教师模型不计算梯度
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()  # 教师模型始终在eval模式
    
    # 优化器只更新学生模型
    weight_decay = 1e-4
    
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        # AdamW需要不同的权重衰减
        optimizer = optim.AdamW(
            student_model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.01,  # 增加权重衰减
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(student_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = optim.RMSprop(
            student_model.parameters(), 
            lr=learning_rate, 
            alpha=0.99,
            momentum=0.0,
            weight_decay=weight_decay,
            eps=1e-8
        )
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    # 学习率调度器（余弦退火）
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # 损失函数
    supervised_criterion = nn.CrossEntropyLoss()
    
    # 一致性损失选择
    if use_kl_loss:
        # 使用对称KL散度
        consistency_criterion = nn.KLDivLoss(reduction='batchmean')
        print("使用KL散度作为一致性损失")
    else:
        # 使用均方误差
        consistency_criterion = nn.MSELoss()
        print("使用MSE作为一致性损失")
    
    print(f"\n开始Mean Teacher训练...")
    print(f"实验名称: {experiment_name}")
    print(f"优化器: {optimizer_name}")
    print(f"学习率: {learning_rate}")
    print(f"一致性权重: {consistency_weight}")
    print(f"EMA衰减率: {ema_decay}")
    print(f"热身epochs: {warmup_epochs}")
    print(f"设备: {device}")
    print(f"批次大小: {batch_size}")
    print(f"训练轮数: {epochs}")
    print(f"梯度裁剪: {max_grad_norm}")
    
    # 训练记录
    train_losses = []
    val_accuracies = []
    supervised_losses = []
    consistency_losses = []
    training_times = []
    learning_rates = []
    gradient_norms = []  # 记录梯度范数
    
    # 用于保存最佳模型
    best_val_accuracy = 0.0
    best_model_state = None
    
    # 训练循环
    for epoch in range(epochs):
        epoch_start_time = time.time()
        student_model.train()
        
        running_supervised_loss = 0.0
        running_consistency_loss = 0.0
        running_total_loss = 0.0
        
        # 计算当前epoch的一致性权重（热身阶段线性增加）
        if epoch < warmup_epochs:
            current_consistency_weight = consistency_weight * ((epoch + 1) / warmup_epochs) ** 2
        else:
            current_consistency_weight = consistency_weight
        
        # 准备无标签数据迭代器
        if unlabeled_loader:
            unlabeled_iter = iter(unlabeled_loader)
        else:
            unlabeled_iter = None
        
        # 训练批次
        for batch_idx, (labeled_images, labels) in enumerate(train_loader):
            labeled_images, labels = labeled_images.to(device), labels.to(device)
            
            # 获取无标签数据批次
            weak_images = None
            strong_images = None
            
            if unlabeled_iter is not None:
                try:
                    batch_data = next(unlabeled_iter)
                    
                    # 根据您的数据结构，batch_data是列表：[images_list, labels]
                    if isinstance(batch_data, list) and len(batch_data) == 2:
                        images_list = batch_data[0]
                        if isinstance(images_list, list) and len(images_list) == 2:
                            weak_images = images_list[0]  # 弱增强图像
                            strong_images = images_list[1]  # 强增强图像
                    
                    if weak_images is not None and strong_images is not None:
                        weak_images = weak_images.to(device)
                        strong_images = strong_images.to(device)
                        
                except StopIteration:
                    # 重新初始化无标签数据迭代器
                    unlabeled_iter = iter(unlabeled_loader)
                    try:
                        batch_data = next(unlabeled_iter)
                        if isinstance(batch_data, list) and len(batch_data) == 2:
                            images_list = batch_data[0]
                            if isinstance(images_list, list) and len(images_list) == 2:
                                weak_images = images_list[0]
                                strong_images = images_list[1]
                                weak_images = weak_images.to(device)
                                strong_images = strong_images.to(device)
                    except:
                        weak_images = strong_images = None
            
            # 使用混合精度训练
            with torch.amp.autocast('cuda', enabled=use_amp):
                # ====== 有标签数据监督损失 ======
                student_labeled_outputs = student_model(labeled_images)
                supervised_loss = supervised_criterion(student_labeled_outputs, labels)
                
                # ====== 无标签数据一致性损失 ======
                consistency_loss = 0.0
                if weak_images is not None and strong_images is not None:
                    # 学生模型预测（使用强增强）
                    student_unlabeled_outputs = student_model(strong_images)
                    
                    # 教师模型预测（使用弱增强，不计算梯度）
                    with torch.no_grad():
                        teacher_unlabeled_outputs = teacher_model(weak_images)
                    
                    # 计算一致性损失
                    if use_kl_loss:
                        # KL散度需要log_softmax输入
                        student_log_probs = torch.log_softmax(student_unlabeled_outputs, dim=1)
                        teacher_probs = torch.softmax(teacher_unlabeled_outputs, dim=1)
                        consistency_loss = consistency_criterion(student_log_probs, teacher_probs)
                    else:
                        # MSE使用概率分布
                        student_probs = torch.softmax(student_unlabeled_outputs, dim=1)
                        teacher_probs = torch.softmax(teacher_unlabeled_outputs, dim=1)
                        consistency_loss = consistency_criterion(student_probs, teacher_probs)
                
                # ====== 总损失 ======
                total_loss = supervised_loss + current_consistency_weight * consistency_loss
            
            # ====== 反向传播和优化 ======
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            
            # 梯度裁剪
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            
            # ====== 梯度监控 ======
            if batch_idx % 50 == 0:
                log_gradient_info(student_model, optimizer, epoch, batch_idx, log_freq=50)
            
            # ====== 更新教师模型（EMA） ======
            with torch.no_grad():
                for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
                    teacher_param.data.mul_(ema_decay).add_(student_param.data, alpha=1 - ema_decay)
            
            # ====== 记录损失 ======
            running_supervised_loss += supervised_loss.item()
            running_consistency_loss += consistency_loss.item() if (weak_images is not None and strong_images is not None) else 0
            running_total_loss += total_loss.item()
            
            # 每50个batch打印进度
            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | '
                      f'Supervised Loss: {supervised_loss.item():.4f} | '
                      f'Consistency Loss: {consistency_loss.item():.4f} | '
                      f'Total Loss: {total_loss.item():.4f} | LR: {current_lr:.6f} | '
                      f'Consistency Weight: {current_consistency_weight:.2f}')
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # ====== 验证 ======
        teacher_model.eval()
        accuracy, _ = evaluate_model(teacher_model, val_loader, device)
        val_accuracies.append(accuracy)
        
        # 保存最佳模型
        if save_best and accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            best_model_state = {
                'teacher_state_dict': teacher_model.state_dict(),
                'student_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch
            }
            torch.save(best_model_state, f'best_model_{experiment_name}.pth')
            print(f"✅ 保存最佳模型，验证准确率: {accuracy:.2f}%")
        
        # 计算平均损失
        avg_supervised_loss = running_supervised_loss / len(train_loader)
        avg_consistency_loss = running_consistency_loss / len(train_loader)
        avg_total_loss = running_total_loss / len(train_loader)
        
        supervised_losses.append(avg_supervised_loss)
        consistency_losses.append(avg_consistency_loss)
        train_losses.append(avg_total_loss)
        
        epoch_time = time.time() - epoch_start_time
        training_times.append(epoch_time)
        
        print(f'Epoch [{epoch+1}/{epochs}] | 时间: {epoch_time:.1f}s | '
              f'监督损失: {avg_supervised_loss:.4f} | 一致性损失: {avg_consistency_loss:.4f} | '
              f'总损失: {avg_total_loss:.4f} | 验证准确率: {accuracy:.2f}% | '
              f'一致性权重: {current_consistency_weight:.2f} | LR: {current_lr:.6f}')
    
    # ====== 最终测试和分析 ======
    if best_model_state and os.path.exists(f'best_model_{experiment_name}.pth'):
        checkpoint = torch.load(f'best_model_{experiment_name}.pth')
        teacher_model.load_state_dict(checkpoint['teacher_state_dict'])
        print(f"加载最佳模型进行测试 (epoch {checkpoint['epoch']+1})")
    
    teacher_model.eval()
    final_accuracy, class_accuracy = evaluate_model(teacher_model, test_loader, device)
    
    # ====== 新增：详细类别分析 ======
    print(f"\n🔍 开始详细类别分析...")
    
    # 1. 测试集类别准确率分析
    class_accuracies, best_class, worst_class, _ = analyze_class_performance(
        teacher_model, test_loader, device
    )
    
    print(f"\n📊 测试集类别准确率排名:")
    for class_name, acc in sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {acc:.2f}%")
    
    print(f"\n🏆 最佳分类类别: {best_class[0]} ({best_class[1]:.2f}%)")
    print(f"📉 最差分类类别: {worst_class[0]} ({worst_class[1]:.2f}%)")
    
    # 2. 无标签数据预测分析
    if unlabeled_loader:
        unlabeled_analysis = analyze_unlabeled_data_performance(
            teacher_model, unlabeled_loader, device,
            confidence_threshold=0.8,
            num_samples=2000
        )
        
        print(f"\n📊 无标签数据预测分析:")
        print(f"  整体平均置信度: {unlabeled_analysis['overall_avg_confidence']:.3f}")
        print(f"  高置信度预测比例: {unlabeled_analysis['high_confidence_ratio']:.3f}")
        print(f"  模型最自信的类别: {unlabeled_analysis['most_confident_class']}")
        print(f"  模型最不自信的类别: {unlabeled_analysis['least_confident_class']}")
        
        print(f"\n📈 每个类别的预测统计:")
        for class_name, stats in unlabeled_analysis['class_stats'].items():
            print(f"  {class_name}: 预测次数={stats['count']}, "
                  f"平均置信度={stats['avg_confidence']:.3f}, "
                  f"高置信度={stats['high_confidence_count']}")
    else:
        unlabeled_analysis = None
        print("\n⚠️  无无标签数据可用于分析")
    
    # 3. 混淆矩阵分析
    confusion_analysis = plot_confusion_matrix_and_analysis(
        teacher_model, test_loader, device, experiment_name
    )
    
    print(f"\n🔍 混淆矩阵分析:")
    print(f"  最佳F1分数类别: {confusion_analysis['best_class']}")
    print(f"  最差F1分数类别: {confusion_analysis['worst_class']}")
    
    print(f"\n🔗 最易混淆的类别对:")
    for pair in confusion_analysis['top_confusion_pairs']:
        print(f"  {pair['from']} → {pair['to']}: {pair['rate']:.3f}")
    
    avg_epoch_time = sum(training_times) / len(training_times)
    print(f"\n训练完成!")
    print(f"平均每轮训练时间: {avg_epoch_time:.1f}秒")
    print(f"最终测试集准确率: {final_accuracy:.2f}%")
    print(f"最佳验证准确率: {best_val_accuracy:.2f}%")
    
    return {
        'student_model': student_model,
        'teacher_model': teacher_model,
        'train_losses': train_losses,
        'supervised_losses': supervised_losses,
        'consistency_losses': consistency_losses,
        'val_accuracies': val_accuracies,
        'final_accuracy': final_accuracy,
        'class_accuracy': class_accuracy,
        'training_times': training_times,
        'learning_rates': learning_rates,
        'best_val_accuracy': best_val_accuracy,
        'experiment_name': experiment_name,
        'gradient_norms': gradient_norms,
        # 新增的分析结果
        'detailed_class_accuracies': class_accuracies,
        'best_class': best_class,
        'worst_class': worst_class,
        'unlabeled_analysis': unlabeled_analysis if unlabeled_loader else None,
        'confusion_analysis': confusion_analysis
    }

def main():
    """主函数，运行Mean Teacher半监督实验"""
    
    print("STL-10 图像分类实验 - Mean Teacher半监督学习")
    print(f"操作系统: {platform.system()}")
    print("Python 3.9 + PyTorch 2.7.1 + CUDA 12.8")
    print("="*60)
    
    # Mean Teacher实验配置
    mean_teacher_experiments = [
        {
            'name': 'MeanTeacher_Adam', 
            'optimizer': 'adam', 
            'lr': 0.0001, 
            'batch_size': 64, 
            'epochs': 50,
            'consistency_weight': 10.0,
            'ema_decay': 0.999,
            'use_kl_loss': False,
            'max_grad_norm': 1.0,
            'num_workers': 0
        },
        {
            'name': 'MeanTeacher_AdamW_Fixed', 
            'optimizer': 'adamw', 
            'lr': 0.0002,  # 提高学习率（原为0.0001）->0.001总损失很高，一致性损失高，降低学习至0.0005->loss波动太大学习率再降到0.0002
            'batch_size': 64, 
            'epochs': 50,
            'consistency_weight': 1.0,  # 降低一致性权重（原为10.0）-》再降低一致性权重
            'ema_decay': 0.997,
            # 尝试MSE损失
            'use_kl_loss': True,
            'max_grad_norm': 1.0,  # 更严格的梯度裁剪，从1.0降低0.5->0.5太严格的裁剪会扭曲梯度方向
            'num_workers': 0
        },
        {
            'name': 'MeanTeacher_SGD', 
            'optimizer': 'sgd', 
            'lr': 0.01, 
            'batch_size': 64, 
            'epochs': 50,
            'consistency_weight': 10.0,
            'ema_decay': 0.999,
            'use_kl_loss': False,
            'max_grad_norm': 1.0,
            'num_workers': 0
        }
    ]
    
    results = {}
    
    for exp_config in mean_teacher_experiments:
        print(f"\n{'='*50}")
        print(f"实验: {exp_config['name']}")
        print(f"{'='*50}")
        
        result = train_mean_teacher(
            optimizer_name=exp_config['optimizer'],
            learning_rate=exp_config['lr'],
            batch_size=exp_config['batch_size'],
            epochs=exp_config['epochs'],
            use_amp=True,
            model_type='resnet18',
            validation_split=0.1,
            consistency_weight=exp_config['consistency_weight'],
            ema_decay=exp_config['ema_decay'],
            warmup_epochs=10,
            use_kl_loss=exp_config['use_kl_loss'],
            max_grad_norm=exp_config['max_grad_norm'],
            experiment_name=exp_config['name'],
            num_workers=exp_config['num_workers']
        )
        
        results[exp_config['name']] = result
        
        print(f"\n📊 {exp_config['name']} 最终测试准确率: {result['final_accuracy']:.2f}%")
        print(f"📊 {exp_config['name']} 最佳验证准确率: {result['best_val_accuracy']:.2f}%")
    
    # 绘制Mean Teacher训练曲线
    plot_mean_teacher_results(results)
    
    # 比较不同配置的性能
    print_experiment_summary(results)
    
    # 保存结果
    with open('mean_teacher_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Mean Teacher结果已保存到 mean_teacher_results.pkl")
    
    # 性能统计
    total_training_time = sum(sum(result['training_times']) for result in results.values())
    print(f"\n总训练时间: {total_training_time:.1f}秒 ({total_training_time/60:.1f}分钟)")
    
    # 找出最佳模型
    best_exp = max(results.items(), key=lambda x: x[1]['final_accuracy'])
    best_val_exp = max(results.items(), key=lambda x: x[1]['best_val_accuracy'])
    
    print(f"\n🎉 最佳Mean Teacher模型 (测试集): {best_exp[0]} - 测试准确率: {best_exp[1]['final_accuracy']:.2f}%")
    print(f"🎉 最佳Mean Teacher模型 (验证集): {best_val_exp[0]} - 验证准确率: {best_val_exp[1]['best_val_accuracy']:.2f}%")
    
    # 检查过拟合情况
    print("\n过拟合检查:")
    check_overfitting(results)  # ✅ 直接传入整个results字典
    
    return results

if __name__ == "__main__":
    results = main()