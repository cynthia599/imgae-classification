# utils.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def plot_mean_teacher_results(results):
    """绘制Mean Teacher训练结果"""
    # ====== 添加标签映射修复显示问题 ======
    # 修复字体导致的显示问题（S显示为5）
    label_mapping = {
        'MeanTeacher_SGD': 'MeanTeacher_SGD',
        'MeanTeacher_Adam': 'MeanTeacher_Adam',
        'MeanTeacher_AdamW': 'MeanTeacher_AdamW',
        # 添加其他可能的实验名称
    }
    
    # 创建显示名称列表
    exp_names = list(results.keys())
    display_names = [label_mapping.get(name, name) for name in exp_names]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 绘制总损失 - 使用display_names而不是exp_names
    for exp_name, display_name in zip(exp_names, display_names):
        ax1.plot(results[exp_name]['train_losses'], label=display_name, linewidth=2)
    
    ax1.set_title('Mean Teacher Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制监督损失
    for exp_name, display_name in zip(exp_names, display_names):
        if 'supervised_losses' in results[exp_name]:
            ax2.plot(results[exp_name]['supervised_losses'], label=display_name, linewidth=2)
    
    ax2.set_title('Supervised Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 绘制一致性损失
    for exp_name, result in results.items():
        if 'consistency_losses' in result:
            ax3.plot(result['consistency_losses'], label=exp_name, linewidth=2)
    
    ax3.set_title('Consistency Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 绘制验证准确率
    for exp_name, result in results.items():
        ax4.plot(result['val_accuracies'], label=exp_name, linewidth=2)
    
    ax4.set_title('Validation Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mean_teacher_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制最终性能比较
    fig, ax = plt.subplots(figsize=(10, 6))
    exp_names = list(results.keys())
    test_accuracies = [results[name]['final_accuracy'] for name in exp_names]
    
    x = np.arange(len(exp_names))
    
    ax.bar(x, test_accuracies, alpha=0.8, color=['blue', 'orange', 'green'])
    
    ax.set_xlabel('Mean Teacher Configuration')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Mean Teacher Final Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(test_accuracies):
        ax.text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('mean_teacher_final_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def evaluate_model(model, test_loader, device):
    """评估模型在测试集上的性能"""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    # 计算每个类别的准确率
    class_correct = [0] * 10
    class_total = [0] * 10
    
    for i in range(len(all_labels)):
        label = all_labels[i]
        class_correct[label] += (all_predictions[i] == label)
        class_total[label] += 1
    
    class_accuracy = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                     for i in range(10)]
    
    return accuracy, class_accuracy

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

def plot_optimizer_comparison(results):
    """绘制不同优化器的训练历史比较图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制训练损失
    for exp_name, result in results.items():
        ax1.plot(result['train_losses'], label=exp_name, linewidth=2)
    
    ax1.set_title('Training Loss Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制验证准确率
    for exp_name, result in results.items():
        ax2.plot(result['val_accuracies'], label=exp_name, linewidth=2)
    
    ax2.set_title('Validation Accuracy Comparison')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制最终性能比较
    fig, ax = plt.subplots(figsize=(10, 6))
    exp_names = list(results.keys())
    test_accuracies = [results[name]['final_accuracy'] for name in exp_names]
    val_accuracies = [results[name]['val_accuracies'][-1] for name in exp_names]
    
    x = np.arange(len(exp_names))
    width = 0.35
    
    ax.bar(x - width/2, test_accuracies, width, label='Test Accuracy', alpha=0.8)
    ax.bar(x + width/2, val_accuracies, width, label='Validation Accuracy', alpha=0.8)
    
    ax.set_xlabel('Optimizer')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Final Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(test_accuracies):
        ax.text(i - width/2, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
    for i, v in enumerate(val_accuracies):
        ax.text(i + width/2, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('final_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# 注释掉了plot_rmsprop_comparison函数，因为它包含针对不同参数设置的RMSprop优化器的代码
# def plot_rmsprop_comparison(results):
#     """专门绘制RMSprop参数优化的比较图"""
#     # 创建2x2的子图布局
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
#     # 绘制训练损失
#     for exp_name, result in results.items():
#         ax1.plot(result['train_losses'], label=exp_name, linewidth=2)
    
#     ax1.set_title('RMSprop Training Loss Comparison')
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Loss')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
    
#     # 绘制验证准确率
#     for exp_name, result in results.items():
#         ax2.plot(result['val_accuracies'], label=exp_name, linewidth=2)
    
#     ax2.set_title('RMSprop Validation Accuracy Comparison')
#     ax2.set_xlabel('Epoch')
#     ax2.set_ylabel('Accuracy (%)')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
    
#     # 绘制训练vs验证损失
#     for exp_name, result in results.items():
#         if 'val_losses' in result and result['val_losses']:
#             epochs = range(1, len(result['train_losses']) + 1)
#             ax3.plot(epochs, result['train_losses'], label=f'{exp_name} (Train)', linewidth=1.5, linestyle='-')
#             ax3.plot(epochs, result['val_losses'], label=f'{exp_name} (Val)', linewidth=1.5, linestyle='--')
    
#     ax3.set_title('RMSprop Train vs Validation Loss')
#     ax3.set_xlabel('Epoch')
#     ax3.set_ylabel('Loss')
#     ax3.legend(fontsize='small')
#     ax3.grid(True, alpha=0.3)
    
#     # 绘制最终性能比较
#     exp_names = list(results.keys())
#     test_accuracies = [results[name]['final_accuracy'] for name in exp_names]
#     val_accuracies = [results[name]['val_accuracies'][-1] for name in exp_names]
    
#     x = np.arange(len(exp_names))
#     width = 0.35
    
#     ax4.bar(x - width/2, test_accuracies, width, label='Test Accuracy', alpha=0.8)
#     ax4.bar(x + width/2, val_accuracies, width, label='Validation Accuracy', alpha=0.8)
    
#     ax4.set_xlabel('RMSprop Configuration')
#     ax4.set_ylabel('Accuracy (%)')
#     ax4.set_title('RMSprop Final Performance Comparison')
#     ax4.set_xticks(x)
#     ax4.set_xticklabels(exp_names, rotation=45, ha='right')
#     ax4.legend()
#     ax4.grid(True, alpha=0.3)
    
#     # 在柱状图上添加数值标签
#     for i, v in enumerate(test_accuracies):
#         ax4.text(i - width/2, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
#     for i, v in enumerate(val_accuracies):
#         ax4.text(i + width/2, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    
#     plt.tight_layout()
#     plt.savefig('rmsprop_optimization_comparison.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
#     # 绘制学习率对比
#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     # 提取学习率信息
#     lr_values = []
#     config_names = []
#     for exp_name in exp_names:
#         if 'VeryLow' in exp_name:
#             lr = 0.00002
#         elif 'Low' in exp_name:
#             lr = 0.00005
#         elif 'Medium' in exp_name:
#             lr = 0.0001
#         else:
#             lr = 0.00005  # 默认
        
#         lr_values.append(lr)
#         config_names.append(exp_name)
    
#     # 绘制学习率与准确率的关系
#     scatter = ax.scatter(lr_values, test_accuracies, c=val_accuracies, cmap='viridis', s=100, alpha=0.7)
#     ax.set_xlabel('Learning Rate')
#     ax.set_ylabel('Test Accuracy (%)')
#     ax.set_title('RMSprop: Learning Rate vs Test Accuracy')
#     ax.set_xscale('log')
#     ax.grid(True, alpha=0.3)
    
#     # 添加颜色条表示验证准确率
#     cbar = plt.colorbar(scatter)
#     cbar.set_label('Validation Accuracy (%)')
    
#     # 添加配置名称标签
#     for i, name in enumerate(config_names):
#         ax.annotate(name, (lr_values[i], test_accuracies[i]), 
#                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
#     plt.tight_layout()
#     plt.savefig('rmsprop_lr_vs_accuracy.png', dpi=300, bbox_inches='tight')
#     plt.show()

def print_experiment_summary(results):
    """打印实验总结"""
    print("\n" + "="*80)
    print("实验总结")
    print("="*80)
    
    # 按测试准确率排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['final_accuracy'], reverse=True)
    
    print(f"{'优化器':<25} {'测试准确率':<12} {'验证准确率':<12} {'训练轮数':<10}")
    print("-"*80)
    
    for exp_name, result in sorted_results:
        test_acc = result['final_accuracy']
        val_acc = result['val_accuracies'][-1] if result['val_accuracies'] else 0
        epochs = len(result['train_losses'])
        
        print(f"{exp_name:<25} {test_acc:<11.2f}% {val_acc:<11.2f}% {epochs:<10}")
    
    # 找出最佳模型
    best_exp = max(results.items(), key=lambda x: x[1]['final_accuracy'])
    print("\n🎉 最佳模型:")
    print(f"   名称: {best_exp[0]}")
    print(f"   测试准确率: {best_exp[1]['final_accuracy']:.2f}%")
    print(f"   最终验证准确率: {best_exp[1]['val_accuracies'][-1]:.2f}%")

def check_overfitting(results, threshold=0.5):
    """检查过拟合情况
    Args:
        results: 训练结果字典
        threshold: 过拟合阈值，测试集与验证集准确率差距超过此值认为可能过拟合
    """
    print("\n" + "="*50)
    print("过拟合分析")
    print("="*50)
    
    for exp_name, result in results.items():
        if len(result['val_accuracies']) > 0:
            train_loss_final = result['train_losses'][-1] if result['train_losses'] else 0
            val_acc_final = result['val_accuracies'][-1]
            test_acc = result['final_accuracy']
            
            # 计算测试集与验证集准确率的差距
            gap = test_acc - val_acc_final
            
            print(f"{exp_name}:")
            print(f"  最终训练损失: {train_loss_final:.4f}")
            if 'val_losses' in result and result['val_losses']:
                print(f"  最终验证损失: {result['val_losses'][-1]:.4f}")
            if 'final_test_loss' in result:
                print(f"  最终测试损失: {result['final_test_loss']:.4f}")
            print(f"  最终验证准确率: {val_acc_final:.2f}%")
            print(f"  测试准确率: {test_acc:.2f}%")
            print(f"  测试-验证差距: {gap:+.2f}%")
            
            if gap > threshold:
                print(f"  ⚠️  可能过拟合 (差距 > {threshold}%)")
            elif gap < -threshold:
                print(f"  🔍 可能欠拟合 (差距 < -{threshold}%)")
            else:
                print(f"  ✅ 拟合良好")
            print()

def plot_confusion_matrix(model, test_loader, device, class_names=None):
    """绘制混淆矩阵"""
    if class_names is None:
        class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 
                      'dog', 'horse', 'monkey', 'ship', 'truck']
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def print_detailed_classification_report(model, test_loader, device, class_names=None):
    """打印详细的分类报告"""
    if class_names is None:
        class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 
                      'dog', 'horse', 'monkey', 'ship', 'truck']
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 生成分类报告
    report = classification_report(all_labels, all_predictions, 
                                  target_names=class_names, digits=4)
    print("\n详细分类报告:")
    print("="*60)
    print(report)
    
    return report

# 测试代码
if __name__ == "__main__":
    print("Utils模块测试")
    
    # 模拟一些结果数据用于测试绘图函数
    test_results = {
        'MeanTeacher_Adam': {
            'train_losses': [2.1, 1.5, 1.2, 0.9, 0.7, 0.5, 0.4, 0.3],
            'val_accuracies': [45.2, 58.7, 65.3, 68.9, 72.1, 75.3, 77.8, 79.2],
            'supervised_losses': [2.0, 1.6, 1.3, 1.0, 0.8, 0.6, 0.5, 0.4],
            'consistency_losses': [0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
            'final_accuracy': 79.5,
        },
        'MeanTeacher_SGD': {
            'train_losses': [2.3, 1.8, 1.4, 1.1, 0.8, 0.6, 0.45, 0.35],
            'val_accuracies': [42.1, 53.4, 61.2, 66.8, 69.5, 72.8, 75.6, 77.9],
            'supervised_losses': [2.2, 1.9, 1.5, 1.2, 0.9, 0.7, 0.55, 0.45],
            'consistency_losses': [0.15, 0.12, 0.09, 0.07, 0.06, 0.05, 0.04, 0.03],
            'final_accuracy': 78.2,
        },
        'MeanTeacher_AdamW': {
            'train_losses': [2.0, 1.4, 1.0, 0.7, 0.5, 0.35, 0.25, 0.2],
            'val_accuracies': [48.5, 62.3, 70.1, 74.8, 77.2, 79.1, 80.5, 81.2],
            'supervised_losses': [1.9, 1.5, 1.1, 0.8, 0.6, 0.45, 0.35, 0.3],
            'consistency_losses': [0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01],
            'final_accuracy': 81.5,
        }
    }
    
    # 测试Mean Teacher比较函数
    plot_mean_teacher_results(test_results)
    print_experiment_summary(test_results)
    check_overfitting(test_results)