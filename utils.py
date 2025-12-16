# utils.py - 完整的工具模块
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'，使用交互式后端
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from torchvision.utils import make_grid
import torch.optim as optim
import pandas as pd

# 设置中文字体和图表参数
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文和英文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['figure.dpi'] = 100  # 设置图像清晰度
plt.rcParams['figure.figsize'] = [12, 8]  # 默认图表尺寸

# STL-10的类别名称
STL10_class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 
                     'dog', 'horse', 'monkey', 'ship', 'truck']

def plot_samples(dataloader, num_samples=10, save_path=None):
    """
    可视化数据集样本 - 适应彩色图像
    """
    images, labels = next(iter(dataloader))
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_samples, 10)):
        img = images[i].numpy()
        # 彩色图像反归一化 (3通道)
        img = np.transpose(img, (1, 2, 0))  # 从 (C,H,W) 转为 (H,W,C)
        img = (img * 0.5) + 0.5  # 反归一化
        img = np.clip(img, 0, 1)  # 确保值在[0,1]范围内
        
        axes[i].imshow(img)
        axes[i].set_title(f'{STL10_class_names[labels[i]]} ({labels[i]})')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 样本图像已保存到: {save_path}")
    
    plt.show()
    return fig

def plot_training_history(train_losses, val_losses, val_accuracies, optimizer_name, save_path=None):
    """
    绘制训练历史曲线 - 修改：增加验证损失
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 训练损失和验证损失
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='训练损失')
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='验证损失')
    ax1.set_title(f'{optimizer_name.upper()} - 损失曲线')
    ax1.set_xlabel('训练轮数')
    ax1.set_ylabel('损失值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 训练损失（单独）
    ax2.plot(epochs, train_losses, 'b-', linewidth=2)
    ax2.set_title(f'{optimizer_name.upper()} - 训练损失')
    ax2.set_xlabel('训练轮数')
    ax2.set_ylabel('损失值')
    ax2.grid(True, alpha=0.3)
    
    # 验证准确率
    ax3.plot(epochs, val_accuracies, 'g-', linewidth=2)
    ax3.set_title(f'{optimizer_name.upper()} - 验证准确率')
    ax3.set_xlabel('训练轮数')
    ax3.set_ylabel('准确率 (%)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 100])
    
    plt.suptitle(f'{optimizer_name.upper()} 优化器训练历史', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 训练曲线已保存到: {save_path}")
    
    plt.show()
    return fig

def plot_optimizer_comparison(results_dict, save_path=None):
    """
    比较不同优化器的性能
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 最终准确率比较
    optimizers = list(results_dict.keys())
    final_accuracies = [results_dict[opt]['final_accuracy'] for opt in optimizers]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax1.bar(optimizers, final_accuracies, color=colors)
    ax1.set_title('优化器最终测试准确率对比', fontsize=12, fontweight='bold')
    ax1.set_ylabel('准确率 (%)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上添加数值
    for bar, accuracy in zip(bars, final_accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{accuracy:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 训练曲线比较
    for optimizer in optimizers:
        ax2.plot(results_dict[optimizer]['val_accuracies'], 
                label=optimizer.upper(), linewidth=2)
    
    ax2.set_title('优化器验证准确率对比', fontsize=12, fontweight='bold')
    ax2.set_xlabel('训练轮数')
    ax2.set_ylabel('准确率 (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 优化器比较图已保存到: {save_path}")
    else:
        plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("✅ 优化器比较图已保存到: accuracy_comparison.png")
    
    plt.show()
    return fig

def plot_loss_comparison(results_dict, save_path=None):
    """
    新增：绘制损失函数对比图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    optimizers = list(results_dict.keys())
    
    # 训练损失对比
    for optimizer in optimizers:
        ax1.plot(results_dict[optimizer]['train_losses'], 
                label=optimizer.upper(), linewidth=2)
    
    ax1.set_title('优化器训练损失对比', fontsize=12, fontweight='bold')
    ax1.set_xlabel('训练轮数')
    ax1.set_ylabel('损失值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 验证损失对比
    for optimizer in optimizers:
        ax2.plot(results_dict[optimizer]['val_losses'], 
                label=optimizer.upper(), linewidth=2)
    
    ax2.set_title('优化器验证损失对比', fontsize=12, fontweight='bold')
    ax2.set_xlabel('训练轮数')
    ax2.set_ylabel('损失值')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 损失函数对比图已保存到: {save_path}")
    else:
        plt.savefig('loss_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("✅ 损失函数对比图已保存到: loss_comparison.png")
    
    plt.show()
    return fig

def evaluate_model(model, test_loader, device='cpu'):
    """
    评估模型性能
    """
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 每个类别的准确率
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    accuracy = 100 * correct / total
    class_accuracy = [100 * class_correct[i] / class_total[i] for i in range(10)]
    
    print(f'整体测试准确率: {accuracy:.2f}%')
    print('\n每个类别的准确率:')
    for i in range(10):
        print(f'{STL10_class_names[i]:15s}: {class_accuracy[i]:.2f}%')
    
    return accuracy, class_accuracy

def setup_optimizer(model, optimizer_name, learning_rate=0.001, momentum=0.9, weight_decay=0.0001):
    """
    设置优化器 - 添加权重衰减支持（对BatchNorm模型很重要）
    """
    if optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd_momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    print(f"使用优化器: {optimizer_name} (学习率: {learning_rate}, 权重衰减: {weight_decay})")
    if optimizer_name.lower() == 'sgd_momentum':
        print(f"动量参数: {momentum}")
    
    return optimizer

def print_experiment_summary(results_dict):
    """
    打印实验总结
    """
    print("\n" + "="*60)
    print("                   实验总结")
    print("="*60)
    
    best_accuracy = 0
    best_optimizer = ""
    
    for optimizer, results in results_dict.items():
        accuracy = results['final_accuracy']
        print(f"{optimizer:15s}: {accuracy:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_optimizer = optimizer
    
    print("-"*60)
    print(f"最佳优化器: {best_optimizer} ({best_accuracy:.2f}%)")
    print("="*60)

def count_parameters(model):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    return total_params, trainable_params

def plot_class_accuracy(class_accuracy, optimizer_name, save_path=None):
    """
    绘制各类别准确率柱状图
    """
    plt.figure(figsize=(12, 6))
    
    # 确保class_accuracy长度正确
    if len(class_accuracy) != len(STL10_class_names):
        # 如果长度不匹配，只绘制已有的数据
        n = min(len(class_accuracy), len(STL10_class_names))
        class_names = STL10_class_names[:n]
        acc_data = class_accuracy[:n]
    else:
        class_names = STL10_class_names
        acc_data = class_accuracy
    
    # 创建颜色渐变
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(class_names)))
    
    # 创建条形图
    bars = plt.bar(range(len(class_names)), acc_data, color=colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # 设置图表样式
    plt.title(f'{optimizer_name.upper()} 优化器 - STL-10 各类别准确率', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('准确率 (%)', fontsize=12)
    
    # 设置x轴刻度
    plt.xticks(range(len(class_names)), class_names, 
               rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # 设置网格和坐标轴范围
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.ylim([0, max(acc_data) * 1.15 if max(acc_data) > 0 else 100])
    
    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, acc_data)):
        height = bar.get_height()
        color = 'white' if acc > max(acc_data) * 0.7 else 'black'
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color=color)
    
    # 添加平均准确率线
    avg_acc = np.mean(acc_data)
    plt.axhline(y=avg_acc, color='red', linestyle='--', linewidth=2, alpha=0.7)
    plt.text(len(class_names) - 0.5, avg_acc + 1, 
             f'平均: {avg_acc:.1f}%', 
             ha='right', va='bottom', color='red', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 类别准确率图已保存到: {save_path}")
    
    plt.show()
    return plt.gcf()

def plot_training_curves_comparison(results_dict, save_path=None):
    """
    绘制所有优化器的训练曲线对比图
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    optimizers = list(results_dict.keys())
    
    # 定义颜色和线型
    colors = ['#2E86C1', '#E74C3C', '#28B463', '#F39C12', '#8E44AD']
    line_styles = ['-', '--', '-.', ':', '-']
    
    # 1. 训练损失对比
    ax1 = axes[0, 0]
    for idx, opt in enumerate(optimizers):
        if 'train_losses' in results_dict[opt] and results_dict[opt]['train_losses']:
            train_losses = results_dict[opt]['train_losses']
            epochs = range(1, len(train_losses) + 1)
            
            ax1.plot(epochs, train_losses, 
                    color=colors[idx % len(colors)],
                    linestyle=line_styles[idx % len(line_styles)],
                    linewidth=2,
                    label=f"{opt.upper()}")
    
    ax1.set_title('训练损失对比', fontsize=12, fontweight='bold')
    ax1.set_xlabel('训练轮数 (Epoch)', fontsize=10)
    ax1.set_ylabel('损失值 (Loss)', fontsize=10)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    if 'epochs' in locals():
        ax1.set_xlim([1, len(epochs)])
    
    # 2. 验证损失对比
    ax2 = axes[0, 1]
    for idx, opt in enumerate(optimizers):
        if 'val_losses' in results_dict[opt] and results_dict[opt]['val_losses']:
            val_losses = results_dict[opt]['val_losses']
            epochs = range(1, len(val_losses) + 1)
            
            ax2.plot(epochs, val_losses, 
                    color=colors[idx % len(colors)],
                    linestyle=line_styles[idx % len(line_styles)],
                    linewidth=2,
                    label=f"{opt.upper()}")
    
    ax2.set_title('验证损失对比', fontsize=12, fontweight='bold')
    ax2.set_xlabel('训练轮数 (Epoch)', fontsize=10)
    ax2.set_ylabel('损失值 (Loss)', fontsize=10)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    if 'epochs' in locals():
        ax2.set_xlim([1, len(epochs)])
    
    # 3. 验证准确率对比
    ax3 = axes[1, 0]
    for idx, opt in enumerate(optimizers):
        if 'val_accuracies' in results_dict[opt] and results_dict[opt]['val_accuracies']:
            val_accuracies = results_dict[opt]['val_accuracies']
            epochs = range(1, len(val_accuracies) + 1)
            
            ax3.plot(epochs, val_accuracies, 
                    color=colors[idx % len(colors)],
                    linestyle=line_styles[idx % len(line_styles)],
                    linewidth=2,
                    label=f"{opt.upper()}")
    
    ax3.set_title('验证准确率对比', fontsize=12, fontweight='bold')
    ax3.set_xlabel('训练轮数 (Epoch)', fontsize=10)
    ax3.set_ylabel('准确率 (%)', fontsize=10)
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_ylim([0, 100])
    if 'epochs' in locals():
        ax3.set_xlim([1, len(epochs)])
    
    # 4. 最终测试准确率对比
    ax4 = axes[1, 1]
    final_accuracies = []
    opt_names = []
    
    for opt in optimizers:
        if 'final_accuracy' in results_dict[opt]:
            final_accuracies.append(results_dict[opt]['final_accuracy'])
            opt_names.append(opt.upper())
        elif 'val_accuracies' in results_dict[opt] and results_dict[opt]['val_accuracies']:
            final_accuracies.append(results_dict[opt]['val_accuracies'][-1])
            opt_names.append(opt.upper())
    
    # 按准确率排序
    sorted_data = sorted(zip(opt_names, final_accuracies), key=lambda x: x[1], reverse=True)
    opt_names = [x[0] for x in sorted_data]
    final_accuracies = [x[1] for x in sorted_data]
    
    bars = ax4.bar(opt_names, final_accuracies, 
                   color=colors[:len(opt_names)], 
                   edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax4.set_title('最终测试准确率对比', fontsize=12, fontweight='bold')
    ax4.set_ylabel('准确率 (%)', fontsize=10)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.set_ylim([0, max(final_accuracies) * 1.15 if max(final_accuracies) > 0 else 100])
    
    # 添加数值标签
    for bar, accuracy in zip(bars, final_accuracies):
        height = bar.get_height()
        color = 'white' if accuracy > max(final_accuracies) * 0.7 else 'black'
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{accuracy:.2f}%', ha='center', va='bottom', 
                fontweight='bold', color=color, fontsize=9)
    
    # 旋转x轴标签
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('STL-10 优化器训练曲线对比', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 训练曲线对比图已保存到: {save_path}")
    else:
        plt.savefig('training_curves_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("✅ 训练曲线对比图已保存到: training_curves_comparison.png")
    
    plt.show()
    return fig

def generate_optimizer_table(results_dict):
    """
    生成优化器性能对比表格并保存为图片
    """
    # 准备数据
    table_data = []
    
    for opt_name, result in results_dict.items():
        # 计算收敛速度（达到最大准确率80%的epoch）
        convergence_info = "N/A"
        if 'val_accuracies' in result and result['val_accuracies']:
            max_acc = max(result['val_accuracies'])
            target_acc = max_acc * 0.8
            
            for epoch, acc in enumerate(result['val_accuracies']):
                if acc >= target_acc:
                    convergence_info = f"{epoch + 1}"
                    break
            if convergence_info == "N/A":
                convergence_info = ">50"
        
        # 计算最终准确率
        final_acc = result.get('final_accuracy', 0)
        if final_acc == 0 and 'val_accuracies' in result and result['val_accuracies']:
            final_acc = result['val_accuracies'][-1]
        
        # 计算训练损失下降率
        train_loss_change = "N/A"
        if 'train_losses' in result and len(result['train_losses']) >= 2:
            start_loss = result['train_losses'][0]
            end_loss = result['train_losses'][-1]
            if start_loss > 0:
                loss_change = (start_loss - end_loss) / start_loss * 100
                train_loss_change = f"{loss_change:.1f}%"
        
        row = [
            opt_name.upper(),
            f"{final_acc:.2f}%",
            len(result.get('train_losses', [])),
            convergence_info,
            f"{result.get('train_losses', [0])[-1]:.4f}" if result.get('train_losses') else "N/A",
            f"{result.get('val_losses', [0])[-1]:.4f}" if result.get('val_losses') else "N/A",
            train_loss_change
        ]
        table_data.append(row)
    
    # 按最终准确率排序
    table_data.sort(key=lambda x: float(x[1].replace('%', '')), reverse=True)
    
    # 创建DataFrame
    headers = ['优化器', '最终准确率', '训练轮数', '收敛轮数(80%)', 
               '最终训练损失', '最终验证损失', '损失下降率']
    
    df = pd.DataFrame(table_data, columns=headers)
    
    # 保存为CSV
    df.to_csv('optimizer_comparison_data.csv', index=False, encoding='utf-8-sig')
    print("✅ 优化器对比数据已保存到: optimizer_comparison_data.csv")
    
    # 创建可视化表格
    fig, ax = plt.subplots(figsize=(14, max(len(table_data) * 0.5 + 2, 6)))
    ax.axis('off')
    
    # 创建表格
    table = ax.table(cellText=table_data,
                     colLabels=headers,
                     loc='center',
                     cellLoc='center',
                     colColours=['#3498db'] * len(headers))
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    
    # 高亮最佳优化器
    if table_data:
        for j in range(len(headers)):
            table[(1, j)].set_facecolor('#2ecc71')  # 绿色高亮第一行（最佳优化器）
            table[(1, j)].set_text_props(weight='bold', color='white')
    
    # 设置标题
    best_optimizer = table_data[0][0] if table_data else "N/A"
    best_accuracy = table_data[0][1] if table_data else "N/A"
    
    plt.title(f'STL-10 优化器性能对比表\n最佳优化器: {best_optimizer} ({best_accuracy})', 
              fontsize=14, fontweight='bold', pad=30)
    
    # 添加脚注
    plt.figtext(0.5, 0.02, 
                f"注：收敛轮数指达到最大准确率80%所需的训练轮数 | 训练模型: ImprovedCNN | 数据: STL-10", 
                ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison_table.png', dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ 优化器对比表格已保存: optimizer_comparison_table.png")
    return df

# 测试代码
if __name__ == "__main__":
    print("✅ utils.py 加载成功！")
    print("包含的函数:")
    print("  - plot_samples()")
    print("  - plot_training_history()")
    print("  - plot_optimizer_comparison()")
    print("  - plot_loss_comparison()")
    print("  - plot_class_accuracy()")
    print("  - plot_training_curves_comparison()")
    print("  - evaluate_model()")
    print("  - setup_optimizer()")
    print("  - print_experiment_summary()")
    print("  - generate_optimizer_table()")
    print(f"STL-10类别: {STL10_class_names}")