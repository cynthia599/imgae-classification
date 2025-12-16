# train_cpu.py - 完整的训练脚本（支持GPU）
import torch
import torch.nn as nn
from data_loader import get_stl10_dataloaders
from model import ImprovedCNN
from utils import evaluate_model, setup_optimizer, plot_optimizer_comparison, plot_loss_comparison, plot_training_curves_comparison, plot_class_accuracy, generate_optimizer_table, print_experiment_summary, STL10_class_names
import os
import pickle
import traceback
import numpy as np
import matplotlib.pyplot as plt

print("当前工作目录:", os.getcwd())
print("文件保存路径:", os.path.abspath('.'))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def train_single_experiment(optimizer_name='adam', learning_rate=0.001, batch_size=32, epochs=50, momentum=0.9):
    """训练单个优化器实验 - 使用GPU"""
    
    # 自动选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 使用设备: {device}")
    
    if device.type == 'cuda':
        print(f"🚀 GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 获取数据
    train_loader, val_loader, test_loader = get_stl10_dataloaders(batch_size)
    
    model = ImprovedCNN().to(device)
    
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = setup_optimizer(model, optimizer_name, learning_rate, momentum)
    
    print(f"\n开始训练 {optimizer_name} 优化器...")
    print(f"设备: {device}")
    print(f"批次大小: {batch_size}")
    print(f"训练轮数: {epochs}")
    print(f"数据集: STL-10")
    print(f"模型: ImprovedCNN (带批量归一化)")
    print(f"学习率: {learning_rate}")
    if optimizer_name == 'sgd_momentum':
        print(f"动量参数: {momentum}")
    
    # 记录训练过程
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # 训练循环
    for epoch in range(epochs):
        # ========== 训练阶段 ==========
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 每50个batch打印一次进度
            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')
        
        # 计算平均训练损失
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # 计算验证损失和准确率
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        
        # 每5个epoch打印详细进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{epochs}] | 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f} | 验证准确率: {accuracy:.2f}%')
            
            # 显示GPU内存使用情况
            if device.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_cached = torch.cuda.memory_reserved(0) / 1024**3
                print(f"💾 GPU内存 - 已分配: {memory_allocated:.2f}GB, 缓存: {memory_cached:.2f}GB")
    
    # 最终评估使用测试集
    final_accuracy, class_accuracy = evaluate_model(model, test_loader, device)
    
    # 清理GPU缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'final_accuracy': final_accuracy,
        'class_accuracy': class_accuracy,
        'model_type': 'ImprovedCNN',
        'optimizer_name': optimizer_name
    }

def generate_experiment_summary_chart(results, save_path='experiment_summary.png'):
    """生成实验汇总图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    optimizers = list(results.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    # 1. 所有优化器的验证准确率
    for idx, (opt_name, result) in enumerate(results.items()):
        axes[0, 0].plot(result['val_accuracies'], 
                       color=colors[idx % len(colors)],
                       linewidth=2,
                       label=opt_name.upper())
    
    axes[0, 0].set_title('所有优化器验证准确率对比', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('训练轮数')
    axes[0, 0].set_ylabel('准确率 (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 100])
    
    # 2. 所有优化器的训练损失
    for idx, (opt_name, result) in enumerate(results.items()):
        axes[0, 1].plot(result['train_losses'], 
                       color=colors[idx % len(colors)],
                       linewidth=2,
                       label=opt_name.upper())
    
    axes[0, 1].set_title('所有优化器训练损失对比', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('训练轮数')
    axes[0, 1].set_ylabel('损失值')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 最终准确率柱状图
    opt_names = []
    final_accs = []
    for opt_name, result in results.items():
        opt_names.append(opt_name.upper())
        final_accs.append(result['final_accuracy'])
    
    bars = axes[1, 0].bar(opt_names, final_accs, 
                          color=colors[:len(opt_names)], 
                          edgecolor='black')
    axes[1, 0].set_title('最终测试准确率对比', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('准确率 (%)')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, acc in zip(bars, final_accs):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. 收敛速度比较
    convergence_data = []
    for opt_name, result in results.items():
        if 'val_accuracies' in result:
            max_acc = max(result['val_accuracies'])
            target_90 = max_acc * 0.9
            
            convergence_epoch = 0
            for epoch, acc in enumerate(result['val_accuracies']):
                if acc >= target_90:
                    convergence_epoch = epoch + 1
                    break
            if convergence_epoch == 0:
                convergence_epoch = len(result['val_accuracies'])
            
            convergence_data.append((opt_name.upper(), convergence_epoch))
    
    conv_names = [x[0] for x in convergence_data]
    conv_epochs = [x[1] for x in convergence_data]
    
    bars2 = axes[1, 1].bar(conv_names, conv_epochs, 
                           color=colors[:len(conv_names)], 
                           edgecolor='black')
    axes[1, 1].set_title('收敛速度对比 (达到90%最大准确率)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('所需轮数')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, epochs in zip(bars2, conv_epochs):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{epochs}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('STL-10 优化器实验结果汇总', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 实验汇总图已保存: {save_path}")
    return fig

def main():
    """主函数，运行不同优化器的实验"""
    
    # 定义要测试的优化器
    optimizers = ['adam', 'sgd', 'sgd_momentum', 'adagrad', 'rmsprop']
    
    # 为不同优化器设置不同的学习率
    learning_rates = {
        'adam': 0.001,
        'sgd': 0.01,
        'sgd_momentum': 0.01,
        'adagrad': 0.01,
        'rmsprop': 0.001
    }
    
    results = {}
    
    for opt_name in optimizers:
        print(f"\n{'='*50}")
        print(f"训练 {opt_name.upper()} 优化器")
        print(f"{'='*50}")
        
        # 训练单个优化器
        result = train_single_experiment(
            optimizer_name=opt_name,
            learning_rate=learning_rates[opt_name],
            batch_size=32,
            epochs=50,
            momentum=0.9
        )
        
        results[opt_name] = result
        
        # 打印最终结果
        print(f"\n{opt_name.upper()} 优化器最终准确率: {result['final_accuracy']:.2f}%")
    
    # ========== 保存结果部分 ==========
    print("\n" + "="*60)
    print("开始保存训练结果和图表...")
    print("="*60)
    
    try:
        # 1. 创建图表保存目录
        charts_dir = "training_charts"
        os.makedirs(charts_dir, exist_ok=True)
        print(f"📁 创建图表目录: {charts_dir}")
        
        # 2. 绘制优化器比较图
        print("\n📊 生成优化器比较图...")
        plot_optimizer_comparison(results)
        
        # 3. 绘制损失函数对比图
        print("📈 生成损失函数对比图...")
        plot_loss_comparison(results)
        
        # 4. 绘制训练曲线对比图
        print("📈 生成训练曲线对比图...")
        plot_training_curves_comparison(results, 'training_curves.png')
        
        # 5. 为每个优化器绘制详细图表
        print("\n📊 为每个优化器生成详细图表...")
        for opt_name, result in results.items():
            print(f"  正在处理 {opt_name.upper()}...")
            
            # 5.1 训练历史曲线
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            epochs = range(1, len(result['train_losses']) + 1)
            
            # 训练损失和验证损失
            axes[0].plot(epochs, result['train_losses'], 'b-', linewidth=2, label='训练损失')
            axes[0].plot(epochs, result['val_losses'], 'r-', linewidth=2, label='验证损失')
            axes[0].set_title(f'{opt_name.upper()} - 损失曲线', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('训练轮数')
            axes[0].set_ylabel('损失值')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 验证准确率
            axes[1].plot(epochs, result['val_accuracies'], 'g-', linewidth=2)
            axes[1].set_title(f'{opt_name.upper()} - 验证准确率', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('训练轮数')
            axes[1].set_ylabel('准确率 (%)')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim([0, 100])
            
            plt.suptitle(f'{opt_name.upper()} 优化器 - 训练详情', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(f'{charts_dir}/training_history_{opt_name}.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"    ✅ {opt_name}训练历史图已保存: {charts_dir}/training_history_{opt_name}.png")
            
            # 5.2 单独的类别准确率图
            if 'class_accuracy' in result and result['class_accuracy'] is not None:
                plot_class_accuracy(result['class_accuracy'], opt_name, 
                                   f'{charts_dir}/class_accuracy_{opt_name}.png')
                print(f"    ✅ {opt_name}类别准确率图已保存: {charts_dir}/class_accuracy_{opt_name}.png")
        
        # 6. 生成最佳优化器的类别准确率图
        print("\n🏆 生成最佳优化器的详细图表...")
        best_opt = max(results.keys(), key=lambda x: results[x]['final_accuracy'])
        best_result = results[best_opt]
        
        # 6.1 最佳优化器的类别准确率图
        if 'class_accuracy' in best_result and best_result['class_accuracy'] is not None:
            plot_class_accuracy(best_result['class_accuracy'], f"{best_opt}(最佳)", 
                               'class_accuracy.png')
            print(f"✅ 最佳优化器类别准确率图已保存: class_accuracy.png")
        
        # 6.2 创建实验汇总图表
        print("\n📋 创建实验汇总图表...")
        generate_experiment_summary_chart(results, 'experiment_summary.png')
        
        # 7. 生成优化器对比表格
        print("\n📋 生成优化器对比表格...")
        generate_optimizer_table(results)
        
        # 8. 比较不同优化器的性能
        print_experiment_summary(results)
        print("✅ 实验总结已生成")
        
        # 9. 保存结果到文件
        print("\n💾 保存训练数据...")
        with open('training_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print("✅ 训练数据已保存到: training_results.pkl")
        
        print("\n" + "="*60)
        print("🎉 所有训练结果和图表保存完成！")
        print("="*60)
        print("\n📊 生成的文件列表:")
        print("├── accuracy_comparison.png (优化器准确率对比)")
        print("├── loss_comparison.png (损失函数对比)")
        print("├── training_curves.png (训练曲线对比)")
        print("├── class_accuracy.png (最佳优化器类别准确率)")
        print("├── experiment_summary.png (实验汇总图)")
        print("├── optimizer_comparison_table.png (优化器对比表)")
        print("├── optimizer_comparison_data.csv (优化器对比数据)")
        print("├── training_results.pkl (训练数据)")
        print("└── training_charts/ (各优化器详细图表目录)")
        print("    ├── training_history_[optimizer].png (各优化器训练历史)")
        print("    └── class_accuracy_[optimizer].png (各优化器类别准确率)")
        
        # 列出实际生成的文件
        print(f"\n📁 当前目录生成的文件:")
        file_count = 0
        for f in sorted(os.listdir('.')):
            if f.endswith('.png') or f.endswith('.pkl') or f.endswith('.csv'):
                file_size = os.path.getsize(f) / 1024  # KB
                print(f"   - {f:35} ({file_size:.1f} KB)")
                file_count += 1
        
        print(f"\n📁 {charts_dir} 目录中的文件:")
        if os.path.exists(charts_dir):
            chart_count = 0
            for f in sorted(os.listdir(charts_dir)):
                if f.endswith('.png'):
                    file_size = os.path.getsize(os.path.join(charts_dir, f)) / 1024
                    print(f"   - {charts_dir}/{f:30} ({file_size:.1f} KB)")
                    chart_count += 1
            print(f"  总共 {chart_count} 个图表文件")
        
        print(f"\n✅ 总共生成 {file_count} 个主文件和 {chart_count if 'chart_count' in locals() else 0} 个图表文件")
        
    except Exception as e:
        print(f"\n❌ 保存失败: {e}")
        print("详细错误信息:")
        traceback.print_exc()
    
    return results

if __name__ == "__main__":
    # 检查GPU状态
    if torch.cuda.is_available():
        print("🎉 使用GPU进行训练!")
        print(f"🚀 GPU设备: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ 使用CPU进行训练 (GPU不可用)")
    
    print("测试的优化器包括: Adam, SGD, SGD with Momentum, Adagrad, RMSprop")
    print("使用的模型: ImprovedCNN (带批量归一化)")
    print("训练轮数: 50 epochs")
    print("开始训练...")
    
    results = main()