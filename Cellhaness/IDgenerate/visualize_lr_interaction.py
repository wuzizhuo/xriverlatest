import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_cell_data(group_idx: int = 0, data_dir: str = '/tmp/lrldm_data') -> dict:
    """加载细胞数据"""
    file_path = os.path.join(data_dir, f"{group_idx}_0.pt")
    return torch.load(file_path, map_location='cpu')

def create_cell_type_labels(n_per_type: int = 100) -> np.ndarray:
    """创建细胞类型标签"""
    labels = []
    cell_types = ['Microglia', 'Astrocyte', 'B_cell', 'T_cell']
    for i, cell_type in enumerate(cell_types):
        labels.extend([cell_type] * n_per_type)
    return np.array(labels)

def aggregate_interactions_by_type(lr_matrix: np.ndarray, cell_labels: np.ndarray) -> pd.DataFrame:
    """按细胞类型聚合相互作用矩阵"""
    cell_types = ['Microglia', 'Astrocyte', 'B_cell', 'T_cell']
    
    # 创建聚合矩阵
    agg_matrix = np.zeros((4, 4))
    
    for i, source_type in enumerate(cell_types):
        for j, target_type in enumerate(cell_types):
            source_mask = cell_labels == source_type
            target_mask = cell_labels == target_type
            
            # 提取source_type到target_type的相互作用
            sub_matrix = lr_matrix[np.ix_(source_mask, target_mask)]
            agg_matrix[i, j] = sub_matrix.mean()
    
    # 创建DataFrame
    df = pd.DataFrame(agg_matrix, index=cell_types, columns=cell_types)
    return df

def plot_interaction_dotplot(df: pd.DataFrame, save_path: str = None):
    """绘制细胞互作dotplot"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 准备数据
    cell_types = df.index.tolist()
    data_for_plot = []
    
    for i, source in enumerate(cell_types):
        for j, target in enumerate(cell_types):
            data_for_plot.append({
                'Source': source,
                'Target': target,
                'Interaction': df.iloc[i, j]
            })
    
    plot_df = pd.DataFrame(data_for_plot)
    
    # 使用scatter plot创建dotplot
    scatter = ax.scatter(
        x=plot_df['Target'],
        y=plot_df['Source'],
        s=plot_df['Interaction'] * 500,  # 点的大小
        c=plot_df['Interaction'],  # 点的颜色
        cmap='Reds',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Interaction Strength', fontsize=12)
    
    # 设置坐标轴
    ax.set_xlabel('Target Cell Type', fontsize=12)
    ax.set_ylabel('Source Cell Type', fontsize=12)
    ax.set_title('Cell-Cell Ligand-Receptor Interactions\n(Hill Function Based)', fontsize=14)
    
    # 设置刻度
    ax.set_xticks(range(len(cell_types)))
    ax.set_xticklabels(cell_types, rotation=45, ha='right')
    ax.set_yticks(range(len(cell_types)))
    ax.set_yticklabels(cell_types)
    
    # 添加网格
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Dotplot saved to {save_path}")
    
    plt.show()
    return fig

def plot_interaction_heatmap(df: pd.DataFrame, save_path: str = None):
    """绘制细胞互作热图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 使用seaborn绘制热图
    sns.heatmap(df, annot=True, fmt='.4f', cmap='Reds', 
                ax=ax, cbar_kws={'label': 'Interaction Strength'})
    
    ax.set_xlabel('Target Cell Type', fontsize=12)
    ax.set_ylabel('Source Cell Type', fontsize=12)
    ax.set_title('Cell-Cell Ligand-Receptor Interaction Matrix\n(Hill Function Based)', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    
    plt.show()
    return fig

def plot_interaction_distribution(lr_matrix: np.ndarray, cell_labels: np.ndarray, save_path: str = None):
    """绘制相互作用强度分布"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 整体分布
    axes[0].hist(lr_matrix.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Interaction Strength', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of All Interactions', fontsize=14)
    axes[0].axvline(np.mean(lr_matrix), color='red', linestyle='--', label=f'Mean: {np.mean(lr_matrix):.4f}')
    axes[0].legend()
    
    # 2. 按细胞类型对的分布
    cell_types = ['Microglia', 'Astrocyte', 'B_cell', 'T_cell']
    type_pair_data = []
    type_pair_labels = []
    
    for i, source in enumerate(cell_types):
        for j, target in enumerate(cell_types):
            source_mask = cell_labels == source
            target_mask = cell_labels == target
            sub_matrix = lr_matrix[np.ix_(source_mask, target_mask)]
            type_pair_data.append(sub_matrix.flatten())
            type_pair_labels.append(f'{source[:3]}->{target[:3]}')
    
    bp = axes[1].boxplot(type_pair_data, labels=type_pair_labels, patch_artist=True)
    
    colors = plt.cm.Set3(np.linspace(0, 1, 16))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[1].set_xlabel('Cell Type Pair', fontsize=12)
    axes[1].set_ylabel('Interaction Strength', fontsize=12)
    axes[1].set_title('Interaction Distribution by Cell Type Pairs', fontsize=14)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Distribution plot saved to {save_path}")
    
    plt.show()
    return fig

def main():
    """主函数"""
    print("Loading cell data...")
    data = load_cell_data(0)
    
    # 提取相互作用矩阵
    lr_matrix = data['lr_data']['interaction_matrix'].detach().numpy()
    print(f"LR interaction matrix shape: {lr_matrix.shape}")
    
    # 创建细胞类型标签
    cell_labels = create_cell_type_labels(100)
    
    # 按细胞类型聚合
    print("\nAggregating interactions by cell type...")
    agg_df = aggregate_interactions_by_type(lr_matrix, cell_labels)
    print("\nCell Type Interaction Matrix:")
    print(agg_df)
    print(f"\nMean interaction: {agg_df.values.mean():.6f}")
    print(f"Max interaction: {agg_df.values.max():.6f}")
    print(f"Min interaction: {agg_df.values.min():.6f}")
    
    # 绘制dotplot
    print("\nGenerating dotplot...")
    save_dir = '/home/wupf_260213/controlnet/Cellhaness/IDgenerate'
    
    plot_interaction_dotplot(
        agg_df, 
        save_path=os.path.join(save_dir, 'cell_interaction_dotplot.png')
    )
    
    # 绘制热图
    print("\nGenerating heatmap...")
    plot_interaction_heatmap(
        agg_df,
        save_path=os.path.join(save_dir, 'cell_interaction_heatmap.png')
    )
    
    # 绘制分布图
    print("\nGenerating distribution plot...")
    plot_interaction_distribution(
        lr_matrix,
        cell_labels,
        save_path=os.path.join(save_dir, 'cell_interaction_distribution.png')
    )
    
    print("\nVisualization complete!")

if __name__ == '__main__':
    main()
