import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CellChatAnalyzer:
    """模拟CellChat分析流程"""
    
    def __init__(self):
        self.ligand_receptor_pairs = self._load_lr_pairs()
    
    def _load_lr_pairs(self):
        """加载受配体对数据库（模拟）"""
        lr_pairs = pd.DataFrame({
            'ligand': ['CD40LG', 'IFNG', 'IL2', 'IL6', 'CXCL12', 'CCL5', 'VEGF', 'PDGF', 'TGFB', 'EGF'],
            'receptor': ['CD40', 'IFNGR1', 'IL2R', 'IL6R', 'CXCR4', 'CCR5', 'VEGFR', 'PDGFR', 'TGFBR', 'EGFR'],
            'family': ['TNFSF', 'IFN', 'IL2', 'IL6', 'CXCL', 'CCL', 'VEGF', 'PDGF', 'TGFB', 'EGF']
        })
        return lr_pairs
    
    def analyze_lr_interactions(self, expression_matrix: np.ndarray, cell_labels: np.ndarray):
        """分析受配体相互作用"""
        cell_types = ['Microglia', 'Astrocyte', 'B_cell', 'T_cell']
        n_cells, n_genes = expression_matrix.shape
        
        # 创建虚拟的受配体基因表达
        # 假设前20个基因是配体，接下来20个是受体
        ligand_expr = expression_matrix[:, :10]  # (N, 10)
        receptor_expr = expression_matrix[:, 10:20]  # (N, 10)
        
        # 计算细胞间受配体相互作用
        interaction_scores = []
        
        for lr_idx in range(len(self.ligand_receptor_pairs)):
            ligand = ligand_expr[:, lr_idx]  # (N,)
            receptor = receptor_expr[:, lr_idx]  # (N,)
            
            # 计算所有细胞对的相互作用
            # interaction[i,j] = ligand[i] * receptor[j]
            ligand_expanded = ligand.reshape(-1, 1)  # (N, 1)
            receptor_expanded = receptor.reshape(1, -1)  # (1, N)
            interaction = ligand_expanded @ receptor_expanded  # (N, N)
            
            interaction_scores.append(interaction)
        
        # 聚合所有受配体对的相互作用
        interaction_matrix = np.mean(np.stack(interaction_scores), axis=0)
        
        # 按细胞类型聚合
        agg_matrix = np.zeros((4, 4))
        for i, source_type in enumerate(cell_types):
            for j, target_type in enumerate(cell_types):
                source_mask = cell_labels == source_type
                target_mask = cell_labels == target_type
                sub_matrix = interaction_matrix[np.ix_(source_mask, target_mask)]
                agg_matrix[i, j] = sub_matrix.mean()
        
        return pd.DataFrame(agg_matrix, index=cell_types, columns=cell_types), interaction_matrix
    
    def nmf_decomposition(self, interaction_matrix: np.ndarray, n_components: int = 5):
        """使用NMF分解相互作用矩阵"""
        # 确保非负
        interaction_matrix = np.abs(interaction_matrix)
        
        # NMF分解
        nmf = NMF(n_components=n_components, random_state=42, max_iter=1000)
        W = nmf.fit_transform(interaction_matrix)  # (N, K) 细胞因子贡献
        H = nmf.components_  # (K, N) 因子模式
        
        # 解释方差
        explained_variance = nmf.reconstruction_err_
        
        return W, H, nmf, explained_variance
    
    def identify_signaling_patterns(self, W: np.ndarray, H: np.ndarray, cell_labels: np.ndarray):
        """识别信号模式"""
        cell_types = ['Microglia', 'Astrocyte', 'B_cell', 'T_cell']
        n_components = W.shape[1]
        
        # 分析每个因子的细胞类型偏好
        patterns = []
        
        for k in range(n_components):
            # 计算每个细胞类型在该因子上的平均得分
            type_scores = []
            for cell_type in cell_types:
                mask = cell_labels == cell_type
                type_scores.append(W[mask, k].mean())
            
            # 找到主要贡献的细胞类型
            main_source = cell_types[np.argmax(type_scores)]
            
            # 分析目标偏好
            target_scores = H[k, :]
            target_type_scores = []
            for cell_type in cell_types:
                mask = cell_labels == cell_type
                target_type_scores.append(target_scores[mask].mean())
            main_target = cell_types[np.argmax(target_type_scores)]
            
            patterns.append({
                'factor': k + 1,
                'main_source': main_source,
                'main_target': main_target,
                'source_scores': dict(zip(cell_types, type_scores)),
                'target_scores': dict(zip(cell_types, target_type_scores))
            })
        
        return patterns
    
    def plot_nmf_factors(self, W: np.ndarray, H: np.ndarray, cell_labels: np.ndarray, save_path: str = None):
        """可视化NMF因子"""
        cell_types = ['Microglia', 'Astrocyte', 'B_cell', 'T_cell']
        n_components = W.shape[1]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 因子贡献（W矩阵）
        w_df = pd.DataFrame(W, index=range(len(cell_labels)))
        w_df['cell_type'] = cell_labels
        
        for k in range(n_components):
            ax = axes[0]
            for cell_type in cell_types:
                mask = w_df['cell_type'] == cell_type
                ax.hist(w_df.loc[mask, k], alpha=0.5, label=f'{cell_type} (factor {k+1})', bins=20)
        ax.set_xlabel('Factor Contribution Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('NMF Factor Contributions by Cell Type', fontsize=14)
        ax.legend()
        
        # 2. 因子模式（H矩阵）
        h_df = pd.DataFrame(H, columns=range(len(cell_labels)))
        h_df['factor'] = [f'Factor {k+1}' for k in range(n_components)]
        
        ax = axes[1]
        sns.heatmap(H, cmap='viridis', ax=ax)
        ax.set_xlabel('Target Cells', fontsize=12)
        ax.set_ylabel('Factors', fontsize=12)
        ax.set_title('NMF Factor Patterns (Target Preferences)', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"NMF visualization saved to {save_path}")
        
        plt.show()
    
    def plot_signaling_patterns(self, patterns: list, save_path: str = None):
        """可视化信号模式"""
        fig, axes = plt.subplots(len(patterns), 2, figsize=(12, 3 * len(patterns)))
        
        for i, pattern in enumerate(patterns):
            ax1 = axes[i, 0] if len(patterns) > 1 else axes[0]
            ax2 = axes[i, 1] if len(patterns) > 1 else axes[1]
            
            # 源细胞贡献
            ax1.bar(pattern['source_scores'].keys(), pattern['source_scores'].values(), color='skyblue')
            ax1.set_title(f'Factor {pattern["factor"]} - Source Contribution', fontsize=10)
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='x', rotation=45)
            
            # 目标细胞偏好
            ax2.bar(pattern['target_scores'].keys(), pattern['target_scores'].values(), color='salmon')
            ax2.set_title(f'Factor {pattern["factor"]} - Target Preference', fontsize=10)
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Signaling patterns saved to {save_path}")
        
        plt.show()

def load_cell_data(group_idx: int = 0, data_dir: str = '/tmp/lrldm_data') -> dict:
    """加载细胞数据"""
    file_path = os.path.join(data_dir, f"{group_idx}_0.pt")
    return torch.load(file_path, map_location='cpu')

def create_cell_type_labels(n_per_type: int = 100) -> np.ndarray:
    """创建细胞类型标签"""
    labels = []
    cell_types = ['Microglia', 'Astrocyte', 'B_cell', 'T_cell']
    for cell_type in cell_types:
        labels.extend([cell_type] * n_per_type)
    return np.array(labels)

def main():
    """主函数"""
    print("Loading cell data...")
    data = load_cell_data(0)
    
    # 提取表达数据
    expression = data['z0']['expression'][0].detach().numpy()
    cell_labels = create_cell_type_labels(100)
    
    print(f"Expression shape: {expression.shape}")
    print(f"Number of cells: {len(cell_labels)}")
    
    # 初始化CellChat分析器
    analyzer = CellChatAnalyzer()
    
    # 1. 分析受配体相互作用
    print("\nAnalyzing ligand-receptor interactions...")
    lr_matrix, interaction_matrix = analyzer.analyze_lr_interactions(expression, cell_labels)
    print("\nCell Type Interaction Matrix:")
    print(lr_matrix.round(4))
    
    # 2. NMF分解
    print("\nPerforming NMF decomposition...")
    n_components = 5
    W, H, nmf, recon_error = analyzer.nmf_decomposition(interaction_matrix, n_components=n_components)
    print(f"NMF Reconstruction Error: {recon_error:.6f}")
    
    # 3. 识别信号模式
    print("\nIdentifying signaling patterns...")
    patterns = analyzer.identify_signaling_patterns(W, H, cell_labels)
    
    print("\nSignaling Patterns:")
    for pattern in patterns:
        print(f"\nFactor {pattern['factor']}:")
        print(f"  Main Source: {pattern['main_source']}")
        print(f"  Main Target: {pattern['main_target']}")
        print(f"  Source Contributions: {pattern['source_scores']}")
        print(f"  Target Preferences: {pattern['target_scores']}")
    
    # 4. 可视化
    print("\nGenerating visualizations...")
    save_dir = '/home/wupf_260213/controlnet/Cellhaness/IDgenerate'
    
    # NMF因子可视化
    analyzer.plot_nmf_factors(
        W, H, cell_labels,
        save_path=os.path.join(save_dir, 'nmf_factors.png')
    )
    
    # 信号模式可视化
    analyzer.plot_signaling_patterns(
        patterns,
        save_path=os.path.join(save_dir, 'signaling_patterns.png')
    )
    
    # 相互作用热图
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(lr_matrix, annot=True, fmt='.4f', cmap='Reds', ax=ax)
    ax.set_title('Cell-Cell Interaction Matrix (CellChat-like Analysis)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cellchat_interaction_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nCellChat-like analysis complete!")

if __name__ == '__main__':
    main()
