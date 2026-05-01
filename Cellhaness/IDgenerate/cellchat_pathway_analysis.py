import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CellChatPathwayAnalyzer:
    """详细的受配体-通路分析器"""
    
    def __init__(self):
        self.lr_database = self._build_lr_database()
    
    def _build_lr_database(self):
        """构建受配体-通路数据库"""
        lr_data = [
            # 免疫相关通路
            {'ligand': 'CD40LG', 'receptor': 'CD40', 'pathway': 'TNFSF', 'function': '免疫激活'},
            {'ligand': 'IFNG', 'receptor': 'IFNGR1', 'pathway': 'IFN', 'function': '抗病毒/免疫调节'},
            {'ligand': 'IL2', 'receptor': 'IL2R', 'pathway': 'IL2', 'function': 'T细胞增殖'},
            {'ligand': 'IL6', 'receptor': 'IL6R', 'pathway': 'IL6', 'function': '炎症/免疫'},
            {'ligand': 'IL10', 'receptor': 'IL10R', 'pathway': 'IL10', 'function': '抗炎'},
            {'ligand': 'TGFB1', 'receptor': 'TGFBR', 'pathway': 'TGFB', 'function': '免疫抑制/纤维化'},
            
            # 趋化因子通路
            {'ligand': 'CXCL12', 'receptor': 'CXCR4', 'pathway': 'CXCL', 'function': '趋化/迁移'},
            {'ligand': 'CXCL10', 'receptor': 'CXCR3', 'pathway': 'CXCL', 'function': '趋化/炎症'},
            {'ligand': 'CCL5', 'receptor': 'CCR5', 'pathway': 'CCL', 'function': '趋化/免疫招募'},
            {'ligand': 'CCL2', 'receptor': 'CCR2', 'pathway': 'CCL', 'function': '单核细胞趋化'},
            
            # 生长因子通路
            {'ligand': 'VEGF', 'receptor': 'VEGFR', 'pathway': 'VEGF', 'function': '血管生成'},
            {'ligand': 'PDGF', 'receptor': 'PDGFR', 'pathway': 'PDGF', 'function': '增殖/迁移'},
            {'ligand': 'EGF', 'receptor': 'EGFR', 'pathway': 'EGF', 'function': '增殖/存活'},
            {'ligand': 'FGF', 'receptor': 'FGFR', 'pathway': 'FGF', 'function': '发育/修复'},
            
            # 神经相关通路
            {'ligand': 'NLGN1', 'receptor': 'NRXN1', 'pathway': 'Neurexin', 'function': '突触形成'},
            {'ligand': 'EPHB2', 'receptor': 'EPHR', 'pathway': 'Ephrin', 'function': '轴突导向'},
            
            # 细胞粘附通路
            {'ligand': 'NCAM1', 'receptor': 'NCAM1', 'pathway': 'CAM', 'function': '细胞粘附'},
            {'ligand': 'ICAM1', 'receptor': 'ITGB2', 'pathway': 'CAM', 'function': '白细胞粘附'},
            
            # TNF超家族
            {'ligand': 'TNFA', 'receptor': 'TNFR1', 'pathway': 'TNFSF', 'function': '炎症/细胞死亡'},
            {'ligand': 'FASL', 'receptor': 'FAS', 'pathway': 'TNFSF', 'function': '凋亡'},
        ]
        
        return pd.DataFrame(lr_data)
    
    def calculate_lr_interaction(self, expression: np.ndarray, lr_pair: dict):
        """计算单个受配体对的相互作用"""
        # 假设我们从表达矩阵中提取配体和受体的表达
        # 这里使用基因索引来模拟
        gene_names = self._get_gene_names()
        
        ligand_idx = gene_names.get(lr_pair['ligand'], np.random.randint(0, 100))
        receptor_idx = gene_names.get(lr_pair['receptor'], np.random.randint(100, 200))
        
        ligand_expr = expression[:, ligand_idx]
        receptor_expr = expression[:, receptor_idx]
        
        # 计算所有细胞对的相互作用
        N = len(ligand_expr)
        interaction = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                if i != j:  # 排除自相互作用
                    interaction[i, j] = ligand_expr[i] * receptor_expr[j]
        
        return interaction
    
    def _get_gene_names(self):
        """生成模拟基因名称映射"""
        genes = {}
        for _, row in self.lr_database.iterrows():
            genes[row['ligand']] = hash(row['ligand']) % 978
            genes[row['receptor']] = hash(row['receptor']) % 978
        return genes
    
    def analyze_cell_type_interactions(self, expression: np.ndarray, cell_labels: np.ndarray):
        """分析细胞类型水平的受配体相互作用"""
        cell_types = ['Microglia', 'Astrocyte', 'B_cell', 'T_cell']
        results = []
        
        for _, lr_pair in self.lr_database.iterrows():
            interaction = self.calculate_lr_interaction(expression, lr_pair)
            
            # 按细胞类型聚合
            agg_matrix = np.zeros((4, 4))
            for i, source_type in enumerate(cell_types):
                for j, target_type in enumerate(cell_types):
                    source_mask = cell_labels == source_type
                    target_mask = cell_labels == target_type
                    sub_matrix = interaction[np.ix_(source_mask, target_mask)]
                    agg_matrix[i, j] = np.mean(sub_matrix[sub_matrix > 0]) if np.any(sub_matrix > 0) else 0
            
            # 计算总体相互作用强度
            total_strength = np.mean(agg_matrix)
            
            results.append({
                'ligand': lr_pair['ligand'],
                'receptor': lr_pair['receptor'],
                'pathway': lr_pair['pathway'],
                'function': lr_pair['function'],
                'total_strength': total_strength,
                'interaction_matrix': agg_matrix,
                'cell_types': cell_types
            })
        
        return pd.DataFrame(results)
    
    def identify_significant_lr_pairs(self, results_df: pd.DataFrame, threshold: float = None):
        """识别显著的受配体对"""
        if threshold is None:
            threshold = np.percentile(results_df['total_strength'], 75)
        
        significant = results_df[results_df['total_strength'] >= threshold].copy()
        significant = significant.sort_values('total_strength', ascending=False)
        
        return significant
    
    def get_pathway_summary(self, results_df: pd.DataFrame):
        """按通路汇总"""
        pathway_summary = results_df.groupby('pathway').agg({
            'total_strength': ['mean', 'max', 'count'],
            'function': lambda x: ', '.join(x.unique())
        }).reset_index()
        
        pathway_summary.columns = ['pathway', 'mean_strength', 'max_strength', 'lr_count', 'functions']
        pathway_summary = pathway_summary.sort_values('mean_strength', ascending=False)
        
        return pathway_summary
    
    def plot_significant_lr_pairs(self, significant_df: pd.DataFrame, save_path: str = None):
        """可视化显著受配体对"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 按通路分组
        pathways = significant_df['pathway'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(pathways)))
        pathway_color_map = dict(zip(pathways, colors))
        
        # 绘制气泡图
        for _, row in significant_df.iterrows():
            ax.scatter(
                x=row['ligand'],
                y=row['receptor'],
                s=row['total_strength'] * 1000,
                c=pathway_color_map[row['pathway']],
                alpha=0.7,
                edgecolors='black',
                linewidth=1,
                label=row['pathway']
            )
        
        ax.set_xlabel('Ligands', fontsize=12)
        ax.set_ylabel('Receptors', fontsize=12)
        ax.set_title('Significant Ligand-Receptor Pairs (Bubble Size = Interaction Strength)', fontsize=14)
        
        # 添加图例（按通路）
        handles = []
        for pathway in pathways:
            handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=pathway_color_map[pathway], 
                                    markersize=10, label=pathway))
        ax.legend(handles=handles, title='Pathway', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"LR pairs plot saved to {save_path}")
        
        plt.show()
    
    def plot_pathway_heatmap(self, significant_df: pd.DataFrame, save_path: str = None):
        """绘制通路热图"""
        cell_types = ['Microglia', 'Astrocyte', 'B_cell', 'T_cell']
        
        # 按通路聚合
        pathway_interactions = {}
        
        for _, row in significant_df.iterrows():
            pathway = row['pathway']
            if pathway not in pathway_interactions:
                pathway_interactions[pathway] = np.zeros((4, 4))
            pathway_interactions[pathway] += row['interaction_matrix']
        
        # 创建综合热图
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, (pathway, matrix) in enumerate(pathway_interactions.items()):
            if i < len(axes):
                ax = axes[i]
                sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Reds', 
                            xticklabels=cell_types, yticklabels=cell_types, ax=ax)
                ax.set_title(f'{pathway} Pathway', fontsize=12)
        
        # 隐藏多余的子图
        for i in range(len(pathway_interactions), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Cell-Cell Interactions by Signaling Pathway', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Pathway heatmap saved to {save_path}")
        
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
    
    # 初始化分析器
    analyzer = CellChatPathwayAnalyzer()
    
    print(f"\nLoaded {len(analyzer.lr_database)} ligand-receptor pairs")
    print("Available pathways:", analyzer.lr_database['pathway'].unique())
    
    # 分析受配体相互作用
    print("\nAnalyzing ligand-receptor interactions...")
    results_df = analyzer.analyze_cell_type_interactions(expression, cell_labels)
    
    # 识别显著受配体对
    print("\nIdentifying significant LR pairs...")
    significant_df = analyzer.identify_significant_lr_pairs(results_df)
    print(f"Found {len(significant_df)} significant LR pairs")
    
    print("\n=== Significant Ligand-Receptor Pairs ===")
    for _, row in significant_df.iterrows():
        print(f"\n{row['ligand']} -> {row['receptor']}")
        print(f"  Pathway: {row['pathway']}")
        print(f"  Function: {row['function']}")
        print(f"  Interaction Strength: {row['total_strength']:.4f}")
    
    # 通路汇总
    print("\n=== Pathway Summary ===")
    pathway_summary = analyzer.get_pathway_summary(results_df)
    print(pathway_summary.round(4))
    
    # 可视化
    print("\nGenerating visualizations...")
    save_dir = '/home/wupf_260213/controlnet/Cellhaness/IDgenerate'
    
    analyzer.plot_significant_lr_pairs(
        significant_df,
        save_path=os.path.join(save_dir, 'significant_lr_pairs.png')
    )
    
    analyzer.plot_pathway_heatmap(
        significant_df,
        save_path=os.path.join(save_dir, 'pathway_heatmap.png')
    )
    
    # 保存分析结果
    output_df = significant_df[['ligand', 'receptor', 'pathway', 'function', 'total_strength']]
    output_df.to_csv(os.path.join(save_dir, 'lr_analysis_results.csv'), index=False)
    print(f"\nAnalysis results saved to {save_dir}/lr_analysis_results.csv")
    
    print("\nCellChat pathway analysis complete!")

if __name__ == '__main__':
    main()
