import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CellCellCommunicationAnalyzer:
    """详细分析细胞间通讯：配体来源细胞 → 受体目标细胞"""
    
    def __init__(self):
        self.lr_database = self._load_cellchat_db()
    
    def _load_cellchat_db(self):
        """加载CellChat人类数据库的核心受配体对"""
        lr_data = [
            # 免疫相关
            ('IL2', 'IL2R', 'Cytokine', 'T细胞增殖'),
            ('IL2', 'IL2RB', 'Cytokine', 'T细胞增殖'),
            ('IFNG', 'IFNGR1', 'Cytokine', '免疫调节'),
            ('IL6', 'IL6R', 'Cytokine', '炎症'),
            ('IL10', 'IL10RA', 'Cytokine', '抗炎'),
            ('TGFB1', 'TGFBR1', 'Cytokine', '免疫抑制'),
            
            # TNFSF家族
            ('CD40LG', 'CD40', 'TNFSF', 'B细胞活化'),
            ('TNFA', 'TNFR1', 'TNFSF', '炎症'),
            ('FASL', 'FAS', 'TNFSF', '凋亡'),
            
            # 趋化因子
            ('CXCL12', 'CXCR4', 'Chemokine', '趋化'),
            ('CXCL10', 'CXCR3', 'Chemokine', '趋化'),
            ('CCL5', 'CCR5', 'Chemokine', '趋化'),
            ('CCL2', 'CCR2', 'Chemokine', '趋化'),
            
            # 生长因子
            ('EGF', 'EGFR', 'GrowthFactor', '增殖'),
            ('VEGFA', 'VEGFR2', 'GrowthFactor', '血管生成'),
            ('PDGFA', 'PDGFRA', 'GrowthFactor', '增殖'),
            
            # 免疫检查点
            ('CD28', 'CD80', 'IgSF', 'T细胞共刺激'),
            ('PDCD1', 'PDL1', 'IgSF', '免疫检查点'),
            ('CTLA4', 'CD86', 'IgSF', '免疫检查点'),
            
            # 细胞粘附
            ('NCAM1', 'NCAM1', 'CAM', '粘附'),
            ('ICAM1', 'ITGB2', 'CAM', '白细胞粘附'),
            ('CDH2', 'CDH2', 'Cadherin', '细胞粘附'),
            
            # 神经相关
            ('NLGN1', 'NRXN1', 'Neurexin', '突触形成'),
            ('EPHB2', 'EPHA4', 'Ephrin', '轴突导向'),
        ]
        
        return pd.DataFrame(lr_data, columns=['ligand', 'receptor', 'pathway', 'function'])
    
    def analyze_cell_communication(self, expression: np.ndarray, cell_labels: np.ndarray):
        """分析细胞间通讯：源细胞(配体) → 目标细胞(受体)"""
        cell_types = ['Microglia', 'Astrocyte', 'B_cell', 'T_cell']
        n_types = len(cell_types)
        
        # 存储所有通讯事件
        communication_events = []
        
        for _, row in self.lr_database.iterrows():
            ligand = row['ligand']
            receptor = row['receptor']
            pathway = row['pathway']
            function = row['function']
            
            # 获取基因索引（模拟）
            ligand_idx = hash(ligand) % 978
            receptor_idx = hash(receptor) % 978
            
            ligand_expr = expression[:, ligand_idx]
            receptor_expr = expression[:, receptor_idx]
            
            # 计算细胞类型水平的相互作用
            for source_type in cell_types:
                for target_type in cell_types:
                    # 获取源细胞和目标细胞的表达
                    source_mask = cell_labels == source_type
                    target_mask = cell_labels == target_type
                    
                    source_ligand = ligand_expr[source_mask]
                    target_receptor = receptor_expr[target_mask]
                    
                    # 计算相互作用强度
                    if len(source_ligand) > 0 and len(target_receptor) > 0:
                        # 平均配体表达 × 平均受体表达
                        interaction_strength = np.mean(source_ligand) * np.mean(target_receptor)
                        
                        communication_events.append({
                            'ligand': ligand,
                            'receptor': receptor,
                            'pathway': pathway,
                            'function': function,
                            'source_cell_type': source_type,
                            'target_cell_type': target_type,
                            'source_ligand_expr': float(np.mean(source_ligand)),
                            'target_receptor_expr': float(np.mean(target_receptor)),
                            'interaction_strength': float(interaction_strength)
                        })
        
        return pd.DataFrame(communication_events)
    
    def get_top_communication_events(self, events_df: pd.DataFrame, top_n: int = 30):
        """获取最强的通讯事件"""
        return events_df.sort_values('interaction_strength', ascending=False).head(top_n)
    
    def get_cell_type_communication_summary(self, events_df: pd.DataFrame):
        """按细胞类型对汇总通讯"""
        summary = events_df.groupby(['source_cell_type', 'target_cell_type']).agg({
            'interaction_strength': ['sum', 'mean', 'count'],
            'pathway': lambda x: ', '.join(x.unique()[:5])
        }).reset_index()
        
        summary.columns = ['source', 'target', 'total_strength', 'mean_strength', 'event_count', 'pathways']
        summary = summary.sort_values('total_strength', ascending=False)
        
        return summary
    
    def get_ligand_source_summary(self, events_df: pd.DataFrame):
        """分析哪些细胞表达哪些配体"""
        ligand_summary = events_df.groupby(['source_cell_type', 'ligand']).agg({
            'source_ligand_expr': 'mean',
            'interaction_strength': 'sum'
        }).reset_index()
        
        ligand_summary = ligand_summary.sort_values('interaction_strength', ascending=False)
        
        return ligand_summary
    
    def get_receptor_target_summary(self, events_df: pd.DataFrame):
        """分析哪些细胞表达哪些受体"""
        receptor_summary = events_df.groupby(['target_cell_type', 'receptor']).agg({
            'target_receptor_expr': 'mean',
            'interaction_strength': 'sum'
        }).reset_index()
        
        receptor_summary = receptor_summary.sort_values('interaction_strength', ascending=False)
        
        return receptor_summary
    
    def plot_communication_network(self, summary_df: pd.DataFrame, save_path: str = None):
        """绘制细胞间通讯网络图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        cell_types = ['Microglia', 'Astrocyte', 'B_cell', 'T_cell']
        type_positions = {
            'Microglia': (0.2, 0.8),
            'Astrocyte': (0.8, 0.8),
            'B_cell': (0.2, 0.2),
            'T_cell': (0.8, 0.2)
        }
        
        # 绘制节点
        for cell_type, (x, y) in type_positions.items():
            ax.scatter(x, y, s=500, c='lightblue', edgecolors='black', zorder=5)
            ax.text(x, y, cell_type, ha='center', va='center', fontsize=12, zorder=6)
        
        # 绘制通讯边
        for _, row in summary_df.iterrows():
            source_pos = type_positions[row['source']]
            target_pos = type_positions[row['target']]
            
            # 边的宽度和颜色基于通讯强度
            line_width = row['total_strength'] * 5
            color_intensity = min(row['total_strength'] / summary_df['total_strength'].max(), 1)
            
            ax.plot(
                [source_pos[0], target_pos[0]],
                [source_pos[1], target_pos[1]],
                'k-',
                linewidth=line_width,
                alpha=0.3 + color_intensity * 0.7,
                zorder=1
            )
        
        ax.set_title('Cell-Cell Communication Network', fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Network plot saved to {save_path}")
        
        plt.show()
    
    def plot_communication_matrix(self, summary_df: pd.DataFrame, save_path: str = None):
        """绘制细胞间通讯矩阵热图"""
        cell_types = ['Microglia', 'Astrocyte', 'B_cell', 'T_cell']
        
        # 创建4x4矩阵
        matrix = np.zeros((4, 4))
        
        for _, row in summary_df.iterrows():
            i = cell_types.index(row['source'])
            j = cell_types.index(row['target'])
            matrix[i, j] = row['total_strength']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Reds',
                    xticklabels=cell_types, yticklabels=cell_types, ax=ax)
        
        ax.set_xlabel('Target Cell Type (Receptor)', fontsize=12)
        ax.set_ylabel('Source Cell Type (Ligand)', fontsize=12)
        ax.set_title('Cell-Cell Communication Strength Matrix', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Matrix plot saved to {save_path}")
        
        plt.show()

def load_cell_data(group_idx: int = 0, data_dir: str = '/tmp/lrldm_data') -> dict:
    """加载细胞数据"""
    file_path = os.path.join(data_dir, f"{group_idx}_0.pt")
    return torch.load(file_path, map_location='cpu')

def create_cell_labels(n_per_type: int = 100) -> np.ndarray:
    """创建细胞类型标签"""
    return np.array(['Microglia'] * n_per_type + ['Astrocyte'] * n_per_type + 
                    ['B_cell'] * n_per_type + ['T_cell'] * n_per_type)

def main():
    """主函数"""
    print("Loading cell data...")
    data = load_cell_data(0)
    expression = data['z0']['expression'][0].detach().numpy()
    cell_labels = create_cell_labels(100)
    
    print(f"Expression shape: {expression.shape}")
    print(f"Cell types: {np.unique(cell_labels)}")
    
    # 初始化分析器
    analyzer = CellCellCommunicationAnalyzer()
    
    # 分析细胞间通讯
    print("\nAnalyzing cell-cell communication...")
    events_df = analyzer.analyze_cell_communication(expression, cell_labels)
    
    print(f"Total communication events: {len(events_df)}")
    
    # 获取最强通讯事件
    top_events = analyzer.get_top_communication_events(events_df, top_n=20)
    
    print("\n=== Top 20 Communication Events ===")
    print("Source(配体) → Target(受体) | 受配体对 | 通路 | 功能 | 强度")
    print("-" * 100)
    for _, row in top_events.iterrows():
        print(f"{row['source_cell_type']} → {row['target_cell_type']} | "
              f"{row['ligand']}→{row['receptor']} | {row['pathway']} | "
              f"{row['function']} | {row['interaction_strength']:.4f}")
    
    # 细胞类型对汇总
    print("\n=== Cell Type Communication Summary ===")
    type_summary = analyzer.get_cell_type_communication_summary(events_df)
    print(type_summary[['source', 'target', 'total_strength', 'mean_strength', 'event_count', 'pathways']].to_string(index=False))
    
    # 配体来源分析
    print("\n=== Ligand Expression by Source Cell ===")
    ligand_summary = analyzer.get_ligand_source_summary(events_df).head(10)
    print(ligand_summary[['source_cell_type', 'ligand', 'source_ligand_expr', 'interaction_strength']].to_string(index=False))
    
    # 受体表达分析
    print("\n=== Receptor Expression by Target Cell ===")
    receptor_summary = analyzer.get_receptor_target_summary(events_df).head(10)
    print(receptor_summary[['target_cell_type', 'receptor', 'target_receptor_expr', 'interaction_strength']].to_string(index=False))
    
    # 可视化
    save_dir = '/home/wupf_260213/controlnet/Cellhaness/IDgenerate'
    analyzer.plot_communication_network(type_summary, os.path.join(save_dir, 'cell_communication_network.png'))
    analyzer.plot_communication_matrix(type_summary, os.path.join(save_dir, 'cell_communication_matrix.png'))
    
    # 保存详细结果
    events_df.to_csv(os.path.join(save_dir, 'cell_communication_events.csv'), index=False)
    type_summary.to_csv(os.path.join(save_dir, 'cell_communication_summary.csv'), index=False)
    
    print(f"\nResults saved to {save_dir}/")
    print("\nCell-cell communication analysis complete!")

if __name__ == '__main__':
    main()
