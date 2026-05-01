import torch
import numpy as np
import os
import sys
from typing import Dict, List

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MultiCellTypeGenerator:
    """
    生成多种细胞类型的GSVA数据
    包含：小胶质细胞、星型胶质细胞、B细胞、T细胞
    """
    
    # Hallmark通路名称（50个）
    HALLMARK_PATHWAYS = [
        'Hallmark_E2F_Targets',
        'Hallmark_G2M_Checkpoint',
        'Hallmark_Epithelial_Mesenchymal_Transition',
        'Hallmark_Glycolysis',
        'Hallmark_Oxidative_Phosphorylation',
        'Hallmark_Hypoxia',
        'Hallmark_P53_Pathway',
        'Hallmark_Apoptosis',
        'Hallmark_Inflammatory_Response',
        'Hallmark_Interferon_Gamma_Response',
        'Hallmark_KRAS_Signaling_Up',
        'Hallmark_KRAS_Signaling_Down',
        'Hallmark_MYC_Targets_V1',
        'Hallmark_MYC_Targets_V2',
        'Hallmark_Unfolded_Protein_Response',
        'Hallmark_Angiogenesis',
        'Hallmark_Invasion_Metastasis',
        'Hallmark_TNFA_Signaling_via_NFKB',
        'Hallmark_IL2_STAT5_Signaling',
        'Hallmark_IL6_JAK_STAT3_Signaling',
        'Hallmark_Hedgehog_Signaling',
        'Hallmark_Wnt_Beta_Catenin_Signaling',
        'Hallmark_Notch_Signaling',
        'Hallmark_Peroxisome',
        'Hallmark_Fatty_Acid_Metabolism',
        'Hallmark_Cholesterol_Homeostasis',
        'Hallmark_Bile_Acid_Metabolism',
        'Hallmark_Xenobiotic_Metabolism',
        'Hallmark_Adipogenesis',
        'Hallmark_Pyruvate_Metabolism',
        'Hallmark_Citrate_Cycle_TCA_Cycle',
        'Hallmark_Glutamate_Metabolism',
        'Hallmark_Biotin_Metabolism',
        'Hallmark_Folate_Metabolism',
        'Hallmark_One_Carbon_Pool_by_Folate',
        'Hallmark_Ribonucleotide_Biosynthesis',
        'Hallmark_Ascorbate_and_Aldarate_Metabolism',
        'Hallmark_Urea_Cycle',
        'Hallmark_Arginine_Proline_Metabolism',
        'Hallmark_Histidine_Metabolism',
        'Hallmark_Tryptophan_Metabolism',
        'Hallmark_Phenylalanine_Metabolism',
        'Hallmark_Tyrosine_Metabolism',
        'Hallmark_Leucine_Isoleucine_Valine_Biosynthesis',
        'Hallmark_Valine_Leucine_Isoleucine_Degradation',
        'Hallmark_Geranylgeranylation',
        'Hallmark_Steroid_Biosynthesis',
        'Hallmark_Androgen_Estrogen_Metabolism',
        'Hallmark_Myelination'
    ]
    
    def __init__(self, n_pathways: int = 50):
        """
        初始化生成器
        
        Args:
            n_pathways: GSVA通路数量（默认50）
        """
        if n_pathways > len(self.HALLMARK_PATHWAYS):
            self.n_pathways = len(self.HALLMARK_PATHWAYS)
        else:
            self.n_pathways = n_pathways
        self.pathway_names = self.HALLMARK_PATHWAYS[:self.n_pathways]
        
        # 设置每种细胞类型的基线GSVA谱
        self._setup_cell_type_baselines()
    
    def _setup_cell_type_baselines(self):
        """
        设置每种细胞类型的基线GSVA谱
        基于真实生物学特性设置不同通路的活性
        """
        self.cell_type_baselines = {
            'microglia': np.zeros(self.n_pathways),      # 小胶质细胞
            'astrocyte': np.zeros(self.n_pathways),       # 星型胶质细胞
            'b_cell': np.zeros(self.n_pathways),          # B细胞
            't_cell': np.zeros(self.n_pathways)           # T细胞
        }
        
        # 小胶质细胞：高免疫相关通路活性
        microglia_high_pathways = [
            'Hallmark_Inflammatory_Response',
            'Hallmark_Interferon_Gamma_Response',
            'Hallmark_TNFA_Signaling_via_NFKB',
            'Hallmark_IL6_JAK_STAT3_Signaling',
            'Hallmark_Apoptosis'
        ]
        
        # 星型胶质细胞：高代谢和支持功能通路活性
        astrocyte_high_pathways = [
            'Hallmark_Glycolysis',
            'Hallmark_Oxidative_Phosphorylation',
            'Hallmark_Hypoxia',
            'Hallmark_Wnt_Beta_Catenin_Signaling',
            'Hallmark_Notch_Signaling'
        ]
        
        # B细胞：高B细胞受体信号和抗体产生通路活性
        b_cell_high_pathways = [
            'Hallmark_IL2_STAT5_Signaling',
            'Hallmark_IL6_JAK_STAT3_Signaling',
            'Hallmark_MYC_Targets_V1',
            'Hallmark_MYC_Targets_V2',
            'Hallmark_E2F_Targets'
        ]
        
        # T细胞：高T细胞受体信号和细胞毒性通路活性
        t_cell_high_pathways = [
            'Hallmark_IL2_STAT5_Signaling',
            'Hallmark_Interferon_Gamma_Response',
            'Hallmark_TNFA_Signaling_via_NFKB',
            'Hallmark_E2F_Targets',
            'Hallmark_G2M_Checkpoint'
        ]
        
        # 设置高活性通路
        for pathway in microglia_high_pathways:
            if pathway in self.pathway_names:
                idx = self.pathway_names.index(pathway)
                self.cell_type_baselines['microglia'][idx] = 0.8 + np.random.rand() * 0.2
        
        for pathway in astrocyte_high_pathways:
            if pathway in self.pathway_names:
                idx = self.pathway_names.index(pathway)
                self.cell_type_baselines['astrocyte'][idx] = 0.7 + np.random.rand() * 0.2
        
        for pathway in b_cell_high_pathways:
            if pathway in self.pathway_names:
                idx = self.pathway_names.index(pathway)
                self.cell_type_baselines['b_cell'][idx] = 0.75 + np.random.rand() * 0.2
        
        for pathway in t_cell_high_pathways:
            if pathway in self.pathway_names:
                idx = self.pathway_names.index(pathway)
                self.cell_type_baselines['t_cell'][idx] = 0.75 + np.random.rand() * 0.2
    
    def generate_cell_group(self, 
                          n_microglia: int = 100,
                          n_astrocytes: int = 100,
                          n_b_cells: int = 100,
                          n_t_cells: int = 100,
                          variation_scale: float = 0.15) -> Dict[str, torch.Tensor]:
        """
        生成一组细胞数据
        
        Args:
            n_microglia: 小胶质细胞数量
            n_astrocytes: 星型胶质细胞数量
            n_b_cells: B细胞数量
            n_t_cells: T细胞数量
            variation_scale: 同一类型细胞间的变异程度
            
        Returns:
            Dict包含每种细胞类型的GSVA数据
        """
        cells_data = {}
        
        # 生成小胶质细胞
        microglia_gsva = []
        for i in range(n_microglia):
            variation = np.random.randn(self.n_pathways) * variation_scale
            gsva = self.cell_type_baselines['microglia'] + variation
            microglia_gsva.append(gsva)
        cells_data['microglia'] = torch.FloatTensor(np.array(microglia_gsva))
        
        # 生成星型胶质细胞
        astrocyte_gsva = []
        for i in range(n_astrocytes):
            variation = np.random.randn(self.n_pathways) * variation_scale
            gsva = self.cell_type_baselines['astrocyte'] + variation
            astrocyte_gsva.append(gsva)
        cells_data['astrocyte'] = torch.FloatTensor(np.array(astrocyte_gsva))
        
        # 生成B细胞
        b_cell_gsva = []
        for i in range(n_b_cells):
            variation = np.random.randn(self.n_pathways) * variation_scale
            gsva = self.cell_type_baselines['b_cell'] + variation
            b_cell_gsva.append(gsva)
        cells_data['b_cell'] = torch.FloatTensor(np.array(b_cell_gsva))
        
        # 生成T细胞
        t_cell_gsva = []
        for i in range(n_t_cells):
            variation = np.random.randn(self.n_pathways) * variation_scale
            gsva = self.cell_type_baselines['t_cell'] + variation
            t_cell_gsva.append(gsva)
        cells_data['t_cell'] = torch.FloatTensor(np.array(t_cell_gsva))
        
        return cells_data
    
    def save_cell_group(self, cells_data: Dict[str, torch.Tensor], output_path: str):
        """
        保存一组细胞数据到pt文件
        
        Args:
            cells_data: 细胞数据字典
            output_path: 输出文件路径
        """
        torch.save({
            'microglia': cells_data['microglia'],
            'astrocyte': cells_data['astrocyte'],
            'b_cell': cells_data['b_cell'],
            't_cell': cells_data['t_cell'],
            'n_pathways': self.n_pathways,
            'pathway_names': self.pathway_names
        }, output_path)
    
    def generate_multiple_groups(self, 
                               n_groups: int = 100,
                               output_dir: str = '/tmp/cell_groups',
                               **kwargs) -> List[str]:
        """
        生成多组细胞数据
        
        Args:
            n_groups: 组数（默认100）
            output_dir: 输出目录
            **kwargs: 传递给generate_cell_group的参数
            
        Returns:
            List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        print(f"Generating {n_groups} cell groups...")
        
        for i in range(n_groups):
            # 生成一组细胞数据
            cells_data = self.generate_cell_group(**kwargs)
            
            # 保存为number.pt格式
            output_path = os.path.join(output_dir, f"{i}.pt")
            self.save_cell_group(cells_data, output_path)
            output_paths.append(output_path)
            
            # 每10组输出一次进度
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{n_groups} groups")
        
        print(f"Successfully generated {n_groups} cell groups")
        return output_paths
    
    def load_cell_group(self, input_path: str) -> Dict[str, torch.Tensor]:
        """
        从pt文件加载细胞数据
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            细胞数据字典
        """
        data = torch.load(input_path)
        return {
            'microglia': data['microglia'],
            'astrocyte': data['astrocyte'],
            'b_cell': data['b_cell'],
            't_cell': data['t_cell']
        }

def main():
    """
    主函数：生成100组细胞数据
    """
    # 创建生成器
    generator = MultiCellTypeGenerator(n_pathways=50)
    
    # 生成100组数据，每组包含：
    # 100个小胶质细胞、100个星型胶质细胞、100个B细胞、100个T细胞
    output_dir = '/tmp/cell_groups'
    paths = generator.generate_multiple_groups(
        n_groups=100,
        output_dir=output_dir,
        n_microglia=100,
        n_astrocytes=100,
        n_b_cells=100,
        n_t_cells=100,
        variation_scale=0.15
    )
    
    # 验证生成的文件
    print(f"\nVerifying generated files...")
    sample_path = paths[0]
    data = generator.load_cell_group(sample_path)
    
    print(f"\nSample group ({sample_path}):")
    print(f"  - 小胶质细胞: {data['microglia'].shape}")
    print(f"  - 星型胶质细胞: {data['astrocyte'].shape}")
    print(f"  - B细胞: {data['b_cell'].shape}")
    print(f"  - T细胞: {data['t_cell'].shape}")
    print(f"  - 每个细胞的GSVA维度: {data['microglia'].shape[1]}")
    print(f"  - 每组总细胞数: {sum([v.shape[0] for v in data.values()])}")
    
    print(f"\nTotal groups generated: {len(paths)}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
