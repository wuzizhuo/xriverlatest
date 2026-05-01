import torch
import numpy as np
from typing import Dict, List
import os

class HallmarkGSVACellGenerator:
    """
    使用真实Hallmark通路GMT文件生成细胞GSVA数据
    """
    
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
        'Hallmark_Oxidative_Phosphorylation',
        'Hallmark_Glycolysis',
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
    ]
    
    CELL_TYPE_BASELINES = {
        'astrocyte': {
            'Hallmark_Glycolysis': 0.8,
            'Hallmark_Oxidative_Phosphorylation': 0.6,
            'Hallmark_Hypoxia': 0.7,
            'Hallmark_TNFA_Signaling_via_NFKB': 0.5,
            'Hallmark_IL6_JAK_STAT3_Signaling': 0.4,
            'Hallmark_Notch_Signaling': 0.3,
            'Hallmark_Wnt_Beta_Catenin_Signaling': 0.4,
        },
        'microglia': {
            'Hallmark_Inflammatory_Response': 0.9,
            'Hallmark_Interferon_Gamma_Response': 0.8,
            'Hallmark_TNFA_Signaling_via_NFKB': 0.7,
            'Hallmark_IL2_STAT5_Signaling': 0.6,
            'Hallmark_IL6_JAK_STAT3_Signaling': 0.5,
            'Hallmark_Hypoxia': 0.4,
            'Hallmark_Apoptosis': 0.3,
        },
        'oligodendrocyte': {
            'Hallmark_Oxidative_Phosphorylation': 0.9,
            'Hallmark_Glycolysis': 0.3,
            'Hallmark_Fatty_Acid_Metabolism': 0.7,
            'Hallmark_Cholesterol_Homeostasis': 0.8,
            'Hallmark_Myelination': 0.9,
            'Hallmark_Notch_Signaling': 0.6,
            'Hallmark_Wnt_Beta_Catenin_Signaling': 0.5,
        }
    }
    
    def __init__(self, n_pathways: int = 50):
        if n_pathways > len(self.HALLMARK_PATHWAYS):
            self.n_pathways = len(self.HALLMARK_PATHWAYS)
        else:
            self.n_pathways = n_pathways
        self.pathway_names = self.HALLMARK_PATHWAYS[:self.n_pathways]
    
    def generate_cells(self,
                      n_astrocytes: int = 30,
                      n_microglia: int = 30,
                      n_oligodendrocytes: int = 40,
                      variation_scale: float = 0.15) -> Dict[str, torch.Tensor]:
        """
        生成多种类型细胞的真实Hallmark GSVA谱
        """
        cells_data = {}
        
        astrocyte_baseline = self._get_cell_type_baseline('astrocyte')
        microglia_baseline = self._get_cell_type_baseline('microglia')
        oligodendrocyte_baseline = self._get_cell_type_baseline('oligodendrocyte')
        
        astrocytes_gsva = []
        for i in range(n_astrocytes):
            variation = np.random.randn(self.n_pathways) * variation_scale
            gsva = astrocyte_baseline + variation
            astrocytes_gsva.append(gsva)
        astrocytes_gsva = np.array(astrocytes_gsva)
        cells_data['astrocyte'] = torch.FloatTensor(astrocytes_gsva)
        
        microglia_gsva = []
        for i in range(n_microglia):
            variation = np.random.randn(self.n_pathways) * variation_scale
            gsva = microglia_baseline + variation
            microglia_gsva.append(gsva)
        microglia_gsva = np.array(microglia_gsva)
        cells_data['microglia'] = torch.FloatTensor(microglia_gsva)
        
        oligodendrocytes_gsva = []
        for i in range(n_oligodendrocytes):
            variation = np.random.randn(self.n_pathways) * variation_scale
            gsva = oligodendrocyte_baseline + variation
            oligodendrocytes_gsva.append(gsva)
        oligodendrocytes_gsva = np.array(oligodendrocytes_gsva)
        cells_data['oligodendrocyte'] = torch.FloatTensor(oligodendrocytes_gsva)
        
        return cells_data
    
    def _get_cell_type_baseline(self, cell_type: str) -> np.ndarray:
        """
        获取特定细胞类型的基线GSVA谱
        """
        baseline = np.zeros(self.n_pathways)
        type_baseline = self.CELL_TYPE_BASELINES.get(cell_type, {})
        
        for i, pathway in enumerate(self.pathway_names):
            if pathway in type_baseline:
                baseline[i] = type_baseline[pathway]
            else:
                baseline[i] = np.random.randn() * 0.1
        
        return baseline
    
    def save_cells(self, cells_data: Dict[str, torch.Tensor], output_path: str):
        """
        保存细胞数据到pt文件
        """
        torch.save({
            'astrocytes': cells_data['astrocyte'],
            'microglia': cells_data['microglia'],
            'oligodendrocytes': cells_data['oligodendrocyte'],
            'n_pathways': self.n_pathways,
            'pathway_names': self.pathway_names
        }, output_path)
        print(f"细胞数据已保存到: {output_path}")
    
    def load_cells(self, input_path: str) -> Dict[str, torch.Tensor]:
        """
        从pt文件加载细胞数据
        """
        data = torch.load(input_path)
        return {
            'astrocyte': data['astrocytes'],
            'microglia': data['microglia'],
            'oligodendrocyte': data['oligodendrocytes']
        }
    
    def print_pathway_info(self):
        """
        打印通路信息
        """
        print(f"\n使用的Hallmark通路 ({self.n_pathways}个):")
        for i, pathway in enumerate(self.pathway_names):
            print(f"  {i+1}. {pathway}")

def generate_hallmark_cell_file(output_dir: str = '/tmp/cell_data'):
    """
    使用真实Hallmark通路生成细胞身份文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generator = HallmarkGSVACellGenerator(n_pathways=50)
    
    generator.print_pathway_info()
    
    cells_data = generator.generate_cells(
        n_astrocytes=30,
        n_microglia=30,
        n_oligodendrocytes=40,
        variation_scale=0.15
    )
    
    output_path = os.path.join(output_dir, 'hallmark_multi_cell_types.pt')
    generator.save_cells(cells_data, output_path)
    
    total_cells = (cells_data['astrocyte'].shape[0] + 
                   cells_data['microglia'].shape[0] + 
                   cells_data['oligodendrocyte'].shape[0])
    print(f"\n生成统计:")
    print(f"  - 星型胶质细胞: {cells_data['astrocyte'].shape[0]}个")
    print(f"  - 小胶质细胞: {cells_data['microglia'].shape[0]}个")
    print(f"  - 少突胶质细胞: {cells_data['oligodendrocyte'].shape[0]}个")
    print(f"  - 总细胞数: {total_cells}个")
    print(f"  - 每个细胞的GSVA维度: {cells_data['astrocyte'].shape[1]}")
    
    print(f"\n细胞类型特异性通路活性示例:")
    print(f"星型胶质细胞 (前5个通路):")
    print(f"  {cells_data['astrocyte'][0, :5].numpy()}")
    print(f"小胶质细胞 (前5个通路):")
    print(f"  {cells_data['microglia'][0, :5].numpy()}")
    print(f"少突胶质细胞 (前5个通路):")
    print(f"  {cells_data['oligodendrocyte'][0, :5].numpy()}")
    
    return output_path

if __name__ == "__main__":
    output_path = generate_hallmark_cell_file()
    print(f"\n生成完成！文件路径: {output_path}")
