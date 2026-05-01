import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from IDgenerate.idgenerated import PhenotypeExtractor, GSVAGenerator

class CellGenerator:
    """
    生成多种类型细胞的GSVA谱
    """
    def __init__(self, n_pathways: int = 50):
        self.n_pathways = n_pathways
        
        pathway_names = [f'Pathway_{i}' for i in range(n_pathways)]
        self.baseline_gsva = {
            'astrocyte': np.random.randn(n_pathways) * 0.3,
            'microglia': np.random.randn(n_pathways) * 0.3,
            'oligodendrocyte': np.random.randn(n_pathways) * 0.3
        }
        
        self.cell_type_baselines = {
            'astrocyte': np.random.randn(n_pathways) * 0.5 + 0.5,
            'microglia': np.random.randn(n_pathways) * 0.5 - 0.3,
            'oligodendrocyte': np.random.randn(n_pathways) * 0.5 + 0.2
        }
        
    def generate_cells(self, 
                      n_astrocytes: int = 30,
                      n_microglia: int = 30,
                      n_oligodendrocytes: int = 40,
                      variation_scale: float = 0.15) -> Dict[str, torch.Tensor]:
        """
        生成多种类型细胞的GSVA谱
        
        Args:
            n_astrocytes: 星型胶质细胞数量
            n_microglia: 小胶质细胞数量
            n_oligodendrocytes: 少突胶质细胞数量
            variation_scale: 每个细胞在类型内的变异程度
            
        Returns:
            Dict包含每种细胞类型的GSVA数据
        """
        cells_data = {}
        
        n_astrocytes = 30
        n_microglia = 30
        n_oligodendrocytes = 40
        
        astrocyte_baseline = self.cell_type_baselines['astrocyte']
        microglia_baseline = self.cell_type_baselines['microglia']
        oligodendrocyte_baseline = self.cell_type_baselines['oligodendrocyte']
        
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
    
    def save_cells(self, cells_data: Dict[str, torch.Tensor], output_path: str):
        """
        保存细胞数据到pt文件
        """
        torch.save({
            'astrocytes': cells_data['astrocyte'],
            'microglia': cells_data['microglia'],
            'oligodendrocytes': cells_data['oligodendrocyte'],
            'n_pathways': self.n_pathways
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

def generate_cell_identity_file(output_dir: str = '/tmp/cell_data'):
    """
    使用idgenerated.py的机制生成细胞身份文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generator = CellGenerator(n_pathways=50)
    
    cells_data = generator.generate_cells(
        n_astrocytes=30,
        n_microglia=30,
        n_oligodendrocytes=40,
        variation_scale=0.15
    )
    
    output_path = os.path.join(output_dir, 'multi_cell_types.pt')
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
    
    return output_path

if __name__ == "__main__":
    output_path = generate_cell_identity_file()
    print(f"\n生成完成！文件路径: {output_path}")
