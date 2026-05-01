import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cellinteraction.Newcell_born import (
    GSVAEmbedding, 
    PairFormerBlock, 
    Z0Generator, 
    CellAtlasDiffusion,
    CellInteractionExtractor
)

class DrugSmileLoader:
    """加载药物SMILE字符串"""
    def __init__(self):
        self.smiles = [
            'N=C(N)N', 'C(=O)O', 'CCO', 'CCN', 'CCC',
            'CC(=O)O', 'C(=O)(O)O', 'CN', 'CO', 'CS',
            'C1=CC=CC=C1', 'C1CCCCC1', 'C1=CN=CN1', 'C1=CC=C(C=C1)Cl',
            'C1=CC=C(C=C1)OH', 'C(=O)N', 'CC(=O)N', 'C(=O)Cl', 'COC', 'CNC',
            'CCOC', 'CCNC', 'CSC', 'CCCC', 'CCCCC',
            'C1=CC=C(C=C1)C', 'C1=CC=C(C=C1)F', 'C1=CC=C(C=C1)Br', 'C1=CC=C(C=C1)I',
            'C1=CC=C(C=C1)NH2', 'C1=CC=C(C=C1)NO2', 'C1=CC=C(C=C1)CH3',
            'C1=CC=C(C=C1)OCH3', 'C1=CC=C(C=C1)NHCH3', 'C1=CC=C(C=C1)N(CH3)2',
            'C(=O)(N)N', 'C(=S)N', 'C(=NH)N', 'C#N', 'C=C',
            'C=C=C', 'CC#N', 'CC=C', 'C1=CC=CC=C1C=C', 'C1=CC=CC=C1C#N',
            'C1=CC=CC=C1C(=O)O', 'C1=CC=CC=C1C(=O)N', 'C1=CC=CC=C1OH',
            'C1=CC=CC=C1NH2', 'C1=CC=CC=C1CH2OH', 'C1=CC=CC=C1CH2NH2',
            'C1CNCCN1', 'C1CCNCC1', 'C1CNHCC1', 'C1CCNHCC1', 'C1CNCNC1',
            'C1=CN=CC=C1', 'C1=CN=C(C=C1)N', 'C1=CN=C(C=C1)O', 'C1=CN=C(C=C1)S',
            'C1=CCN=C1', 'C1=CC=NC=C1', 'C1=NC=CC=C1', 'C1=N=C(C=C1)N',
            'C1=NC=C(C=C1)N', 'C1=NC=C(C=C1)O', 'C1=C(C=CC=C1)N',
            'C1=C(C=CC=C1)O', 'C1=C(C=CC=C1)S', 'C1=C(C=CC=C1)Cl',
            'C1=C(C=CC=C1)F', 'C1=C(C=CC=C1)Br', 'C1=C(C=CC=C1)I',
            'C1=C(C=C(C=C1)N)N', 'C1=C(C=C(C=C1)O)O', 'C1=C(C=C(C=C1)C)C',
            'C1=C(C=C(C=C1)Cl)Cl', 'C1=C(C=C(C=C1)F)F', 'C1=C(C=C(C=C1)Br)Br',
            'CC(C)O', 'CC(C)N', 'CC(C)C', 'CC(C)Cl', 'CC(C)F',
            'CC(C)Br', 'CC(C)I', 'CC(C)OH', 'CC(C)NH2', 'CC(C)CH3',
            'CCC(O)C', 'CCN(C)C', 'CCC(C)C', 'CCCl(C)C', 'CCF(C)C',
            'CCBr(C)C', 'CCI(C)C', 'CC(C)(C)O', 'CC(C)(C)N', 'CC(C)(C)C'
        ]
    
    def get_all_smiles(self) -> List[str]:
        return self.smiles

class HillFunctionCalculator:
    """Hill函数计算器 - 计算细胞间受配体相互作用强度"""
    
    @staticmethod
    def hill_equation(x: torch.Tensor, Kd: float = 1.0, n: float = 2.0) -> torch.Tensor:
        """
        Hill方程: y = x^n / (Kd^n + x^n)
        
        Args:
            x: 配体浓度
            Kd: 解离常数
            n: Hill系数
        """
        # 将标量转换为Tensor，确保与x在同一设备上
        Kd_tensor = torch.tensor(Kd, device=x.device, dtype=x.dtype)
        n_tensor = torch.tensor(n, device=x.device, dtype=x.dtype)
        return torch.pow(x, n_tensor) / (torch.pow(Kd_tensor, n_tensor) + torch.pow(x, n_tensor))
    
    def calculate_ligand_receptor_interaction(
        self,
        ligand_expr: torch.Tensor,  # 配体表达量
        receptor_expr: torch.Tensor,  # 受体表达量
        cell_distance: torch.Tensor  # 细胞间距离
    ) -> torch.Tensor:
        """
        计算受配体相互作用强度
        
        Args:
            ligand_expr: (B, N) 配体表达
            receptor_expr: (B, N) 受体表达
            cell_distance: (B, N, N) 细胞间距离
        
        Returns:
            interaction_strength: (B, N, N) 相互作用强度
        """
        B, N = ligand_expr.shape
        
        # 将配体表达扩展为 (B, N, 1)
        ligand_expanded = ligand_expr.unsqueeze(2)  # (B, N, 1)
        
        # 将受体表达扩展为 (B, 1, N)
        receptor_expanded = receptor_expr.unsqueeze(1)  # (B, 1, N)
        
        # 计算基础相互作用: 配体表达 * 受体表达
        base_interaction = ligand_expanded * receptor_expanded  # (B, N, N)
        
        # 使用Hill函数根据距离衰减
        # 距离越远，相互作用越弱
        distance_factor = self.hill_equation(1.0 / (cell_distance + 1e-8), Kd=0.1, n=2)
        
        # 综合相互作用强度
        interaction = base_interaction * distance_factor
        
        return interaction

class LRLDMDataGenerator:
    """生成LR-LDM训练数据"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'n_pathways': 50,
            'd_model': 256,
            'd_pair': 128,
            'n_genes': 978,
            'n_timesteps': 1000,
            'cell_group_dir': '/tmp/cell_groups'
        }
        
        self.device = torch.device('cpu')
        self.smile_loader = DrugSmileLoader()
        self.hill_calculator = HillFunctionCalculator()
        
        # 初始化模型
        self._init_models()
    
    def _init_models(self):
        """初始化所有必要的模型组件"""
        self.gsva_embedding = GSVAEmbedding(
            n_pathways=self.config['n_pathways'],
            d_model=self.config['d_model']
        ).to(self.device)
        
        self.pairformer = PairFormerBlock(
            d_single=self.config['d_model'],
            d_pair=self.config['d_pair']
        ).to(self.device)
        
        self.z0_generator = Z0Generator(
            d_single=self.config['d_model'],
            d_pair=self.config['d_pair'],
            n_genes=self.config['n_genes']
        ).to(self.device)
        
        self.diffusion = CellAtlasDiffusion(
            n_genes=self.config['n_genes'],
            d_model=self.config['d_model'],
            n_timesteps=self.config['n_timesteps']
        ).to(self.device)
        
        self.interaction_extractor = CellInteractionExtractor(
            d_pair=self.config['d_pair'],
            n_pathways=self.config['n_pathways']
        ).to(self.device)
    
    def load_cell_group(self, group_idx: int) -> Dict:
        """加载单个细胞组数据"""
        file_path = os.path.join(self.config['cell_group_dir'], f"{group_idx}.pt")
        if os.path.exists(file_path):
            return torch.load(file_path, map_location=self.device)
        else:
            raise FileNotFoundError(f"Cell group {group_idx} not found")
    
    def generate_gsva_for_smile(self, cell_data: Dict, smile: str) -> torch.Tensor:
        """
        为给定smile生成细胞组的GSVA谱
        
        Args:
            cell_data: 细胞数据字典（包含microglia, astrocyte, b_cell, t_cell）
            smile: 药物SMILE字符串
        
        Returns:
            mgsva: (N, P) 多组GSVA特征，N=400（4种细胞各100个）
        """
        # 从cell_data中提取所有细胞类型的GSVA数据并合并
        cell_types = ['microglia', 'astrocyte', 'b_cell', 't_cell']
        gsva_list = []
        
        for cell_type in cell_types:
            if cell_type in cell_data:
                gsva_list.append(cell_data[cell_type])
        
        # 沿维度0拼接（按1轴对齐）
        combined_gsva = torch.cat(gsva_list, dim=0)  # (400, 49)
        
        # 获取细胞数量
        n_cells = combined_gsva.shape[0]
        n_pathways = self.config['n_pathways']
        
        # 模拟GSVA生成过程（基于smile的影响）
        # 将smile转换为数值特征
        smile_hash = hash(smile) % 1000 / 1000.0
        
        # 生成GSVA谱，smile影响基线
        # 使用原始GSVA作为基础，添加smile调制
        gsva_base = combined_gsva
        if gsva_base.shape[1] < n_pathways:
            # 如果原始通路数少于目标通路数，进行扩展
            padding = torch.randn(n_cells, n_pathways - gsva_base.shape[1]) * 0.5
            gsva_base = torch.cat([gsva_base, padding], dim=1)
        elif gsva_base.shape[1] > n_pathways:
            # 如果原始通路数多于目标通路数，进行截断
            gsva_base = gsva_base[:, :n_pathways]
        
        # smile调制
        gsva_modulated = gsva_base + smile_hash * 2.0
        
        return gsva_modulated
    
    def generate_lr_interactions(self, gene_expression: torch.Tensor, coords: torch.Tensor) -> Dict:
        """
        生成受配体相互作用数据
        
        Args:
            gene_expression: (N, G) 基因表达矩阵
            coords: (N, 3) 细胞坐标
        
        Returns:
            lr_data: 受配体相互作用数据
        """
        N = gene_expression.shape[0]
        
        # 计算细胞间距离
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (N, N, 3)
        distances = torch.norm(diff, dim=-1)  # (N, N)
        
        # 模拟配体和受体基因表达（选取前20个基因作为受配体对）
        n_lr_pairs = 20
        ligand_expr = gene_expression[:, :n_lr_pairs]  # (N, 20)
        receptor_expr = gene_expression[:, 20:40]  # (N, 20)
        
        # 使用Hill函数计算相互作用
        interactions = []
        for i in range(n_lr_pairs):
            ligand = ligand_expr[:, i]  # (N,)
            receptor = receptor_expr[:, i]  # (N,)
            interaction = self.hill_calculator.calculate_ligand_receptor_interaction(
                ligand.unsqueeze(0),
                receptor.unsqueeze(0),
                distances.unsqueeze(0)
            )[0]  # (N, N)
            interactions.append(interaction)
        
        # 聚合所有受配体对的相互作用
        interaction_matrix = torch.stack(interactions, dim=0).mean(dim=0)  # (N, N)
        
        # 构建source-target映射
        lr_data = {
            'ligand_expression': ligand_expr,
            'receptor_expression': receptor_expr,
            'interaction_matrix': interaction_matrix,
            'distances': distances,
            'source_cells': torch.arange(N),
            'target_cells': torch.arange(N)
        }
        
        return lr_data
    
    def generate_all_data(self, n_groups: int = 100, n_smiles: int = 100) -> Tuple[str, List[str]]:
        """
        生成所有10000组数据并记录映射关系
        
        Args:
            n_groups: 细胞组数
            n_smiles: smile数
        
        Returns:
            mapping_file: 映射关系表文件路径
            data_files: 生成的数据文件路径列表
        """
        smiles = self.smile_loader.get_all_smiles()[:n_smiles]
        
        # 创建输出目录
        output_dir = '/tmp/lrldm_data'
        os.makedirs(output_dir, exist_ok=True)
        
        # 映射关系表
        mapping_data = []
        data_files = []
        
        total_combinations = n_groups * n_smiles
        counter = 0
        
        print(f"Generating {total_combinations} combinations...")
        
        for group_idx in range(n_groups):
            try:
                cell_data = self.load_cell_group(group_idx)
            except FileNotFoundError:
                print(f"Warning: Cell group {group_idx} not found, skipping...")
                continue
            
            for smile_idx, smile in enumerate(smiles):
                counter += 1
                
                if counter % 100 == 0:
                    print(f"Processing {counter}/{total_combinations} ({counter/total_combinations*100:.1f}%)")
                
                # 生成GSVA
                gsva = self.generate_gsva_for_smile(cell_data, smile)
                
                # PairFormer处理
                gsva_batch = gsva.unsqueeze(0).to(self.device)  # (1, N, P)
                cell_types = torch.zeros(1, gsva.shape[0], dtype=torch.long).to(self.device)
                tissue_id = torch.zeros(1, dtype=torch.long).to(self.device)
                cell_order = torch.arange(gsva.shape[0]).unsqueeze(0).to(self.device)
                
                h_single = self.gsva_embedding(gsva_batch, cell_types, tissue_id, cell_order)
                h_pair = torch.randn(1, gsva.shape[0], gsva.shape[0], self.config['d_pair']).to(self.device)
                
                # 生成Z0
                interact_info = {
                    'interaction_types': torch.randn(1, gsva.shape[0], gsva.shape[0], 4).to(self.device),
                    'ligand_receptor': torch.randn(1, gsva.shape[0], gsva.shape[0], 10).to(self.device),
                    'crosstalk': torch.randn(1, gsva.shape[0], gsva.shape[0], 50).to(self.device),
                    'distance': torch.rand(1, gsva.shape[0], gsva.shape[0]).to(self.device),
                    'predicted_distance': torch.rand(1, gsva.shape[0], gsva.shape[0]).to(self.device) * 10 + 1
                }
                
                z0 = self.z0_generator(h_single, h_pair, interact_info)
                
                # 生成受配体相互作用
                lr_data = self.generate_lr_interactions(
                    z0['expression'][0],  # (N, G)
                    z0['coordinates'][0]   # (N, 3)
                )
                
                # 保存数据
                data_id = f"{group_idx}_{smile_idx}"
                data_path = os.path.join(output_dir, f"{data_id}.pt")
                
                saved_data = {
                    'group_idx': group_idx,
                    'smile_idx': smile_idx,
                    'smile': smile,
                    'gsva': gsva.cpu(),
                    'h_single': h_single.cpu(),
                    'h_pair': h_pair.cpu(),
                    'z0': {k: v.cpu() for k, v in z0.items()},
                    'lr_data': {k: v.cpu() for k, v in lr_data.items()}
                }
                
                torch.save(saved_data, data_path)
                data_files.append(data_path)
                
                # 记录映射关系
                mapping_data.append({
                    'data_id': data_id,
                    'group_idx': group_idx,
                    'smile_idx': smile_idx,
                    'smile': smile,
                    'n_cells': gsva.shape[0],
                    'file_path': data_path
                })
        
        # 保存映射关系表
        mapping_df = pd.DataFrame(mapping_data)
        mapping_file = os.path.join(output_dir, 'mapping_table.csv')
        mapping_df.to_csv(mapping_file, index=False)
        
        print(f"Generated {len(data_files)} data files")
        print(f"Mapping table saved to: {mapping_file}")
        
        return mapping_file, data_files

class LRLDMModel(nn.Module):
    """受配体条件扩散模型"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 受配体嵌入
        self.source_lr_encoder = nn.Sequential(
            nn.Linear(config['d_lr'], config['d_model']),
            nn.ReLU(),
            nn.Linear(config['d_model'], config['d_model'])
        )
        
        self.target_lr_encoder = nn.Sequential(
            nn.Linear(config['d_lr'], config['d_model']),
            nn.ReLU(),
            nn.Linear(config['d_model'], config['d_model'])
        )
        
        # SMILE VAE (简化版本)
        self.smile_encoder = nn.Sequential(
            nn.Linear(config['smile_dim'], config['d_model']),
            nn.ReLU(),
            nn.Linear(config['d_model'], config['latent_dim'])
        )
        
        # 扩散模型
        self.diffusion = CellAtlasDiffusion(
            n_genes=config['n_genes'],
            d_model=config['d_model'],
            n_timesteps=config['n_timesteps']
        )
    
    def forward(self, source_lr, target_lr, smile_feat, z_0, t):
        """
        训练前向传播
        
        Args:
            source_lr: 源细胞受配体特征
            target_lr: 目标细胞受配体特征
            smile_feat: SMILE特征
            z_0: 初始细胞图谱
            t: 时间步
        """
        # 编码条件信息
        source_emb = self.source_lr_encoder(source_lr)
        target_emb = self.target_lr_encoder(target_lr)
        smile_latent = self.smile_encoder(smile_feat)
        
        # 聚合条件
        cond = torch.cat([source_emb, target_emb, smile_latent], dim=-1)
        
        # 扩散训练
        pred_noise, noise = self.diffusion(z_0, cond, cond, t)
        
        return pred_noise, noise

def main():
    """主函数"""
    print("Starting LRLDM data generation pipeline...")
    
    # 生成数据（由于磁盘空间限制，先生成少量数据进行演示）
    generator = LRLDMDataGenerator()
    mapping_file, data_files = generator.generate_all_data(
        n_groups=10,
        n_smiles=10
    )
    
    print(f"\nData generation complete!")
    print(f"Total data files: {len(data_files)}")
    print(f"Mapping table: {mapping_file}")
    
    # 验证生成的数据
    sample_data = torch.load(data_files[0])
    print(f"\nSample data structure:")
    print(f"  group_idx: {sample_data['group_idx']}")
    print(f"  smile: {sample_data['smile']}")
    print(f"  gsva shape: {sample_data['gsva'].shape}")
    print(f"  z0['expression'] shape: {sample_data['z0']['expression'].shape}")
    print(f"  z0['coordinates'] shape: {sample_data['z0']['coordinates'].shape}")
    print(f"  lr_data['interaction_matrix'] shape: {sample_data['lr_data']['interaction_matrix'].shape}")

if __name__ == '__main__':
    main()
