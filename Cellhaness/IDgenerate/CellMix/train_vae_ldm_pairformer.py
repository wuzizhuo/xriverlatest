import torch
import numpy as np
import os
import sys
import random
from typing import Dict, List

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SmileLoader:
    """
    从数据源加载smile
    """
    def __init__(self):
        self.smiles = self._load_smiles()
    
    def _load_smiles(self) -> List[str]:
        """加载smile数据"""
        smile_file = '/home/wupf_260213/controlnet/ALLmodels/model/data/cellpainting/images/BRset/images/image-smiles.csv'
        if os.path.exists(smile_file):
            import pandas as pd
            df = pd.read_csv(smile_file)
            return df['smiles'].dropna().unique().tolist()
        # 如果文件不存在，使用预定义的smile
        return [
            'CC(=O)NC1=CC=C(C=C1)O',
            'C1=CC=C(C=C1)O',
            'CC(=O)Nc1ccccc1',
            'C1=CC(=CC=C1)CO',
            'CC(=O)OC1=CC=CC=C1',
            'C1=CC=C2C(=C1)C(=CN2)N',
            'CC(=O)N1CCCCC1',
            'C1=CC=C(C=C1)CN',
            'CC(=O)Nc1ccc(cc1)Cl',
            'C1=CC=C(C=C1)Cl'
        ]
    
    def get_random_smile(self) -> str:
        """获取随机smile"""
        return random.choice(self.smiles)

class DrugSapoGSVA:
    """
    利用drugsapo生成GSVA过程
    """
    def __init__(self, n_pathways: int = 50):
        self.n_pathways = n_pathways
    
    def generate_gsva_for_cell(self, cell_type: str, smile: str) -> np.ndarray:
        """
        为单个细胞生成GSVA
        
        Args:
            cell_type: 细胞类型
            smile: 药物smile
            
        Returns:
            GSVA谱 (n_pathways,)
        """
        # 基于细胞类型设置基线GSVA
        baseline = self._get_cell_baseline(cell_type)
        
        # 基于smile添加药物效应
        drug_effect = self._compute_drug_effect(smile)
        
        # 组合基线和药物效应
        gsva = baseline + drug_effect * 0.3
        
        return gsva
    
    def _get_cell_baseline(self, cell_type: str) -> np.ndarray:
        """获取细胞类型的基线GSVA"""
        baselines = {
            'microglia': np.random.randn(self.n_pathways) * 0.3 + 0.2,
            'astrocyte': np.random.randn(self.n_pathways) * 0.3 + 0.3,
            'b_cell': np.random.randn(self.n_pathways) * 0.3 - 0.1,
            't_cell': np.random.randn(self.n_pathways) * 0.3 - 0.2
        }
        return baselines.get(cell_type, np.random.randn(self.n_pathways) * 0.3)
    
    def _compute_drug_effect(self, smile: str) -> np.ndarray:
        """计算药物效应"""
        # 基于smile的哈希值生成随机效应
        hash_val = hash(smile) % 1000
        np.random.seed(hash_val)
        return np.random.randn(self.n_pathways) * 0.5

class MultiGSVAProcessor:
    """
    处理多组细胞数据的GSVA
    """
    def __init__(self, cell_groups_dir: str = '/tmp/cell_groups'):
        self.cell_groups_dir = cell_groups_dir
        self.smile_loader = SmileLoader()
        self.gsva_generator = DrugSapoGSVA()
    
    def load_cell_group(self, group_idx: int) -> Dict[str, torch.Tensor]:
        """加载一组细胞数据"""
        path = os.path.join(self.cell_groups_dir, f"{group_idx}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Group {group_idx} not found")
        return torch.load(path)
    
    def process_group(self, group_idx: int, smile: str) -> torch.Tensor:
        """
        处理一组细胞数据，生成mgsva
        
        Args:
            group_idx: 组索引
            smile: 药物smile
            
        Returns:
            mgsva: 拼接后的GSVA张量 (400, n_pathways)
        """
        cell_data = self.load_cell_group(group_idx)
        
        all_gsva = []
        
        # 处理每种细胞类型
        for cell_type, cells in cell_data.items():
            if cell_type in ['microglia', 'astrocyte', 'b_cell', 't_cell']:
                n_cells = cells.shape[0]
                for i in range(n_cells):
                    gsva = self.gsva_generator.generate_gsva_for_cell(cell_type, smile)
                    all_gsva.append(gsva)
        
        # 按1轴拼接
        mgsva = np.array(all_gsva)
        return torch.FloatTensor(mgsva)

class PairFormerProcessor:
    """
    利用PairFormer处理mgsva
    """
    def __init__(self, d_model: int = 256, d_pair: int = 128, n_heads: int = 8):
        self.d_model = d_model
        self.d_pair = d_pair
        self.n_heads = n_heads
        
        # 导入PairFormerBlock
        from cellinteraction.Newcell_born import PairFormerBlock, GSVAEmbedding
        self.embedding = GSVAEmbedding(
            n_pathways=50,
            d_model=d_model,
            n_cell_types=100,
            max_cells=1000
        )
        self.pairformer = PairFormerBlock(
            d_single=d_model,
            d_pair=d_pair,
            n_heads=n_heads,
            n_layers=4
        )
    
    def process(self, mgsva: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        处理mgsva，生成条件特征
        
        Args:
            mgsva: (n_cells, n_pathways)
            
        Returns:
            Dict包含处理后的特征
        """
        # 添加batch维度
        gsva_batch = mgsva.unsqueeze(0)  # (1, n_cells, n_pathways)
        
        # 生成模拟的细胞类型和位置信息
        n_cells = mgsva.shape[0]
        cell_types = torch.zeros(1, n_cells, dtype=torch.long)
        tissue_id = torch.tensor([0])
        cell_order = torch.arange(n_cells).unsqueeze(0)
        
        # 嵌入GSVA
        h_single = self.embedding(gsva_batch, cell_types, tissue_id, cell_order)
        
        # 初始化细胞对表示
        h_pair = torch.zeros(1, n_cells, n_cells, self.d_pair)
        
        # PairFormer处理
        h_single_out, h_pair_out = self.pairformer(h_single, h_pair)
        
        return {
            'h_single': h_single_out,
            'h_pair': h_pair_out
        }

class VAELDMModel:
    """
    VAE + LDM + PairFormer条件模型
    """
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化VAE和LDM模型"""
        # 导入模型
        from cellinteraction.Newcell_born import Z0Generator, CellAtlasDiffusion
        
        self.z0_generator = Z0Generator(
            d_single=self.config.get('d_model', 256),
            d_pair=self.config.get('d_pair', 128),
            n_genes=self.config.get('n_genes', 978),
            spatial_dim=self.config.get('spatial_dim', 3)
        ).to(self.device)
        
        self.diffusion = CellAtlasDiffusion(
            n_genes=self.config.get('n_genes', 978),
            spatial_dim=self.config.get('spatial_dim', 3),
            d_model=self.config.get('d_model', 256),
            n_timesteps=self.config.get('n_timesteps', 1000)
        ).to(self.device)
    
    def train(self, h_single: torch.Tensor, h_pair: torch.Tensor, cell_data: Dict):
        """
        训练模型
        
        Args:
            h_single: PairFormer输出的单细胞表示
            h_pair: PairFormer输出的细胞对表示
            cell_data: 原始细胞数据
        """
        # 生成模拟的互作信息
        B, N = h_single.shape[0], h_single.shape[1]
        interact_info = {
            'interaction_types': torch.randn(B, N, N, 4),
            'ligand_receptor': torch.randn(B, N, N, 10),
            'crosstalk': torch.randn(B, N, N, 50),
            'distance': torch.rand(B, N, N),
            'predicted_distance': torch.rand(B, N, N) * 10 + 1  # 预测距离，范围1-11
        }
        
        # 移动到设备
        h_single = h_single.to(self.device)
        h_pair = h_pair.to(self.device)
        for key, value in interact_info.items():
            if isinstance(value, torch.Tensor):
                interact_info[key] = value.to(self.device)
        
        # 生成Z0
        z0_output = self.z0_generator(h_single, h_pair, interact_info)
        
        # 组装训练数据
        B = h_single.shape[0]
        N = h_single.shape[1]
        
        if isinstance(z0_output, dict):
            expression = z0_output['expression']
            coords = z0_output['coordinates']
            z_0_true = torch.cat([expression, coords], dim=-1)
        else:
            z_0_true = z0_output
        
        # 模拟扩散训练步骤
        optimizer = torch.optim.Adam(
            list(self.z0_generator.parameters()) + list(self.diffusion.parameters()),
            lr=1e-4
        )
        
        # 执行一个训练步骤
        optimizer.zero_grad()
        
        # 随机采样时间步
        t = torch.randint(0, self.config.get('n_timesteps', 1000), (B,), device=self.device)
        
        # 调用扩散模型的forward方法，它会返回(pred_noise, noise)
        pred_noise, noise = self.diffusion(z_0_true, h_single, h_pair, t)
        
        # 计算损失：预测噪声与真实噪声的MSE
        loss = torch.mean((pred_noise - noise) ** 2)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        return float(loss.item())

class CellTrainingPipeline:
    """
    完整的细胞训练pipeline
    """
    def __init__(self, config: Dict):
        self.config = config
        self.smile_loader = SmileLoader()
        self.mgsva_processor = MultiGSVAProcessor()
        self.pairformer_processor = PairFormerProcessor()
        self.vae_ldm_model = VAELDMModel(config)
    
    def run(self, n_groups: int = 100):
        """
        运行完整的训练流程
        
        Args:
            n_groups: 要处理的组数
        """
        # 随机抽取一个smile
        smile = self.smile_loader.get_random_smile()
        print(f"Selected smile: {smile}")
        
        # 处理每组数据
        total_loss = 0.0
        processed_groups = 0
        
        for group_idx in range(n_groups):
            try:
                print(f"\nProcessing group {group_idx + 1}/{n_groups}")
                
                # 1. 处理组数据，生成mgsva
                mgsva = self.mgsva_processor.process_group(group_idx, smile)
                print(f"  Generated mgsva: {mgsva.shape}")
                
                # 2. 使用PairFormer处理mgsva
                features = self.pairformer_processor.process(mgsva)
                print(f"  PairFormer processed: h_single={features['h_single'].shape}, h_pair={features['h_pair'].shape}")
                
                # 3. 加载原始细胞数据用于训练
                cell_data = self.mgsva_processor.load_cell_group(group_idx)
                
                # 4. 训练VAE+LDM模型
                loss = self.vae_ldm_model.train(features['h_single'], features['h_pair'], cell_data)
                print(f"  Training loss: {loss:.6f}")
                
                total_loss += loss
                processed_groups += 1
                
            except FileNotFoundError:
                print(f"  Group {group_idx} not found, skipping")
                continue
            except Exception as e:
                print(f"  Error processing group {group_idx}: {e}")
                continue
        
        # 输出统计
        if processed_groups > 0:
            avg_loss = total_loss / processed_groups
            print(f"\nTraining completed!")
            print(f"Total groups processed: {processed_groups}/{n_groups}")
            print(f"Average training loss: {avg_loss:.6f}")
            print(f"Used smile: {smile}")
        else:
            print(f"\nNo groups were processed successfully.")

def main():
    """主函数"""
    # 配置参数
    config = {
        'd_model': 256,
        'd_pair': 128,
        'n_heads': 8,
        'n_layers': 4,
        'n_pathways': 50,
        'n_genes': 978,
        'spatial_dim': 3,
        'n_timesteps': 1000
    }
    
    # 创建pipeline
    pipeline = CellTrainingPipeline(config)
    
    # 运行pipeline，处理100组数据
    pipeline.run(n_groups=100)

if __name__ == "__main__":
    main()
