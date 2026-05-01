import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cellinteraction.Newcell_born import CellAtlasDiffusion

class SMILEEncoder(nn.Module):
    """SMILE字符串编码器 - 将SMILE转换为潜在向量"""
    
    def __init__(self, vocab_size: int = 100, d_model: int = 256, latent_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(self, smile_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            smile_tokens: (B, seq_len) SMILE字符串的token序列
        
        Returns:
            latent: (B, latent_dim) 潜在向量
        """
        embed = self.embedding(smile_tokens)  # (B, seq_len, d_model)
        _, h_n = self.rnn(embed)  # h_n: (num_layers * 2, B, d_model)
        h_concat = torch.cat([h_n[-1], h_n[-2]], dim=-1)  # (B, d_model * 2)
        latent = self.fc(h_concat)  # (B, latent_dim)
        return latent

class LRFeatureExtractor(nn.Module):
    """受配体特征提取器"""
    
    def __init__(self, d_model: int = 256):
        super().__init__()
        # 输入是3个聚合特征（mean, max, diag）
        self.fc = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, lr_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lr_matrix: (B, N, N) 受配体相互作用矩阵
        
        Returns:
            features: (B, d_model) 提取的特征向量
        """
        # 聚合细胞间的相互作用信息
        mean_feat = lr_matrix.mean(dim=(1, 2))  # (B,)
        max_feat = lr_matrix.max(dim=1)[0].max(dim=1)[0]  # (B,)
        diag_feat = lr_matrix.diagonal(dim1=1, dim2=2).mean(dim=1)  # (B,)
        
        combined = torch.cat([
            mean_feat.unsqueeze(1),
            max_feat.unsqueeze(1),
            diag_feat.unsqueeze(1)
        ], dim=1)  # (B, 3)
        
        # 扩展到d_model维度
        return self.fc(combined)

class LRLDMModel(nn.Module):
    """受配体条件扩散模型"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 受配体特征提取器
        self.source_lr_extractor = LRFeatureExtractor(
            d_model=config['d_model']
        )
        
        self.target_lr_extractor = LRFeatureExtractor(
            d_model=config['d_model']
        )
        
        # SMILE编码器（输出维度改为d_model）
        self.smile_encoder = SMILEEncoder(
            vocab_size=config['smile_vocab_size'],
            d_model=config['d_model'],
            latent_dim=config['d_model']  # 修改为d_model，使三个条件维度一致
        )
        
        # 条件融合层
        self.condition_fusion = nn.Sequential(
            nn.Linear(config['d_model'] * 3, config['d_model']),
            nn.ReLU(),
            nn.Linear(config['d_model'], config['d_model'])
        )
        
        # 扩散模型
        self.diffusion = CellAtlasDiffusion(
            n_genes=config['n_genes'],
            d_model=config['d_model'],
            n_timesteps=config['n_timesteps']
        )
    
    def encode_smile(self, smile: str) -> torch.Tensor:
        """将SMILE字符串转换为token并编码"""
        # 简单的SMILE tokenization
        tokens = [ord(c) % 100 for c in smile]
        if len(tokens) < 20:
            tokens += [0] * (20 - len(tokens))
        else:
            tokens = tokens[:20]
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    
    def forward(self, source_lr: torch.Tensor, target_lr: torch.Tensor, 
                smile: str, z_0: Dict[str, torch.Tensor], t: torch.Tensor):
        """
        训练前向传播
        
        Args:
            source_lr: (B, N, N) 源细胞受配体相互作用矩阵
            target_lr: (B, N, N) 目标细胞受配体相互作用矩阵
            smile: SMILE字符串
            z_0: 初始细胞图谱字典
            t: 时间步
        """
        # 提取受配体特征
        source_feat = self.source_lr_extractor(source_lr)
        target_feat = self.target_lr_extractor(target_lr)
        
        # 编码SMILE
        smile_tokens = self.encode_smile(smile).repeat(source_lr.shape[0], 1).to(source_lr.device)
        smile_feat = self.smile_encoder(smile_tokens)
        
        # 融合条件
        cond = torch.cat([source_feat, target_feat, smile_feat], dim=-1)
        cond = self.condition_fusion(cond)
        
        # 组装z_0: (B, N, C) where N=400, C=978+3=981
        z_0_tensor = torch.cat([
            z_0['expression'],  # (B, N, 978)
            z_0['coordinates']   # (B, N, 3)
        ], dim=-1)  # (B, N, 981)
        
        # 将条件向量扩展成CellAtlasDiffusion期望的格式
        # h_single: (B, N, d_model), h_pair: (B, N, N, d_model)
        B = z_0['expression'].shape[0]
        N = z_0['expression'].shape[1]
        h_single = cond.unsqueeze(1).repeat(1, N, 1)  # (B, N, d_model)
        h_pair = cond.unsqueeze(1).unsqueeze(1).repeat(1, N, N, 1)  # (B, N, N, d_model)
        
        # 扩散训练
        pred_noise, noise = self.diffusion(z_0_tensor, h_single, h_pair, t)
        
        return pred_noise, noise

class LRLDMTrainer:
    """LRLDM模型训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = LRLDMModel(config).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        
        # 损失函数
        self.loss_fn = nn.MSELoss()
        
        # 加载映射表
        self.mapping_df = pd.read_csv(config['mapping_file'])
        
        # 数据缓存
        self.data_cache = {}
    
    def load_data(self, data_id: str) -> Dict:
        """加载单个数据文件"""
        if data_id in self.data_cache:
            return self.data_cache[data_id]
        
        file_path = self.mapping_df[self.mapping_df['data_id'] == data_id]['file_path'].iloc[0]
        data = torch.load(file_path, map_location=self.device)
        self.data_cache[data_id] = data
        return data
    
    def train(self, num_epochs: int = 100, batch_size: int = 4, patience: int = 10):
        """训练模型（带早停机制）"""
        self.model.train()
        data_ids = self.mapping_df['data_id'].tolist()
        
        # 早停相关变量
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # 打乱数据顺序
            np.random.shuffle(data_ids)
            
            # 分批训练
            for i in tqdm(range(0, len(data_ids), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"):
                batch_ids = data_ids[i:i+batch_size]
                
                # 加载批次数据
                batch_source_lr = []
                batch_target_lr = []
                batch_z0 = []
                batch_smile = []
                
                for data_id in batch_ids:
                    data = self.load_data(data_id)
                    lr_matrix = data['lr_data']['interaction_matrix']
                    
                    # 使用相同的相互作用矩阵作为source和target（演示用）
                    batch_source_lr.append(lr_matrix.unsqueeze(0))
                    batch_target_lr.append(lr_matrix.unsqueeze(0))
                    batch_z0.append(data['z0'])
                    batch_smile.append(data['smile'])
                
                # 拼接批次数据
                source_lr = torch.cat(batch_source_lr, dim=0).to(self.device)
                target_lr = torch.cat(batch_target_lr, dim=0).to(self.device)
                
                # 组装z0
                z0_expressions = [z['expression'] for z in batch_z0]
                z0_coordinates = [z['coordinates'] for z in batch_z0]
                z0 = {
                    'expression': torch.cat(z0_expressions, dim=0).to(self.device),
                    'coordinates': torch.cat(z0_coordinates, dim=0).to(self.device)
                }
                
                # 随机时间步
                B = len(batch_ids)
                t = torch.randint(0, self.config['n_timesteps'], (B,), device=self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                pred_noise, noise = self.model(source_lr, target_lr, batch_smile[0], z0, t)
                
                # 计算损失
                loss = self.loss_fn(pred_noise, noise)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # 计算平均损失
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
            
            # 早停检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                # 保存最佳模型
                best_model_path = os.path.join(self.config['output_dir'], "lrldm_best.pt")
                torch.save(best_model_state, best_model_path)
                print(f"New best model saved to {best_model_path}")
            else:
                patience_counter += 1
                print(f"No improvement, patience counter: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    # 加载最佳模型
                    self.model.load_state_dict(best_model_state)
                    break
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                model_path = os.path.join(self.config['output_dir'], f"lrldm_epoch_{epoch+1}.pt")
                torch.save(self.model.state_dict(), model_path)
                print(f"Checkpoint saved to {model_path}")
    
    def generate(self, source_lr: torch.Tensor, target_lr: torch.Tensor, 
                 smile: str, num_samples: int = 1) -> Dict[str, torch.Tensor]:
        """生成新的细胞图谱"""
        self.model.eval()
        
        with torch.no_grad():
            # 提取条件特征
            source_feat = self.model.source_lr_extractor(source_lr)
            target_feat = self.model.target_lr_extractor(target_lr)
            
            # 编码SMILE
            smile_tokens = self.model.encode_smile(smile).repeat(num_samples, 1).to(self.device)
            smile_feat = self.model.smile_encoder(smile_tokens)
            
            # 融合条件
            cond = torch.cat([source_feat, target_feat, smile_feat], dim=-1)
            cond = self.model.condition_fusion(cond)
            
            # 生成（简化版采样）
            z_dim = self.config['n_genes'] + 3  # 表达 + 坐标
            z_t = torch.randn(num_samples, z_dim).to(self.device)
            
            # DDIM采样
            for t in range(self.config['n_timesteps'] - 1, -1, -1):
                t_tensor = torch.tensor([t], device=self.device).repeat(num_samples)
                pred_noise, _ = self.model.diffusion(z_t, cond, cond, t_tensor)
                z_t = z_t - pred_noise * 0.01
            
            # 解码
            expression = z_t[:, :self.config['n_genes']]
            coordinates = z_t[:, self.config['n_genes']:]
            
            return {
                'expression': expression,
                'coordinates': coordinates
            }

def main():
    """主函数"""
    config = {
        'n_cells': 400,
        'n_genes': 978,
        'd_model': 256,
        'latent_dim': 128,
        'smile_vocab_size': 100,
        'n_timesteps': 1000,
        'lr': 1e-4,
        'mapping_file': '/tmp/lrldm_data/mapping_table.csv',
        'output_dir': '/tmp/lrldm_models'
    }
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    print("Starting LRLDM training...")
    print(f"Config: {config}")
    
    # 初始化训练器
    trainer = LRLDMTrainer(config)
    
    # 开始训练
    trainer.train(num_epochs=10, batch_size=4)
    
    print("Training complete!")
    
    # 测试生成
    print("\nTesting generation...")
    sample_data = trainer.load_data('0_0')
    source_lr = sample_data['lr_data']['interaction_matrix'].unsqueeze(0).to(trainer.device)
    target_lr = sample_data['lr_data']['interaction_matrix'].unsqueeze(0).to(trainer.device)
    smile = sample_data['smile']
    
    generated = trainer.generate(source_lr, target_lr, smile, num_samples=1)
    print(f"Generated expression shape: {generated['expression'].shape}")
    print(f"Generated coordinates shape: {generated['coordinates'].shape}")

if __name__ == '__main__':
    main()
