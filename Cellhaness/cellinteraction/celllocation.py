import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class GraphVariationalNetwork(nn.Module):
    """
    图变分网络：从GSVA结果学习细胞间的关系并生成潜在空间表示
    """
    def __init__(self, 
                 input_dim: int = 50,  # GSVA通路数
                 hidden_dim: int = 128,
                 latent_dim: int = 32,
                 n_cell_types: int = 2):  # 星型胶质细胞和少胶质细胞
        super().__init__()
        
        # 细胞类型嵌入
        self.cell_type_embed = nn.Embedding(n_cell_types, hidden_dim)
        
        # 图编码器
        self.graph_encoder = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 变分参数网络
        self.mu_net = nn.Linear(hidden_dim, latent_dim)
        self.logvar_net = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def encode(self, gsva: torch.Tensor, cell_types: torch.Tensor):
        """
        编码GSVA和细胞类型到潜在空间
        """
        # 细胞类型嵌入
        type_embed = self.cell_type_embed(cell_types)
        
        # 融合GSVA和细胞类型信息
        input = torch.cat([gsva, type_embed], dim=-1)
        
        # 图编码
        h = self.graph_encoder(input)
        
        # 计算均值和方差
        mu = self.mu_net(h)
        logvar = self.logvar_net(h)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        重参数化技巧
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor):
        """
        从潜在空间解码回GSVA
        """
        return self.decoder(z)
    
    def forward(self, gsva: torch.Tensor, cell_types: torch.Tensor):
        """
        前向传播
        """
        mu, logvar = self.encode(gsva, cell_types)
        z = self.reparameterize(mu, logvar)
        recon_gsva = self.decode(z)
        return recon_gsva, mu, logvar

def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0):
    """
    计算高斯核函数
    """
    distance = torch.cdist(x, y, p=2)
    return torch.exp(-distance ** 2 / (2 * sigma ** 2))

def generate_coordinates(z: torch.Tensor, n_cells: int, sigma: float = 1.0):
    """
    利用高斯核函数从潜在空间生成细胞坐标
    """
    # 初始化坐标
    coords = torch.randn(n_cells, 3)
    
    # 计算潜在空间中的相似度
    similarity = gaussian_kernel(z, z, sigma)
    
    # 基于相似度更新坐标
    for _ in range(10):  # 迭代优化
        # 计算力
        forces = torch.zeros_like(coords)
        for i in range(n_cells):
            for j in range(n_cells):
                if i != j:
                    diff = coords[j] - coords[i]
                    distance = torch.norm(diff)
                    if distance > 0:
                        force = similarity[i, j] * diff / distance
                        forces[i] += force
        
        # 更新坐标
        coords += 0.1 * forces
        
        # 归一化
        coords = coords / torch.max(torch.abs(coords))
    
    return coords

def smooth_coordinates(coords: torch.Tensor, k: int = 3):
    """
    对坐标进行平滑处理
    """
    n_cells = coords.shape[0]
    smoothed_coords = torch.zeros_like(coords)
    
    for i in range(n_cells):
        # 计算与其他细胞的距离
        distances = torch.cdist(coords[i:i+1], coords).squeeze()
        
        # 找到最近的k个细胞
        _, indices = torch.topk(distances, k, largest=False)
        
        # 平均最近k个细胞的坐标
        smoothed_coords[i] = torch.mean(coords[indices], dim=0)
    
    return smoothed_coords

def visualize_coordinates(coords: torch.Tensor, cell_types: torch.Tensor, output_path: str):
    """
    可视化细胞坐标
    """
    # 转换为numpy数组
    coords_np = coords.detach().cpu().numpy()
    cell_types_np = cell_types.detach().cpu().numpy()
    
    # 绘制3D散点图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 星型胶质细胞 (0)
    astrocytes_idx = cell_types_np == 0
    ax.scatter(coords_np[astrocytes_idx, 0], 
               coords_np[astrocytes_idx, 1], 
               coords_np[astrocytes_idx, 2], 
               c='blue', s=200, label='Astrocytes')
    
    # 少胶质细胞 (1)
    oligodendrocytes_idx = cell_types_np == 1
    ax.scatter(coords_np[oligodendrocytes_idx, 0], 
               coords_np[oligodendrocytes_idx, 1], 
               coords_np[oligodendrocytes_idx, 2], 
               c='red', s=200, label='Oligodendrocytes')
    
    # 设置标题和标签
    ax.set_title('Cell Spatial Distribution', fontsize=16)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_zlabel('Z Coordinate', fontsize=12)
    ax.legend()
    
    # 保存图片
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def train_gvn(gsva_data: Dict, n_epochs: int = 100):
    """
    训练图变分网络
    """
    # 准备数据
    astrocytes_gsva = gsva_data['astrocytes']
    oligodendrocytes_gsva = gsva_data['oligodendrocytes']
    
    # 创建细胞类型标签
    astrocytes_types = torch.zeros(astrocytes_gsva.shape[0], dtype=torch.long)
    oligodendrocytes_types = torch.ones(oligodendrocytes_gsva.shape[0], dtype=torch.long)
    
    # 合并数据
    gsva = torch.cat([astrocytes_gsva, oligodendrocytes_gsva], dim=0)
    cell_types = torch.cat([astrocytes_types, oligodendrocytes_types], dim=0)
    
    # 初始化模型
    model = GraphVariationalNetwork(input_dim=gsva.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # 前向传播
        recon_gsva, mu, logvar = model(gsva, cell_types)
        
        # 计算损失
        recon_loss = F.mse_loss(recon_gsva, gsva)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + 0.01 * kl_loss
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}')
    
    return model

def run_cell_location_pipeline(gsva_data: Dict, output_dir: str = './output'):
    """
    运行细胞定位 pipeline
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 训练图变分网络
    print("Training Graph Variational Network...")
    model = train_gvn(gsva_data)
    
    # 2. 准备数据
    astrocytes_gsva = gsva_data['astrocytes']
    oligodendrocytes_gsva = gsva_data['oligodendrocytes']
    
    # 创建细胞类型标签
    astrocytes_types = torch.zeros(astrocytes_gsva.shape[0], dtype=torch.long)
    oligodendrocytes_types = torch.ones(oligodendrocytes_gsva.shape[0], dtype=torch.long)
    
    # 合并数据
    gsva = torch.cat([astrocytes_gsva, oligodendrocytes_gsva], dim=0)
    cell_types = torch.cat([astrocytes_types, oligodendrocytes_types], dim=0)
    
    # 3. 获取潜在空间表示
    with torch.no_grad():
        mu, logvar = model.encode(gsva, cell_types)
        z = model.reparameterize(mu, logvar)
    
    # 4. 生成细胞坐标
    print("Generating cell coordinates...")
    n_cells = gsva.shape[0]
    coords = generate_coordinates(z, n_cells)
    
    # 5. 平滑坐标
    print("Smoothing coordinates...")
    smoothed_coords = smooth_coordinates(coords)
    
    # 6. 可视化
    print("Visualizing cell distribution...")
    output_path = os.path.join(output_dir, 'cell_location.png')
    visualize_coordinates(smoothed_coords, cell_types, output_path)
    
    print(f"Cell location visualization saved to: {output_path}")
    
    return {
        'coordinates': smoothed_coords,
        'cell_types': cell_types,
        'latent_code': z,
        'visualization_path': output_path
    }

if __name__ == "__main__":
    # 生成模拟数据
    print("Generating synthetic GSVA data...")
    n_astrocytes = 10
    n_oligodendrocytes = 10
    n_pathways = 50
    
    # 生成不同分布的GSVA数据
    astrocytes_gsva = torch.randn(n_astrocytes, n_pathways) * 0.5 + 0.5
    oligodendrocytes_gsva = torch.randn(n_oligodendrocytes, n_pathways) * 0.5 - 0.5
    
    gsva_data = {
        'astrocytes': astrocytes_gsva,
        'oligodendrocytes': oligodendrocytes_gsva
    }
    
    # 运行pipeline
    result = run_cell_location_pipeline(gsva_data)
    print("Pipeline completed successfully!")
