"""
Interaction-Guided Atlas LDM (IGA-LDM)
基于互作关系生成细胞图谱的潜在扩散模型

整合组件:
1. CellInteract-AF3: 预测细胞间互作关系
2. Interaction Condition Encoder: 将互作编码为LDM条件
3. Spatial Diffusion Model: 生成空间化的细胞图谱
4. Atlas Reconstructor: 重构可解释的细胞空间布局

作者: AI Assistant
日期: 2026-04-16
版本: 1.0.0

参考:
- Squidiff: Diffusion model for single-cell perturbation [^38^]
- scLDM: Scalable Single-Cell Generation with LDM [^55^]
- AlphaFold3: Pair representation and triangular attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass

# =============================================================================
# 第一部分: 互作条件编码器 (Interaction Condition Encoder)
# =============================================================================
def acctetp()cell_gsva:
  return gsva
  
@dataclass
class InteractionFeatures:
    """CellInteract-AF3输出的互作特征容器"""
    pathway_matrix: torch.Tensor      # (batch, N, N) 通路级互作强度
    interaction_type: torch.Tensor    # (batch,) 互作类型索引
    direction: torch.Tensor           # (batch,) 方向性索引
    global_score: torch.Tensor        # (batch, 1) 全局互作分数
    pathway_contrib_A: torch.Tensor   # (batch, N) A细胞通路贡献
    pathway_contrib_B: torch.Tensor   # (batch, N) B细胞通路贡献
    attention_matrix: torch.Tensor    # (batch, N, N) 注意力权重


class InteractionConditionEncoder(nn.Module):
    """
    将CellInteract-AF3的互作预测编码为LDM可用的条件向量

    策略:
    1. 将通路互作矩阵压缩为低维条件向量
    2. 嵌入互作类型和方向性
    3. 生成空间位置先验 (基于互作强度推断细胞间距离)
    """
    def __init__(
        self,
        num_pathways: int = 50,
        d_model: int = 256,
        num_interaction_types: int = 4,
        num_directions: int = 4,
        spatial_dim: int = 2  # 2D或3D空间
    ):
        super().__init__()
        self.num_pathways = num_pathways
        self.d_model = d_model
        self.spatial_dim = spatial_dim

        # 1. 通路互作矩阵编码 (使用类似CNN的局部聚合)
        self.pathway_encoder = nn.Sequential(
            # 模拟"卷积": 聚合局部通路互作模式
            nn.Linear(num_pathways * num_pathways, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )

        # 2. 互作类型嵌入
        self.type_embedding = nn.Embedding(num_interaction_types, 64)

        # 3. 方向性嵌入
        self.direction_embedding = nn.Embedding(num_directions, 64)

        # 4. 全局分数MLP
        self.score_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # 5. 融合所有条件
        self.fusion = nn.Sequential(
            nn.Linear(d_model + 64 + 64 + 64, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 6. 空间位置先验生成器 (基于互作强度推断空间距离)
        # 高互作强度 -> 空间邻近 (短距离)
        self.spatial_prior = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, spatial_dim)  # 输出相对位置偏移
        )

    def forward(self, interaction: InteractionFeatures) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            interaction: CellInteract-AF3的输出特征

        Returns:
            condition: (batch, d_model) - 用于LDM的条件向量
            spatial_prior: (batch, spatial_dim) - 细胞间的空间位置先验
        """
        batch = interaction.pathway_matrix.size(0)

        # 1. 编码通路互作矩阵
        pathway_flat = interaction.pathway_matrix.view(batch, -1)
        pathway_emb = self.pathway_encoder(pathway_flat)

        # 2. 嵌入类型和方向
        type_emb = self.type_embedding(interaction.interaction_type)
        dir_emb = self.direction_embedding(interaction.direction)

        # 3. 编码全局分数
        score_emb = self.score_mlp(interaction.global_score)

        # 4. 融合
        combined = torch.cat([pathway_emb, type_emb, dir_emb, score_emb], dim=-1)
        condition = self.fusion(combined)

        # 5. 生成空间先验
        spatial_offset = self.spatial_prior(condition)

        return condition, spatial_offset


# =============================================================================
# 第二部分: 语义VAE (Semantic VAE for scRNA-seq)
# =============================================================================

class SemanticVAE(nn.Module):
    """
    语义VAE: 将单细胞表达编码到潜在空间
    类似Squidiff的架构 [^38^]: 分离语义变量z_sem和随机噪声x_T
    """
    def __init__(
        self,
        input_dim: int = 500,      # 基因数量
        latent_dim: int = 128,     # 语义潜在维度
        hidden_dims: List[int] = [512, 256]
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # 解码器
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码到语义潜在空间"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """从语义潜在空间解码"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        Returns:
            recon: 重建的表达
            mu, logvar: 潜在分布参数
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_function(
        self, 
        recon: torch.Tensor, 
        x: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor,
        beta: float = 1.0
    ) -> torch.Tensor:
        """VAE损失: 重建损失 + KL散度"""
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_loss


# =============================================================================
# 第三部分: 条件化扩散模型 (Conditional Diffusion for Atlas Generation)
# =============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """时间步的正弦位置编码"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalResBlock(nn.Module):
    """条件化残差块: 整合时间步和互作条件"""
    def __init__(self, in_dim: int, cond_dim: int, time_emb_dim: int):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, in_dim)
        )

        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, in_dim)
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, in_dim)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(8, in_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim, in_dim)
        )

        self.residual_conv = nn.Linear(in_dim, in_dim) if in_dim != in_dim else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (batch, in_dim)
            t_emb: 时间步嵌入 (batch, time_emb_dim)
            c_emb: 条件嵌入 (batch, cond_dim)
        """
        h = self.block1(x)

        # 添加时间条件
        t = self.time_mlp(t_emb)
        h = h + t

        # 添加互作条件
        c = self.cond_mlp(c_emb)
        h = h + c

        h = self.block2(h)

        return h + self.residual_conv(x)


class InteractionGuidedDiffusion(nn.Module):
    """
    互作引导的扩散模型
    核心: 在语义潜在空间进行条件化扩散，生成细胞图谱的潜在表示
    """
    def __init__(
        self,
        latent_dim: int = 128,
        cond_dim: int = 256,
        time_emb_dim: int = 128,
        num_steps: int = 1000,
        beta_schedule: str = 'cosine'
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.num_steps = num_steps

        # 时间步嵌入
        self.time_embed = SinusoidalPositionEmbeddings(time_emb_dim)

        # 条件投影
        self.cond_proj = nn.Linear(cond_dim, cond_dim)

        # 去噪网络 (MLP-based for latent space)
        self.denoiser = nn.ModuleList([
            nn.Sequential(
                ConditionalResBlock(latent_dim, cond_dim, time_emb_dim),
                ConditionalResBlock(latent_dim, cond_dim, time_emb_dim)
            ) for _ in range(4)  # 4层去噪块
        ])

        # 最终输出
        self.final_proj = nn.Sequential(
            nn.GroupNorm(8, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # 设置扩散参数
        self._setup_diffusion(beta_schedule)

    def _setup_diffusion(self, schedule: str):
        """设置扩散过程的alpha和beta参数"""
        if schedule == 'cosine':
            # Cosine schedule
            steps = torch.arange(self.num_steps + 1, dtype=torch.float32)
            alphas_cumprod = torch.cos(((steps / self.num_steps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            # Linear schedule
            betas = torch.linspace(0.0001, 0.02, self.num_steps)
            alphas_cumprod = torch.cumprod(1 - betas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

    def forward_diffusion(self, z_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向扩散: q(z_t | z_0)
        """
        if noise is None:
            noise = torch.randn_like(z_0)

        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

        z_t = sqrt_alpha_t * z_0 + sqrt_one_minus_alpha_t * noise
        return z_t, noise

    def reverse_diffusion(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        逆向去噪: p(z_{t-1} | z_t, cond)
        使用条件化去噪网络预测噪声
        """
        # 时间嵌入
        t_emb = self.time_embed(t.float())

        # 条件投影
        c_emb = self.cond_proj(cond)

        # 去噪
        h = z_t
        for block in self.denoiser:
            for layer in block:
                h = layer(h, t_emb, c_emb)

        pred_noise = self.final_proj(h)
        return pred_noise

    def p_sample(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """单步采样 (DDPM风格)"""
        betas_t = self.betas[t].view(-1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        sqrt_recip_alpha_t = torch.sqrt(1.0 / (1 - betas_t))

        # 预测噪声
        pred_noise = self.reverse_diffusion(z_t, t, cond)

        # 计算均值
        model_mean = sqrt_recip_alpha_t * (z_t - betas_t * pred_noise / sqrt_one_minus_alpha_t)

        if t[0] == 0:
            return model_mean
        else:
            variance = betas_t
            noise = torch.randn_like(z_t)
            return model_mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, num_samples: int, device: str = 'cuda') -> torch.Tensor:
        """
        生成样本: 从噪声到结构化潜在表示

        Args:
            cond: (num_samples, cond_dim) - 互作条件
            num_samples: 样本数

        Returns:
            z_0: (num_samples, latent_dim) - 生成的语义潜在表示
        """
        # 从纯噪声开始
        z_t = torch.randn(num_samples, self.latent_dim, device=device)

        # 逐步去噪
        for t in reversed(range(self.num_steps)):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            z_t = self.p_sample(z_t, t_batch, cond)

        return z_t


# =============================================================================
# 第四部分: 空间图谱重构器 (Spatial Atlas Reconstructor)
# =============================================================================

class SpatialAtlasReconstructor(nn.Module):
    """
    将生成的潜在表示重构为空间化的细胞图谱

    核心功能:
    1. 基于互作强度推断细胞空间位置
    2. 生成细胞间的空间邻域关系
    3. 构建可交互的图谱结构
    """
    def __init__(
        self,
        latent_dim: int = 128,
        spatial_dim: int = 2,
        num_genes: int = 500,
        grid_size: int = 32  # 空间网格大小
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.spatial_dim = spatial_dim
        self.num_genes = num_genes
        self.grid_size = grid_size

        # 潜在到空间的映射网络
        self.latent_to_spatial = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, spatial_dim)  # 输出空间坐标
        )

        # 潜在到表达的解码 (如果VAE不可用)
        self.latent_to_expression = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_genes),
            nn.Softplus()  # 保证非负表达值
        )

        # 空间平滑层 (确保邻近细胞表达相似)
        self.spatial_smooth = nn.Conv2d(
            in_channels=num_genes,
            out_channels=num_genes,
            kernel_size=3,
            padding=1,
            bias=False
        )
        # 初始化为平滑核
        with torch.no_grad():
            self.spatial_smooth.weight.fill_(0.0)
            for i in range(num_genes):
                self.spatial_smooth.weight[i, i, :, :] = torch.tensor([
                    [0.1, 0.1, 0.1],
                    [0.1, 0.2, 0.1],
                    [0.1, 0.1, 0.1]
                ])

    def forward(
        self, 
        z_A: torch.Tensor, 
        z_B: torch.Tensor,
        interaction_strength: torch.Tensor,
        vae_decoder: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        重构空间图谱

        Args:
            z_A, z_B: 生成的细胞A/B的潜在表示 (batch, latent_dim)
            interaction_strength: (batch,) 互作强度
            vae_decoder: 可选的外部VAE解码器

        Returns:
            atlas: 包含空间坐标、表达、邻接关系的字典
        """
        batch = z_A.size(0)
        device = z_A.device

        # 1. 生成空间坐标
        coords_A = self.latent_to_spatial(z_A)  # (batch, spatial_dim)
        coords_B = self.latent_to_spatial(z_B)

        # 基于互作强度调整位置: 高强度 -> 近距离
        # 计算B相对于A的位置偏移
        distance = 1.0 / (interaction_strength.unsqueeze(1) + 0.1)  # 反比关系
        direction = torch.randn(batch, self.spatial_dim, device=device)
        direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-8)

        # B的位置 = A的位置 + 方向 * 距离
        coords_B_adjusted = coords_A + direction * distance

        # 2. 解码基因表达
        if vae_decoder is not None:
            expr_A = vae_decoder(z_A)
            expr_B = vae_decoder(z_B)
        else:
            expr_A = self.latent_to_expression(z_A)
            expr_B = self.latent_to_expression(z_B)

        # 3. 构建空间网格表示 (用于可视化)
        grid = torch.zeros(batch, self.num_genes, self.grid_size, self.grid_size, device=device)

        # 将细胞放置到网格 (简化: 最近邻分配)
        for b in range(batch):
            # 归一化坐标到网格
            x_A = int((coords_A[b, 0] + 1) / 2 * (self.grid_size - 1))
            y_A = int((coords_A[b, 1] + 1) / 2 * (self.grid_size - 1))
            x_B = int((coords_B_adjusted[b, 0] + 1) / 2 * (self.grid_size - 1))
            y_B = int((coords_B_adjusted[b, 1] + 1) / 2 * (self.grid_size - 1))

            # 裁剪到边界
            x_A, y_A = max(0, min(x_A, self.grid_size-1)), max(0, min(y_A, self.grid_size-1))
            x_B, y_B = max(0, min(x_B, self.grid_size-1)), max(0, min(y_B, self.grid_size-1))

            grid[b, :, y_A, x_A] = expr_A[b]
            grid[b, :, y_B, x_B] = expr_B[b]

        # 4. 应用空间平滑
        grid_smooth = self.spatial_smooth(grid)

        # 5. 构建邻接矩阵 (基于空间距离)
        all_coords = torch.stack([coords_A, coords_B_adjusted], dim=1)  # (batch, 2, spatial_dim)

        # 计算两细胞间距离
        dist_matrix = torch.cdist(all_coords, all_coords)  # (batch, 2, 2)

        # 邻接: 距离小于阈值则连接
        adjacency = (dist_matrix < 0.5).float()

        return {
            'coordinates': all_coords,  # (batch, 2, spatial_dim)
            'expression': torch.stack([expr_A, expr_B], dim=1),  # (batch, 2, num_genes)
            'spatial_grid': grid_smooth,  # (batch, num_genes, H, W)
            'adjacency_matrix': adjacency,  # (batch, 2, 2)
            'distance_matrix': dist_matrix,
            'cell_labels': torch.tensor([[0, 1]] * batch, device=device)  # 0=A, 1=B
        }


# =============================================================================
# 第五部分: 完整的IGA-LDM系统 (Interaction-Guided Atlas LDM)
# =============================================================================

class IGALDM(nn.Module):
    """
    完整的互作引导图谱LDM系统

    整合流程:
    CellInteract-AF3 -> Interaction Encoder -> LDM -> Spatial Reconstructor
    """
    def __init__(
        self,
        num_pathways: int = 50,
        num_genes: int = 500,
        latent_dim: int = 128,
        cond_dim: int = 256,
        spatial_dim: int = 2,
        vae_hidden_dims: List[int] = [512, 256]
    ):
        super().__init__()

        # 1. 互作条件编码器
        self.interaction_encoder = InteractionConditionEncoder(
            num_pathways=num_pathways,
            d_model=cond_dim,
            spatial_dim=spatial_dim
        )

        # 2. 语义VAE
        self.vae = SemanticVAE(
            input_dim=num_genes,
            latent_dim=latent_dim,
            hidden_dims=vae_hidden_dims
        )

        # 3. 条件化扩散模型
        self.diffusion = InteractionGuidedDiffusion(
            latent_dim=latent_dim,
            cond_dim=cond_dim
        )

        # 4. 空间重构器
        self.reconstructor = SpatialAtlasReconstructor(
            latent_dim=latent_dim,
            spatial_dim=spatial_dim,
            num_genes=num_genes
        )

    def forward(
        self,
        gsva_A: torch.Tensor,
        gsva_B: torch.Tensor,
        expr_A: Optional[torch.Tensor] = None,
        expr_B: Optional[torch.Tensor] = None,
        mode: str = 'train'
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            gsva_A, gsva_B: GSVA通路活性 (batch, num_pathways)
            expr_A, expr_B: 可选的真实表达用于VAE训练 (batch, num_genes)
            mode: 'train' 或 'sample'

        Returns:
            包含所有中间结果和最终图谱的字典
        """
        batch = gsva_A.size(0)
        device = gsva_A.device

        # 步骤1: 使用CellInteract-AF3预测互作 (这里模拟输出)
        # 实际使用时应该调用真实的CellInteract-AF3模型
        interaction = self._simulate_interaction_prediction(gsva_A, gsva_B)

        # 步骤2: 编码互作为条件
        cond, spatial_offset = self.interaction_encoder(interaction)

        if mode == 'train' and expr_A is not None and expr_B is not None:
            # 训练模式: VAE编码 + 扩散训练

            # VAE编码
            z_A_mu, z_A_logvar = self.vae.encode(expr_A)
            z_B_mu, z_B_logvar = self.vae.encode(expr_B)

            z_A = self.vae.reparameterize(z_A_mu, z_A_logvar)
            z_B = self.vae.reparameterize(z_B_mu, z_B_logvar)

            # 扩散前向过程
            t = torch.randint(0, self.diffusion.num_steps, (batch,), device=device)

            z_A_noisy, noise_A = self.diffusion.forward_diffusion(z_A, t)
            z_B_noisy, noise_B = self.diffusion.forward_diffusion(z_B, t)

            # 预测噪声
            pred_noise_A = self.diffusion.reverse_diffusion(z_A_noisy, t, cond)
            pred_noise_B = self.diffusion.reverse_diffusion(z_B_noisy, t, cond)

            # 计算损失
            diffusion_loss = F.mse_loss(pred_noise_A, noise_A) + F.mse_loss(pred_noise_B, noise_B)

            # VAE重建
            recon_A = self.vae.decode(z_A)
            recon_B = self.vae.decode(z_B)
            vae_loss = self.vae.loss_function(recon_A, expr_A, z_A_mu, z_A_logvar) +                       self.vae.loss_function(recon_B, expr_B, z_B_mu, z_B_logvar)

            return {
                'diffusion_loss': diffusion_loss,
                'vae_loss': vae_loss,
                'total_loss': diffusion_loss + 0.1 * vae_loss,
                'z_A': z_A,
                'z_B': z_B
            }

        else:
            # 采样模式: 生成新的细胞状态
            with torch.no_grad():
                # 从扩散模型采样
                z_A_gen = self.diffusion.sample(cond, batch, device)
                z_B_gen = self.diffusion.sample(cond, batch, device)

                # 空间重构
                atlas = self.reconstructor(
                    z_A_gen, 
                    z_B_gen, 
                    interaction.global_score.squeeze(),
                    vae_decoder=self.vae.decode
                )

                return {
                    'atlas': atlas,
                    'z_A': z_A_gen,
                    'z_B': z_B_gen,
                    'interaction': interaction,
                    'condition': cond
                }

    def _simulate_interaction_prediction(
        self, 
        gsva_A: torch.Tensor, 
        gsva_B: torch.Tensor
    ) -> InteractionFeatures:
        """
        模拟CellInteract-AF3的输出 (实际使用时替换为真实模型)
        """
        batch = gsva_A.size(0)
        N = gsva_A.size(1)
        device = gsva_A.device

        # 模拟通路互作矩阵 (基于GSVA相关性)
        pathway_matrix = torch.bmm(gsva_A.unsqueeze(2), gsva_B.unsqueeze(1))
        pathway_matrix = torch.sigmoid(pathway_matrix)

        # 模拟其他特征
        interaction_type = torch.randint(0, 4, (batch,), device=device)
        direction = torch.randint(0, 4, (batch,), device=device)
        global_score = torch.rand(batch, 1, device=device)
        pathway_contrib_A = torch.rand(batch, N, device=device)
        pathway_contrib_B = torch.rand(batch, N, device=device)
        attention_matrix = pathway_matrix

        return InteractionFeatures(
            pathway_matrix=pathway_matrix,
            interaction_type=interaction_type,
            direction=direction,
            global_score=global_score,
            pathway_contrib_A=pathway_contrib_A,
            pathway_contrib_B=pathway_contrib_B,
            attention_matrix=attention_matrix
        )

    @torch.no_grad()
    def generate_atlas(
        self,
        gsva_A: torch.Tensor,
        gsva_B: torch.Tensor,
        num_samples: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        生成细胞图谱的便捷接口

        Args:
            gsva_A, gsva_B: GSVA通路活性
            num_samples: 生成样本数

        Returns:
            atlas: 包含空间坐标、表达、邻接关系的字典
        """
        # 重复输入以生成多个样本
        gsva_A = gsva_A.repeat(num_samples, 1)
        gsva_B = gsva_B.repeat(num_samples, 1)

        return self.forward(gsva_A, gsva_B, mode='sample')


# =============================================================================
# 第六部分: 可视化工具 (Visualization Tools)
# =============================================================================

def visualize_cell_atlas(
    atlas: Dict[str, torch.Tensor],
    pathway_names: Optional[List[str]] = None,
    gene_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    可视化生成的细胞图谱

    生成包含以下内容的图表:
    1. 空间散点图 (细胞位置)
    2. 邻接关系图 (细胞间连接)
    3. 基因表达热图
    4. 通路活性对比
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import FancyArrowPatch
    except ImportError:
        print("请安装matplotlib和seaborn: pip install matplotlib seaborn")
        return

    coords = atlas['coordinates'][0].cpu().numpy()  # (2, spatial_dim)
    adjacency = atlas['adjacency_matrix'][0].cpu().numpy()  # (2, 2)
    expression = atlas['expression'][0].cpu().numpy()  # (2, num_genes)
    labels = atlas['cell_labels'][0].cpu().numpy()  # (2,)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. 空间布局图
    ax1 = fig.add_subplot(gs[0, :2])
    colors = ['#1f77b4', '#ff7f0e']
    names = ['Cell A', 'Cell B']

    for i in range(2):
        ax1.scatter(coords[i, 0], coords[i, 1], 
                   c=colors[i], s=500, alpha=0.7, 
                   edgecolors='black', linewidth=2,
                   label=names[i], zorder=5)
        ax1.annotate(names[i], (coords[i, 0], coords[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=12, fontweight='bold')

    # 绘制连接边
    if adjacency[0, 1] > 0:
        ax1.plot([coords[0, 0], coords[1, 0]], 
                [coords[0, 1], coords[1, 1]], 
                'k--', alpha=0.5, linewidth=2, zorder=1)
        mid_x = (coords[0, 0] + coords[1, 0]) / 2
        mid_y = (coords[0, 1] + coords[1, 1]) / 2
        ax1.annotate(f'Interact: {adjacency[0,1]:.2f}', 
                    (mid_x, mid_y), fontsize=10, ha='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    ax1.set_xlabel('Spatial X')
    ax1.set_ylabel('Spatial Y')
    ax1.set_title('Cell Atlas Spatial Layout', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # 2. 基因表达对比 (前20个基因)
    ax2 = fig.add_subplot(gs[0, 2])
    n_genes_show = min(20, expression.shape[1])
    expr_subset = expression[:, :n_genes_show]

    x = np.arange(n_genes_show)
    width = 0.35
    ax2.bar(x - width/2, expr_subset[0], width, label='Cell A', alpha=0.8, color=colors[0])
    ax2.bar(x + width/2, expr_subset[1], width, label='Cell B', alpha=0.8, color=colors[1])
    ax2.set_xlabel('Gene Index')
    ax2.set_ylabel('Expression Level')
    ax2.set_title('Gene Expression Comparison', fontsize=12, fontweight='bold')
    ax2.legend()

    # 3. 空间网格热图 (选定基因)
    ax3 = fig.add_subplot(gs[1, :])
    grid = atlas['spatial_grid'][0].cpu().numpy()  # (num_genes, H, W)

    # 选择前3个基因展示
    n_show = min(3, grid.shape[0])
    for i in range(n_show):
        ax_sub = fig.add_subplot(gs[1, i])
        sns.heatmap(grid[i], ax=ax_sub, cmap='viridis', cbar=True)
        ax_sub.set_title(f'Gene {i} Spatial Pattern', fontsize=10)

    # 4. 邻接矩阵可视化
    ax4 = fig.add_subplot(gs[2, 0])
    sns.heatmap(adjacency, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['A', 'B'], yticklabels=['A', 'B'], ax=ax4)
    ax4.set_title('Adjacency Matrix', fontsize=12, fontweight='bold')

    # 5. 细胞间距离
    ax5 = fig.add_subplot(gs[2, 1])
    dist_matrix = atlas['distance_matrix'][0].cpu().numpy()
    sns.heatmap(dist_matrix, annot=True, fmt='.3f', cmap='Reds_r',
                xticklabels=['A', 'B'], yticklabels=['A', 'B'], ax=ax5)
    ax5.set_title('Distance Matrix', fontsize=12, fontweight='bold')

    # 6. 统计信息
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    info_text = f"""
    Atlas Statistics:

    Cell A Position: ({coords[0,0]:.3f}, {coords[0,1]:.3f})
    Cell B Position: ({coords[1,0]:.3f}, {coords[1,1]:.3f})

    Euclidean Distance: {dist_matrix[0,1]:.3f}
    Adjacency Weight: {adjacency[0,1]:.3f}

    Mean Expr A: {expression[0].mean():.3f}
    Mean Expr B: {expression[1].mean():.3f}

    Generated by IGA-LDM
    """
    ax6.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Interaction-Guided Cell Atlas Generation', fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图谱已保存至: {save_path}")

    plt.show()


# =============================================================================
# 第七部分: 训练与评估 (Training & Evaluation)
# =============================================================================

def train_iga_ldm(
    model: IGALDM,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int = 100,
    device: str = 'cuda'
) -> List[float]:
    """
    训练IGA-LDM模型

    训练策略:
    1. 联合训练VAE和扩散模型
    2. 使用真实的细胞对数据
    3. 优化重建质量和生成多样性
    """
    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0

        for batch in dataloader:
            gsva_A = batch['gsva_A'].to(device)
            gsva_B = batch['gsva_B'].to(device)
            expr_A = batch['expr_A'].to(device)
            expr_B = batch['expr_B'].to(device)

            optimizer.zero_grad()

            outputs = model(gsva_A, gsva_B, expr_A, expr_B, mode='train')
            loss = outputs['total_loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, "
                  f"Diffusion: {outputs['diffusion_loss'].item():.4f}, "
                  f"VAE: {outputs['vae_loss'].item():.4f}")

    return losses


def evaluate_atlas_quality(
    model: IGALDM,
    test_data: torch.utils.data.Dataset,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    评估生成的图谱质量

    指标:
    1. 空间合理性 (细胞间距离分布)
    2. 表达保真度 (与真实细胞的相似性)
    3. 互作一致性 (生成的邻接关系与输入GSVA的相关性)
    """
    model.eval()
    metrics = {
        'mean_distance': [],
        'expression_correlation': [],
        'adjacency_accuracy': []
    }

    with torch.no_grad():
        for batch in test_data:
            gsva_A = batch['gsva_A'].unsqueeze(0).to(device)
            gsva_B = batch['gsva_B'].unsqueeze(0).to(device)

            result = model.generate_atlas(gsva_A, gsva_B, num_samples=1)
            atlas = result['atlas']

            # 空间距离
            dist = atlas['distance_matrix'][0, 0, 1].item()
            metrics['mean_distance'].append(dist)

            # 这里可以添加更多评估指标...

    return {
        'mean_distance': np.mean(metrics['mean_distance']),
        'std_distance': np.std(metrics['mean_distance'])
    }


# =============================================================================
# 第八部分: 使用示例 (Usage Example)
# =============================================================================

def example_usage():
    """
    完整的使用示例: 从GSVA到细胞图谱
    """
    print("=" * 70)
    print("Interaction-Guided Atlas LDM (IGA-LDM) 演示")
    print("=" * 70)

    # 1. 初始化模型
    print("\n[1] 初始化IGA-LDM模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = IGALDM(
        num_pathways=50,
        num_genes=500,
        latent_dim=128,
        cond_dim=256,
        spatial_dim=2
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")
    print(f"使用设备: {device}")

    # 2. 模拟输入数据
    print("\n[2] 准备模拟数据...")
    batch_size = 4
    num_pathways = 50
    num_genes = 500

    # GSVA通路活性 (通常来自GSEApy)
    gsva_A = torch.randn(batch_size, num_pathways).to(device)
    gsva_B = torch.randn(batch_size, num_pathways).to(device)

    # 标准化
    gsva_A = (gsva_A - gsva_A.mean(dim=1, keepdim=True)) / (gsva_A.std(dim=1, keepdim=True) + 1e-8)
    gsva_B = (gsva_B - gsva_B.mean(dim=1, keepdim=True)) / (gsva_B.std(dim=1, keepdim=True) + 1e-8)

    print(f"GSVA A形状: {gsva_A.shape}")
    print(f"GSVA B形状: {gsva_B.shape}")

    # 3. 生成细胞图谱
    print("\n[3] 生成细胞图谱...")
    model.eval()

    with torch.no_grad():
        result = model.generate_atlas(gsva_A[0:1], gsva_B[0:1], num_samples=1)
        atlas = result['atlas']

        print(f"生成坐标形状: {atlas['coordinates'].shape}")
        print(f"生成表达形状: {atlas['expression'].shape}")
        print(f"空间网格形状: {atlas['spatial_grid'].shape}")
        print(f"邻接矩阵: \n{atlas['adjacency_matrix'][0]}")
        print(f"距离矩阵: \n{atlas['distance_matrix'][0]}")

    # 4. 可视化
    print("\n[4] 可视化图谱...")
    try:
        visualize_cell_atlas(atlas, save_path='cell_atlas_example.png')
    except Exception as e:
        print(f"可视化跳过: {e}")

    # 5. 训练演示 (可选)
    print("\n[5] 训练模式演示...")
    expr_A = torch.rand(batch_size, num_genes).to(device)
    expr_B = torch.rand(batch_size, num_genes).to(device)

    model.train()
    outputs = model(gsva_A, gsva_B, expr_A, expr_B, mode='train')

    print(f"扩散损失: {outputs['diffusion_loss'].item():.4f}")
    print(f"VAE损失: {outputs['vae_loss'].item():.4f}")
    print(f"总损失: {outputs['total_loss'].item():.4f}")

    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)

    return model, atlas


if __name__ == "__main__":
    # 运行示例
    model, atlas = example_usage()
