import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

class GSVAEmbedding(nn.Module):
    """
    将每个细胞的GSVA谱编码为单细胞表示
    类比: AF3的氨基酸嵌入 + 位置编码
    """
    def __init__(self,
                 n_pathways: int = 50,      # GSVA通路数
                 d_model: int = 256,         # 嵌入维度
                 n_cell_types: int = 100,    # 预定义细胞类型数
                 max_cells: int = 1000):     # 最大细胞数（位置编码）
        super().__init__()
        
        self.d_model = d_model
        
        # 1. GSVA投影: P维 → d_model维
        self.gsva_proj = nn.Sequential(
            nn.Linear(n_pathways, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, d_model)
        )
        
        # 2. 细胞类型嵌入 (可学习或预定义)
        self.cell_type_embed = nn.Embedding(n_cell_types, d_model)
        
        # 3. 位置编码 (细胞在组织中的相对位置)
        # 使用正弦位置编码，但这里是"细胞序号"而非空间坐标
        self.pos_encoding = self._create_sinusoidal_positions(max_cells, d_model)
        
        # 4. 组织上下文嵌入 (肝/脑/肿瘤等)
        self.tissue_embed = nn.Embedding(20, d_model)  # 20种组织类型
        
    def _create_sinusoidal_positions(self, max_len, d_model):
        """正弦位置编码"""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe, requires_grad=False)
    
    def forward(self,
                gsva_batch: torch.Tensor,      # (B, N, P) 批次×细胞数×通路数
                cell_types: torch.Tensor,       # (B, N) 细胞类型索引
                tissue_id: torch.Tensor,        # (B,) 组织类型
                cell_order: torch.Tensor):      # (B, N) 细胞序列位置
        """
        输出: H_single ∈ ℝ^(B, N, d_model)
        """
        B, N, P = gsva_batch.shape
        
        # GSVA特征投影
        h_gsva = self.gsva_proj(gsva_batch)  # (B, N, d_model)
        
        # 细胞类型嵌入
        h_type = self.cell_type_embed(cell_types)  # (B, N, d_model)
        
        # 位置编码 (按细胞在序列中的顺序)
        h_pos = self.pos_encoding[cell_order]  # (B, N, d_model)
        
        # 组织上下文
        h_tissue = self.tissue_embed(tissue_id).unsqueeze(1)  # (B, 1, d_model)
        
        # 融合: 残差连接 + LayerNorm
        h = h_gsva + h_type + h_pos + h_tissue
        h = nn.LayerNorm(self.d_model).to(h.device)(h)
        
        return h


class RelativePositionEncoding(nn.Module):
    """
    细胞间的相对位置编码 (用于PairFormer)
    类比: AF3的相对位置编码 (残基对距离)
    """
    def __init__(self, d_pair: int = 128, max_dist: int = 100):
        super().__init__()
        self.d_pair = d_pair
        
        # 将欧氏距离分桶编码 (模仿AF3的残基距离分桶)
        self.dist_bins = torch.linspace(0, max_dist, 32)  # 32个距离桶
        
        # 方向编码 (如果已知空间方向)
        self.angle_embed = nn.Linear(3, d_pair // 4)  # 3D方向向量
        
        # 最终投影
        self.proj = nn.Sequential(
            nn.Linear(32 + d_pair//4, d_pair),
            nn.ReLU(),
            nn.LayerNorm(d_pair)
        )
        
    def forward(self, spatial_coords: torch.Tensor):
        """
        spatial_coords: (B, N, 3) 细胞3D坐标
        输出: rel_pos_enc ∈ ℝ^(B, N, N, d_pair)
        """
        B, N, _ = spatial_coords.shape
        
        # 计算细胞间距离矩阵
        diff = spatial_coords.unsqueeze(2) - spatial_coords.unsqueeze(1)  # (B, N, N, 3)
        dist = torch.norm(diff, dim=-1)  # (B, N, N)
        
        # 距离分桶 (one-hot-ish)
        dist_bucketed = torch.bucketize(dist, self.dist_bins.to(dist.device))  # (B, N, N)
        dist_embed = nn.functional.one_hot(dist_bucketed, num_classes=32).float()  # (B, N, N, 32)
        
        # 方向编码 (单位向量)
        direction = diff / (dist.unsqueeze(-1) + 1e-8)  # (B, N, N, 3)
        dir_embed = self.angle_embed(direction)  # (B, N, N, d_pair//4)
        
        # 拼接并投影
        combined = torch.cat([dist_embed, dir_embed], dim=-1)
        return self.proj(combined)

class PairFormerBlock(nn.Module):
    """
    模仿AF3的PairFormer:
    - 单细胞表示 (类比MSA) 通过Triangular Attention更新
    - 细胞对表示 (类比Pair Rep) 通过双向注意力更新
    
    核心创新: 将蛋白质残基对替换为细胞对
    """
    def __init__(self,
                 d_single: int = 256,
                 d_pair: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 4):
        super().__init__()
        
        self.n_layers = n_layers
        
        # 单细胞→细胞对投影 (用于Triangular Update)
        self.single_to_pair = nn.Linear(d_single * 2, d_pair)
        
        # Triangular Multiplicative Update (核心)
        # 模仿AF3: 通过第三条边更新一条边
        self.tri_mul_out = TriangularMultiplicativeUpdate(d_pair, mode='outgoing')
        self.tri_mul_in = TriangularMultiplicativeUpdate(d_pair, mode='incoming')
        
        # Triangular Self-Attention
        self.tri_att_start = TriangleAttention(d_pair, n_heads, mode='starting')
        self.tri_att_end = TriangleAttention(d_pair, n_heads, mode='ending')
        
        # 单细胞表示更新 (类比MSA更新)
        self.msa_att = nn.MultiheadAttention(d_single, n_heads, batch_first=True)
        
        # Transition层 (FFN)
        self.pair_transition = nn.Sequential(
            nn.Linear(d_pair, d_pair * 4),
            nn.ReLU(),
            nn.Linear(d_pair * 4, d_pair)
        )
        self.single_transition = nn.Sequential(
            nn.Linear(d_single, d_single * 4),
            nn.ReLU(),
            nn.Linear(d_single * 4, d_single)
        )
        
        # LayerNorms
        self.norm_single = nn.LayerNorm(d_single)
        self.norm_pair = nn.LayerNorm(d_pair)
        
    def forward(self,
                h_single: torch.Tensor,    # (B, N, d_single)
                h_pair: torch.Tensor,      # (B, N, N, d_pair)
                mask: Optional[torch.Tensor] = None):
        """
        输出: 更新后的 (h_single, h_pair)
        """
        for layer in range(self.n_layers):
            # --- 1. 从单细胞表示更新细胞对表示 ---
            # 拼接每对细胞的单细胞表示
            h_i = h_single.unsqueeze(2).expand(-1, -1, h_single.size(1), -1)  # (B, N, N, d)
            h_j = h_single.unsqueeze(1).expand(-1, h_single.size(1), -1, -1)  # (B, N, N, d)
            pair_update = self.single_to_pair(torch.cat([h_i, h_j], dim=-1))  # (B, N, N, d_pair)
            
            h_pair = h_pair + pair_update
            h_pair = self.norm_pair(h_pair)
            
            # --- 2. Triangular Multiplicative Update ---
            h_pair = h_pair + self.tri_mul_out(h_pair, mask)
            h_pair = h_pair + self.tri_mul_in(h_pair, mask)
            h_pair = self.norm_pair(h_pair)
            
            # --- 3. Triangular Self-Attention ---
            h_pair = h_pair + self.tri_att_start(h_pair, mask)
            h_pair = h_pair + self.tri_att_end(h_pair, mask)
            h_pair = self.norm_pair(h_pair)
            
            # --- 4. Pair Transition (FFN) ---
            h_pair = h_pair + self.pair_transition(h_pair)
            h_pair = self.norm_pair(h_pair)
            
            # --- 5. 从细胞对表示更新单细胞表示 ---
            # 平均池化所有配对信息
            pair_to_single = h_pair.mean(dim=2)  # (B, N, d_pair)
            # 投影到单细胞维度
            pair_to_single_proj = nn.Linear(self.d_pair, self.d_single).to(h_single.device)(pair_to_single)
            
            # 自注意力 + 配对信息
            attn_out, _ = self.msa_att(h_single, h_single, h_single, key_padding_mask=mask)
            h_single = h_single + attn_out + pair_to_single_proj
            h_single = self.norm_single(h_single)
            
            # --- 6. Single Transition ---
            h_single = h_single + self.single_transition(h_single)
            h_single = self.norm_single(h_single)
            
        return h_single, h_pair


class TriangularMultiplicativeUpdate(nn.Module):
    """
    AF3核心: 三角乘法更新
    对于细胞(i,j)，通过细胞k作为"桥接"更新
    
    outgoing: 固定i,j, 对k求和:  h_pair[i,j] += h_pair[i,k] * h_pair[k,j]
    incoming: 固定i,j, 对k求和:  h_pair[i,j] += h_pair[k,i] * h_pair[j,k]
    
    生物学意义: 细胞i和j的互作，通过共同邻居k间接调制
    """
    def __init__(self, d_pair: int, mode: str = 'outgoing'):
        super().__init__()
        self.mode = mode
        
        # 门控投影
        self.gate_i = nn.Linear(d_pair, d_pair, bias=False)
        self.gate_j = nn.Linear(d_pair, d_pair, bias=False)
        
        # 输出投影
        self.out_proj = nn.Linear(d_pair, d_pair, bias=False)
        
        self.norm = nn.LayerNorm(d_pair)
        
    def forward(self, h_pair: torch.Tensor, mask=None):
        B, N, _, d = h_pair.shape
        
        # 门控
        gate_i = torch.sigmoid(self.gate_i(h_pair))  # (B, N, N, d)
        gate_j = torch.sigmoid(self.gate_j(h_pair))  # (B, N, N, d)
        
        if self.mode == 'outgoing':
            # 对k求和: gate_i[i,k] * gate_j[k,j]
            # h_pair[i,k]作为query, h_pair[k,j]作为key
            update = torch.einsum('bnkd,bkmd->bnmd', 
                                 gate_i,  # (B, N, K, d)
                                 gate_j)  # (B, K, N, d) -> 转置后 (B, K, N, d)
            # 实际上需要更仔细的对齐...
            # 简化实现:
            left = gate_i.unsqueeze(3)  # (B, N, K, 1, d)
            right = gate_j.unsqueeze(1)  # (B, 1, K, N, d)
            update = (left * right).sum(dim=2)  # 对K求和 -> (B, N, N, d)
            
        else:  # incoming
            left = gate_i.permute(0, 2, 1, 3).unsqueeze(3)  # (B, K, N, 1, d)
            right = gate_j.permute(0, 2, 1, 3).unsqueeze(1)  # (B, 1, N, K, d)
            update = (left * right).sum(dim=2)  # (B, N, K, d) -> permute...
            # 需要更仔细实现，此处简化
            
        update = self.out_proj(update)
        return self.norm(update)


class TriangleAttention(nn.Module):
    """
    三角注意力: 沿三角形的一条边做注意力
    """
    def __init__(self, d_pair: int, n_heads: int, mode: str = 'starting'):
        super().__init__()
        self.mode = mode
        self.attention = nn.MultiheadAttention(d_pair, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_pair)
        
    def forward(self, h_pair: torch.Tensor, mask=None):
        B, N, _, d = h_pair.shape
        
        if self.mode == 'starting':
            # 对每对(i,j)，沿"starting edge" (i,*)做注意力
            # 即: 固定i, 对所有k, 计算attn(h_pair[i,j], h_pair[i,k])
            h = h_pair.reshape(B * N, N, d)  # 把(B,N,N,d)展平为(B*N, N, d)
            # 实际上需要更复杂的reshape...
            pass
        else:  # ending
            # 固定j, 对所有k
            pass
            
        # 简化实现: 使用标准注意力近似
        attn_out, _ = self.attention(h_pair, h_pair, h_pair)
        return self.norm(attn_out)
class CellInteractionExtractor(nn.Module):
    """
    从PairFormer输出的细胞对表示中提取生物学互作信息
    
    输出:
    1. 互作强度矩阵 (用于空间布局)
    2. 配体-受体预测 (用于分子机制)
    3. 通路串扰模式 (用于功能解释)
    """
    def __init__(self,
                 d_pair: int = 128,
                 n_ligands: int = 500,      # 预定义配体数
                 n_receptors: int = 500,    # 预定义受体数
                 n_pathways: int = 50):
        super().__init__()
        
        # 1. 互作强度分类 (无互作/旁分泌/近分泌/直接接触)
        self.interaction_type = nn.Sequential(
            nn.Linear(d_pair, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4种互作类型
        )
        
        # 2. 配体-受体对预测
        self.ligand_head = nn.Linear(d_pair, n_ligands)
        self.receptor_head = nn.Linear(d_pair, n_receptors)
        
        # 3. 通路串扰预测 (哪些通路在细胞对间传递信号)
        self.pathway_crosstalk = nn.Sequential(
            nn.Linear(d_pair, 256),
            nn.ReLU(),
            nn.Linear(256, n_pathways * n_pathways)  # 通路间串扰矩阵
        )
        
        # 4. 空间距离预测 (用于后续空间布局)
        self.distance_pred = nn.Sequential(
            nn.Linear(d_pair, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # 保证正距离
        )
        
    def forward(self, h_pair: torch.Tensor):
        """
        h_pair: (B, N, N, d_pair)
        
        返回: Dict 包含所有互作信息
        """
        B, N, _, d = h_pair.shape
        
        # 互作类型 (排除对角线自互作)
        mask = ~torch.eye(N, dtype=torch.bool).unsqueeze(0).unsqueeze(-1).to(h_pair.device)
        h_pair_masked = h_pair * mask.float()
        
        # 1. 互作类型概率
        interact_type = self.interaction_type(h_pair_masked)  # (B, N, N, 4)
        
        # 2. 配体-受体预测 (对每对细胞)
        ligand_scores = self.ligand_head(h_pair_masked)  # (B, N, N, n_ligands)
        receptor_scores = self.receptor_head(h_pair_masked)  # (B, N, N, n_receptors)
        
        # 3. 通路串扰 (reshape为矩阵)
        crosstalk_flat = self.pathway_crosstalk(h_pair_masked)  # (B, N, N, P*P)
        crosstalk = crosstalk_flat.reshape(B, N, N, -1, -1)  # (B, N, N, P, P)
        
        # 4. 预测距离
        pred_dist = self.distance_pred(h_pair_masked).squeeze(-1)  # (B, N, N)
        
        return {
            'interaction_type': interact_type,      # 互作类型概率
            'ligand_scores': ligand_scores,          # 配体表达预测
            'receptor_scores': receptor_scores,      # 受体表达预测
            'pathway_crosstalk': crosstalk,          # 通路串扰矩阵
            'predicted_distance': pred_dist,          # 预测细胞间距
            'attention_weights': torch.softmax(h_pair_masked.sum(-1), dim=-1)  # 注意力作为互作强度
        }
class Z0Generator(nn.Module):
    """
    从PairFormer输出生成扩散模型的初始噪声Z_0
    
    关键: Z_0不是纯噪声，而是包含结构先验的"粗粒度"细胞图谱
    类比: AF3的结构模块输出粗结构，再refine
    """
    def __init__(self,
                 d_single: int = 256,
                 d_pair: int = 128,
                 n_genes: int = 978,        # LINCS L1000基因数
                 spatial_dim: int = 3):     # 3D空间
        super().__init__()
        
        self.n_genes = n_genes
        self.spatial_dim = spatial_dim
        
        # 1. 细胞空间坐标生成 (基于互作距离)
        self.coord_generator = nn.Sequential(
            nn.Linear(d_single + d_pair, 256),
            nn.ReLU(),
            nn.Linear(256, spatial_dim)  # (x, y, z)
        )
        
        # 2. 基因表达初始化 (基于单细胞表示)
        self.expression_generator = nn.Sequential(
            nn.Linear(d_single, 512),
            nn.ReLU(),
            nn.Linear(512, n_genes)  # 978维基因表达
        )
        
        # 3. 空间-表达耦合 (细胞环境影响基因表达)
        self.spatial_expression_modulator = nn.Sequential(
            nn.Linear(d_pair + spatial_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_genes)
        )
        
    def forward(self,
                h_single: torch.Tensor,      # (B, N, d_single)
                h_pair: torch.Tensor,        # (B, N, N, d_pair)
                interact_info: Dict):
        """
        生成Z_0: (B, N, n_genes, spatial_dim+1) 
        实际上表示为 (B, N, G) 表达 + (B, N, 3) 坐标
        """
        B, N, d_s = h_single.shape
        _, _, _, d_p = h_pair.shape
        
        # --- 1. 生成空间坐标 ---
        # 每对细胞的互作信息聚合
        pair_agg = h_pair.mean(dim=2)  # (B, N, d_pair) 对所有邻居平均
        
        coord_input = torch.cat([h_single, pair_agg], dim=-1)
        coords = self.coord_generator(coord_input)  # (B, N, 3)
        
        # 使用预测距离约束坐标 (力导向布局的神经网络版)
        pred_dist = interact_info['predicted_distance']  # (B, N, N)
        coords = self._enforce_distance_constraints(coords, pred_dist)
        
        # --- 2. 生成基因表达 ---
        base_expr = self.expression_generator(h_single)  # (B, N, 978)
        
        # 空间调制: 邻近细胞通过配对表示影响表达
        spatial_mod = self.spatial_expression_modulator(
            torch.cat([pair_agg, coords], dim=-1)
        )  # (B, N, 978)
        
        # 耦合: 基础表达 + 空间环境调制
        expression = base_expr + 0.3 * spatial_mod
        
        # 约束在合理范围 (z-score)
        expression = torch.tanh(expression) * 5  # 限制在[-5, 5]
        
        # --- 3. 组装Z_0 ---
        # Z_0可以表示为 (B, N, G+3) 或分离的tensor
        Z_0 = {
            'coordinates': coords,           # (B, N, 3) 空间位置
            'expression': expression,          # (B, N, 978) 基因表达
            'cell_features': h_single,         # (B, N, d) 细胞特征
            'pair_features': h_pair            # (B, N, N, d_pair) 互作特征
        }
        
        return Z_0
    
    def _enforce_distance_constraints(self, coords, target_dist):
        """
        力导向布局: 使实际距离逼近预测距离
        迭代优化坐标
        """
        # 简化: 单次梯度更新
        actual_dist = torch.cdist(coords, coords)  # (B, N, N)
        diff = actual_dist - target_dist
        
        # 力的方向
        # 这里简化实现，实际需要更复杂的物理模拟
        return coords  # 简化返回
class CellAtlasDiffusion(nn.Module):
    """
    条件扩散模型: 从Z_T噪声生成精细细胞图谱
    
    条件: PairFormer输出的 (h_single, h_pair, Z_0)
    去噪目标: Z_0* (优化的空间基因表达)
    """
    def __init__(self,
                 n_genes: int = 978,
                 spatial_dim: int = 3,
                 d_model: int = 256,
                 n_timesteps: int = 1000,
                 beta_schedule: str = 'cosine'):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_timesteps = n_timesteps
        
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(n_timesteps, d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # 去噪网络: U-Net + Cross-Attention条件
        self.denoiser = CellDenoiser(
            in_channels=n_genes + spatial_dim,  # 表达 + 坐标
            cond_channels=d_model * 2,         # h_single + h_pair聚合
            base_channels=128,
            n_resolutions=4
        )
        
        # 噪声调度
        self.register_buffer('betas', self._get_beta_schedule(beta_schedule))
        alphas = 1.0 - self.betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                            torch.sqrt(1.0 - self.alphas_cumprod))
        
    def _get_beta_schedule(self, schedule):
        if schedule == 'cosine':
            # cosine schedule (Improved DDPM)
            steps = self.n_timesteps + 1
            x = torch.linspace(0, self.n_timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.n_timesteps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            # linear
            return torch.linspace(0.0001, 0.02, self.n_timesteps)
    
    def q_sample(self, z_0, t, noise=None):
        """
        前向扩散: 加噪过程
        z_t = sqrt(α_t) * z_0 + sqrt(1-α_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(z_0)
        
        sqrt_acp = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_1m_acp = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_acp * z_0 + sqrt_1m_acp * noise, noise
    
    def forward(self, z_0, h_single, h_pair, t):
        """
        训练: 预测噪声
        """
        # 加噪
        z_t, noise = self.q_sample(z_0, t)
        
        # 条件聚合
        cond_single = h_single.mean(dim=1)  # (B, d) 全局细胞特征
        cond_pair = h_pair.mean(dim=(1,2))  # (B, d_pair) 全局互作特征
        cond = torch.cat([cond_single, cond_pair], dim=-1)  # (B, d*2)
        
        # 时间嵌入
        t_emb = self.time_embed(t)
        
        # 预测噪声
        pred_noise = self.denoiser(z_t, cond, t_emb)
        
        return pred_noise, noise
    
    @torch.no_grad()
    def sample(self, z_0_init, h_single, h_pair, n_steps=50):
        """
        采样生成 (DDPM/DDIM)
        
        z_0_init: 从Z0Generator获得的初始化
        """
        B = z_0_init.size(0)
        device = z_0_init.device
        
        # 从噪声开始，但用Z_0作为均值先验
        # 实际上可以从Z_0加噪开始，而非纯噪声
        z = z_0_init + torch.randn_like(z_0_init) * 0.5  # 带先验的初始化
        
        # DDIM采样
        for i in reversed(range(n_steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            
            # 预测噪声
            pred_noise, _ = self.forward(z, h_single, h_pair, t)
            
            # DDIM更新 (确定性)
            alpha_t = self.alphas_cumprod[t].view(-1, 1, 1)
            alpha_t_prev = self.alphas_cumprod[t-1].view(-1, 1, 1) if i > 0 else torch.ones_like(alpha_t)
            
            pred_z_0 = (z - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            z = torch.sqrt(alpha_t_prev) * pred_z_0 + torch.sqrt(1 - alpha_t_prev) * pred_noise
            
        return z


class CellDenoiser(nn.Module):
    """
    去噪U-Net + 交叉注意力条件
    """
    def __init__(self, in_channels, cond_channels, base_channels=128, n_resolutions=4):
        super().__init__()
        
        # 编码器
        self.encoder = nn.ModuleList()
        ch = in_channels
        for i in range(n_resolutions):
            self.encoder.append(nn.Sequential(
                nn.Conv1d(ch, base_channels * (2**i), 3, padding=1),
                nn.GroupNorm(8, base_channels * (2**i)),
                nn.SiLU(),
                nn.Conv1d(base_channels * (2**i), base_channels * (2**i), 3, padding=1),
                nn.GroupNorm(8, base_channels * (2**i)),
                nn.SiLU(),
            ))
            ch = base_channels * (2**i)
        
        # 交叉注意力条件
        self.cross_attn = nn.MultiheadAttention(ch, 8, batch_first=True)
        self.cond_proj = nn.Linear(cond_channels, ch)
        
        # 解码器
        self.decoder = nn.ModuleList()
        for i in reversed(range(n_resolutions)):
            self.decoder.append(nn.Sequential(
                nn.Conv1d(ch * 2, base_channels * (2**i), 3, padding=1),
                nn.GroupNorm(8, base_channels * (2**i)),
                nn.SiLU(),
            ))
            ch = base_channels * (2**i)
        
        self.out = nn.Conv1d(ch, in_channels, 1)
        
    def forward(self, z_t, cond, t_emb):
        # z_t: (B, C, L)  L = N * (G+3)
        # cond: (B, cond_dim)
        # t_emb: (B, d_model)
        
        cond_combined = cond + t_emb  # 简单相加，实际可更复杂
        
        # 编码
        skips = []
        h = z_t
        for block in self.encoder:
            h = block(h)
            skips.append(h)
            h = nn.functional.avg_pool1d(h, 2)
        
        # 交叉注意力
        h = h.permute(0, 2, 1)  # (B, L, C)
        cond_proj = self.cond_proj(cond_combined).unsqueeze(1)  # (B, 1, C)
        h, _ = self.cross_attn(h, cond_proj, cond_proj)
        h = h.permute(0, 2, 1)  # (B, C, L)
        
        # 解码
        for block in self.decoder:
            h = nn.functional.interpolate(h, scale_factor=2, mode='linear')
            h = torch.cat([h, skips.pop()], dim=1)
            h = block(h)
        
        return self.out(h)