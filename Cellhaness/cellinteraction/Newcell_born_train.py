from cellinteraction.Newcell_born import RelativePositionEncoding

class CellAtlasTrainer:
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
    def train_step(self, batch):
        """
        batch: {
            'gsva': (B, N, P),           # 细胞GSVA序列
            'cell_types': (B, N),         # 细胞类型
            'tissue': (B,),               # 组织类型
            'coords': (B, N, 3),          # 真实空间坐标 (监督信号)
            'expression': (B, N, 978),    # 真实基因表达 (监督信号)
            'cell_order': (B, N),         # 序列位置
        }
        """
        # 1. 编码
        h_single = self.model.encoder(
            batch['gsva'], 
            batch['cell_types'],
            batch['tissue'],
            batch['cell_order']
        )
        
        # 2. PairFormer
        # 构建相对位置编码 (使用真实坐标或预测坐标)
        rel_pos = RelativePositionEncoding()(batch['coords'])
        h_pair_init = rel_pos  # 初始化pair表示
        
        h_single, h_pair = self.model.pairformer(h_single, h_pair_init)
        
        # 3. 互作提取
        interact_info = self.model.interaction_extractor(h_pair)
        
        # 4. 生成Z_0
        Z_0 = self.model.z0_generator(h_single, h_pair, interact_info)
        
        # 5. 扩散训练
        # 组装真实Z_0
        z_0_true = torch.cat([batch['expression'], batch['coords']], dim=-1)
        
        # 获取批次大小
        B = batch['gsva'].shape[0]
        
        # 随机时间步
        t = torch.randint(0, self.model.diffusion.n_timesteps, (B,), device=self.device)
        
        # 预测噪声
        pred_noise, true_noise = self.model.diffusion(z_0_true, h_single, h_pair, t)
        
        # 损失
        loss_diffusion = nn.functional.mse_loss(pred_noise, true_noise)
        
        # 辅助损失
        loss_coord = nn.functional.mse_loss(Z_0['coordinates'], batch['coords'])
        loss_expr = nn.functional.mse_loss(Z_0['expression'], batch['expression'])
        
        # 互作损失 (如果有真实互作标注)
        # loss_interact = ...
        
        loss = loss_diffusion + 0.1 * loss_coord + 0.1 * loss_expr
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'loss_diffusion': loss_diffusion.item(),
            'loss_coord': loss_coord.item(),
            'loss_expr': loss_expr.item()
        }