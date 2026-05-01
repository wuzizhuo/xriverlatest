import torch

def get_cell_type_gsva(cell_type):
    """
    获取细胞类型的基线GSVA值
    
    Args:
        cell_type: 细胞类型名称
        
    Returns:
        torch.Tensor: 细胞类型的基线GSVA值
    """
    # 模拟不同细胞类型的基线GSVA
    gsva_dict = {
        'astrocytes': torch.randn(50),  # 星型胶质细胞
        'microglia': torch.randn(50),   # 小胶质细胞
        'hepatocyte': torch.randn(50),  # 肝细胞
        'kupffer': torch.randn(50),     # 枯否细胞
        'stellate': torch.randn(50)     # 星状细胞
    }
    
    return gsva_dict.get(cell_type, torch.randn(50))

@torch.no_grad()
def generate_cell_atlas(model, prompt, n_cells=100):
    """
    从Prompt生成完整细胞图谱
    
    prompt: {
        'cell_types': ['hepatocyte', 'kupffer', 'stellate', ...],  # 细胞类型列表
        'tissue': 'liver',
        'pathway_bias': {'Wnt': 2.0, 'TGF-β': -1.5},  # 通路偏置
    }
    """
    # 1. 构建GSVA序列
    gsva_sequence = []
    for cell_type in prompt['cell_types']:
        # 从细胞类型查询基线GSVA + 通路偏置
        base_gsva = get_cell_type_gsva(cell_type)  # 查表
        for pw, val in prompt['pathway_bias'].items():
            base_gsva[pw] += val
        gsva_sequence.append(base_gsva)
    
    gsva_batch = torch.stack(gsva_sequence).unsqueeze(0)  # (1, N, P)
    N = len(prompt['cell_types'])
    
    # 准备其他必要参数
    cell_types = torch.tensor([0 if ct == 'astrocytes' else 1 for ct in prompt['cell_types']]).unsqueeze(0)
    tissue_id = torch.tensor([0])  # 0: brain
    cell_order = torch.arange(N).unsqueeze(0)
    
    # 2. 编码 → PairFormer
    h_single = model.encoder(gsva_batch, cell_types, tissue_id, cell_order)
    
    # 初始化h_pair
    # 由于我们没有坐标信息，使用随机初始化
    d_pair = 128
    h_pair = torch.randn(1, N, N, d_pair)
    
    h_single, h_pair = model.pairformer(h_single, h_pair)
    
    # 3. 提取互作
    interact_info = model.interaction_extractor(h_pair)
    
    # 4. 生成Z_0 (粗粒度图谱)
    Z_0 = model.z0_generator(h_single, h_pair, interact_info)
    
    # 5. 扩散精细化
    z_final = model.diffusion().sample(Z_0, h_single, h_pair, n_steps=50)
    
    # 解析输出
    expression = z_final[:, :978]  # (N, 978)
    coords = z_final[:, 978:981]   # (N, 3)
    
    return {
        'coordinates': coords,
        'expression': expression,
        'interaction_matrix': interact_info['attention_weights'],
        'pathway_crosstalk': interact_info['pathway_crosstalk'],
        'predicted_distances': interact_info['predicted_distance']
    }