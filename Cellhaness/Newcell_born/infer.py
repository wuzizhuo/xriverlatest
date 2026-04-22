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
    
    # 2. 编码 → PairFormer
    h_single = model.encoder(gsva_batch, ...)
    h_single, h_pair = model.pairformer(h_single, ...)
    
    # 3. 提取互作
    interact_info = model.interaction_extractor(h_pair)
    
    # 4. 生成Z_0 (粗粒度图谱)
    Z_0 = model.z0_generator(h_single, h_pair, interact_info)
    
    # 5. 扩散精细化
    z_final = model.diffusion.sample(Z_0, h_single, h_pair, n_steps=50)
    
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