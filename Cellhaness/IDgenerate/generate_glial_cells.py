#!/usr/bin/env python3
"""
生成小胶质细胞和星型胶质细胞的完整流水线
流程：Prompt → GSVA → L1000 → MorphDiff
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from typing import List, Dict, Tuple

# 导入 idgenerated.py 中的类
sys.path.append(os.path.dirname(__file__))
from idgenerated import GSVAGenerator, GSVA2L1000Solver, Prompt2L1000Pipeline


def load_gmt(path: str) -> List[str]:
    """从 gmt 文件加载通路名称"""
    pathways = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 0:
                pathways.append(parts[0])
    return pathways


def load_image_gene_data() -> Tuple[np.ndarray, List[str]]:
    """从 image-gene.csv 加载真实的基因表达数据"""
    # 读取 image-gene.csv
    image_gene_path = "/home/wupf_260213/controlnet/ALLmodels/model/data/cellpainting/image-gene.csv"
    df = pd.read_csv(image_gene_path)
    
    # 选择第一个基因表达文件
    gene_file_path = df['gene_file_path'].iloc[0]
    # 替换路径中的 /ppotools/ 为 /ALLmodels/
    gene_file_path = gene_file_path.replace('/ppotools/', '/ALLmodels/')
    
    print(f"Loading gene expression from: {gene_file_path}")
    
    # 读取基因表达文件
    gene_df = pd.read_csv(gene_file_path)
    
    # 提取基因表达值
    genes = gene_df['gene'].tolist() if 'gene' in gene_df.columns else gene_df.iloc[:, 0].tolist()
    expression_values = gene_df['expression'].values if 'expression' in gene_df.columns else gene_df.iloc[:, 1].values
    
    # 如果表达值不足 978 个，补足到 978 个
    if len(expression_values) < 978:
        padding = np.zeros(978 - len(expression_values))
        expression_values = np.concatenate([expression_values, padding])
    else:
        expression_values = expression_values[:978]
    
    # 归一化到 z-score
    mean = np.mean(expression_values)
    std = np.std(expression_values)
    expression_values = (expression_values - mean) / (std + 1e-8)
    
    # 生成多个样本（使用不同的噪声）
    n_samples = 1000
    l1000_data = np.zeros((n_samples, 978))
    
    for i in range(n_samples):
        # 添加小噪声
        noise = np.random.normal(0, 0.1, 978)
        l1000_data[i] = expression_values + noise
    
    return l1000_data, genes


def mock_gsva_calculator(x: np.ndarray) -> np.ndarray:
    """模拟 GSVA 计算函数"""
    # 真实使用时需要使用真实的 GSVA 计算
    n_pathways = 50
    return np.random.randn(n_pathways)


def generate_glial_cells():
    """生成小胶质细胞和星型胶质细胞"""
    # 1. 加载 gmt 通路文件
    gmt_path = "/home/wupf_260213/controlnet/Cellhaness/IDgenerate/Mordiffreal/MorphDiff/gsvagmt/h.all.v2026.1.Hs.symbols.gmt"
    pathway_names = load_gmt(gmt_path)[:50]  # 取前50个通路
    print(f"Loaded {len(pathway_names)} pathways from GMT file")
    
    # 2. 初始化 GSVA 生成器
    gsva_gen = GSVAGenerator(
        pathway_names=pathway_names,
        baseline_gsva=np.zeros(len(pathway_names))
    )
    
    # 3. 加载真实的基因表达数据
    l1000_db, gene_names = load_image_gene_data()
    if len(gene_names) < 978:
        gene_names.extend([f"gene_{i}" for i in range(len(gene_names), 978)])
    else:
        gene_names = gene_names[:978]
    
    # 4. 初始化 L1000 求解器
    solver = GSVA2L1000Solver(
        l1000_database=l1000_db,
        gsva_calculator=mock_gsva_calculator,
        gene_names=gene_names,
        pathway_names=pathway_names
    )
    
    # 5. 构建完整流水线
    pipeline = Prompt2L1000Pipeline(gsva_gen, solver)
    
    # 6. 生成小胶质细胞
    print("\n=== 生成小胶质细胞 ===")
    microglia_prompt = "生成小胶质细胞，免疫激活状态，高炎症反应"
    microglia_result = pipeline.generate(
        microglia_prompt,
        n_gsva_samples=3,
        temperature=1.0,
        return_intermediate=True
    )
    
    # 7. 生成星型胶质细胞
    print("\n=== 生成星型胶质细胞 ===")
    astrocyte_prompt = "生成星型胶质细胞，营养支持状态，低炎症反应"
    astrocyte_result = pipeline.generate(
        astrocyte_prompt,
        n_gsva_samples=3,
        temperature=1.0,
        return_intermediate=True
    )
    
    # 8. 保存结果
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存小胶质细胞结果
    with open(os.path.join(output_dir, "microglia_result.json"), 'w') as f:
        json.dump({
            'prompt': microglia_prompt,
            'l1000_expression': microglia_result['l1000_expression'].tolist(),
            'gsva_mean': microglia_result['gsva_target'].tolist(),
            'gsva_uncertainty': microglia_result['gsva_uncertainty'].tolist(),
            'pathway_names': pathway_names,
            'gene_names': gene_names
        }, f, indent=2)
    
    # 保存小胶质细胞 GSVA 结果
    with open(os.path.join(output_dir, "microglia_gsva.json"), 'w') as f:
        json.dump({
            'prompt': microglia_prompt,
            'gsva_mean': microglia_result['gsva_target'].tolist(),
            'gsva_uncertainty': microglia_result['gsva_uncertainty'].tolist(),
            'pathway_names': pathway_names
        }, f, indent=2)
    
    # 保存星型胶质细胞结果
    with open(os.path.join(output_dir, "astrocyte_result.json"), 'w') as f:
        json.dump({
            'prompt': astrocyte_prompt,
            'l1000_expression': astrocyte_result['l1000_expression'].tolist(),
            'gsva_mean': astrocyte_result['gsva_target'].tolist(),
            'gsva_uncertainty': astrocyte_result['gsva_uncertainty'].tolist(),
            'pathway_names': pathway_names,
            'gene_names': gene_names
        }, f, indent=2)
    
    # 保存星型胶质细胞 GSVA 结果
    with open(os.path.join(output_dir, "astrocyte_gsva.json"), 'w') as f:
        json.dump({
            'prompt': astrocyte_prompt,
            'gsva_mean': astrocyte_result['gsva_target'].tolist(),
            'gsva_uncertainty': astrocyte_result['gsva_uncertainty'].tolist(),
            'pathway_names': pathway_names
        }, f, indent=2)
    
    print("\n=== 生成完成 ===")
    print(f"小胶质细胞结果保存到: {os.path.join(output_dir, 'microglia_result.json')}")
    print(f"小胶质细胞 GSVA 结果保存到: {os.path.join(output_dir, 'microglia_gsva.json')}")
    print(f"星型胶质细胞结果保存到: {os.path.join(output_dir, 'astrocyte_result.json')}")
    print(f"星型胶质细胞 GSVA 结果保存到: {os.path.join(output_dir, 'astrocyte_gsva.json')}")
    
    return microglia_result, astrocyte_result


if __name__ == "__main__":
    generate_glial_cells()
