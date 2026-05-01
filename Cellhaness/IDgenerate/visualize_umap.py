import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not installed, will use PCA instead")

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_cell_group(group_idx: int, data_dir: str = '/tmp/lrldm_data') -> dict:
    """加载细胞组数据"""
    file_path = os.path.join(data_dir, f"{group_idx}_0.pt")
    if os.path.exists(file_path):
        return torch.load(file_path, map_location='cpu')
    else:
        raise FileNotFoundError(f"Data file {file_path} not found")

def extract_cell_features(data: dict) -> tuple:
    """提取细胞特征"""
    # 提取基因表达和坐标（使用detach避免梯度问题）
    expression = data['z0']['expression'][0].detach().numpy()  # (N, 978)
    coordinates = data['z0']['coordinates'][0].detach().numpy()  # (N, 3)

    # 获取细胞类型信息（如果有）
    gsva = data['gsva']  # (N, 50)

    # 生成细胞类型标签（基于GSVA聚类）
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cell_type_labels = kmeans.fit_predict(gsva.detach().numpy())

    cell_type_names = ['Microglia', 'Astrocyte', 'B_cell', 'T_cell']

    return expression, coordinates, gsva.detach().numpy(), cell_type_labels, cell_type_names

def perform_umap(expression: np.ndarray, n_components: int = 2, random_state: int = 42):
    """执行UMAP降维"""
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        embedding = reducer.fit_transform(expression)
    else:
        pca = PCA(n_components=n_components, random_state=random_state)
        embedding = pca.fit_transform(expression)
    return embedding

def perform_pca(expression: np.ndarray, n_components: int = 50):
    """先PCA降维再UMAP"""
    pca = PCA(n_components=n_components, random_state=42)
    expression_pca = pca.fit_transform(expression)
    return perform_umap(expression_pca)

def visualize_cells(expression: np.ndarray, coordinates: np.ndarray,
                    cell_labels: np.ndarray, cell_type_names: list,
                    embedding_2d: np.ndarray, save_path: str = None):
    """可视化细胞"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. UMAP可视化（按细胞类型着色）
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, cell_type in enumerate(cell_type_names):
        mask = cell_labels == i
        axes[0].scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                       c=colors[i], label=cell_type, alpha=0.6, s=10)
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    axes[0].set_title('UMAP of Cell Atlas (by cell type)')
    axes[0].legend()

    # 2. 3D空间坐标可视化
    colors_3d = [colors[label] for label in cell_labels]
    scatter = axes[1].scatter(coordinates[:, 0], coordinates[:, 1],
                             c=cell_labels, cmap='tab10', alpha=0.6, s=10)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title('Cell Coordinates (XY plane)')
    plt.colorbar(scatter, ax=axes[1], label='Cell Type')

    # 3. 基因表达热图（按细胞类型聚合）
    n_cells_per_type = len(cell_labels) // 4
    type_expr_means = []
    for i in range(4):
        mask = cell_labels == i
        type_expr_means.append(expression[mask].mean(axis=0))
    type_expr_means = np.array(type_expr_means)

    im = axes[2].imshow(type_expr_means[:, :50].T, aspect='auto', cmap='viridis')
    axes[2].set_xlabel('Cell Type')
    axes[2].set_ylabel('Genes (first 50)')
    axes[2].set_title('Gene Expression Heatmap')
    axes[2].set_xticks(range(4))
    axes[2].set_xticklabels(cell_type_names, rotation=45)
    plt.colorbar(im, ax=axes[2], label='Expression')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

def main():
    """主函数"""
    print("Loading cell group data...")

    # 加载第0组数据
    data = load_cell_group(0)

    print(f"Data keys: {data.keys()}")
    print(f"GSVA shape: {data['gsva'].shape}")
    print(f"Expression shape: {data['z0']['expression'].shape}")
    print(f"Coordinates shape: {data['z0']['coordinates'].shape}")
    print(f"LR interaction matrix shape: {data['lr_data']['interaction_matrix'].shape}")

    # 提取特征
    expression, coordinates, gsva, cell_labels, cell_type_names = extract_cell_features(data)
    print(f"\nExtracted features:")
    print(f"  Expression: {expression.shape}")
    print(f"  Coordinates: {coordinates.shape}")
    print(f"  GSVA: {gsva.shape}")
    print(f"  Cell labels: {np.bincount(cell_labels)}")

    # 执行UMAP
    print("\nPerforming UMAP...")
    if HAS_UMAP:
        print("Using UMAP for dimensionality reduction")
        embedding_2d = perform_pca(expression, n_components=50)
    else:
        print("UMAP not available, using PCA only")
        pca = PCA(n_components=2, random_state=42)
        embedding_2d = pca.fit_transform(expression)

    print(f"Embedding shape: {embedding_2d.shape}")

    # 可视化
    print("\nGenerating visualization...")
    save_path = '/home/wupf_260213/controlnet/Cellhaness/IDgenerate/cell_umap_visualization.png'
    visualize_cells(expression, coordinates, cell_labels, cell_type_names,
                   embedding_2d, save_path=save_path)

    # 额外：可视化受配体相互作用网络
    visualize_lr_network(data['lr_data']['interaction_matrix'].detach().numpy(),
                       cell_labels, cell_type_names)

def visualize_lr_network(lr_matrix: np.ndarray, cell_labels: np.ndarray,
                         cell_type_names: list, save_path: str = None):
    """可视化受配体相互作用网络"""
    import networkx as nx

    # 简化网络：只保留强相互作用
    threshold = np.percentile(lr_matrix, 95)
    adj_matrix = (lr_matrix > threshold).astype(float)

    # 创建图
    G = nx.Graph()

    # 添加节点
    n_cells = len(cell_labels)
    for i in range(n_cells):
        G.add_node(i, cell_type=cell_type_names[cell_labels[i]])

    # 添加边
    for i in range(n_cells):
        for j in range(i+1, n_cells):
            if adj_matrix[i, j] > 0:
                G.add_edge(i, j, weight=lr_matrix[i, j])

    # 绘制网络图
    fig, ax = plt.subplots(figsize=(12, 10))

    # 按细胞类型着色
    node_colors = [cell_labels[node] for node in G.nodes()]

    # 使用spring布局
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # 绘制
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.6, s=5, cmap='tab10')
    nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.3)

    # 添加图例
    for i, cell_type in enumerate(cell_type_names):
        ax.scatter([], [], c=[plt.cm.tab10(i)], label=cell_type, s=50)
    ax.legend(title='Cell Type', loc='upper right')

    ax.set_title('Ligand-Receptor Interaction Network')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Network figure saved to {save_path}")

    plt.show()

if __name__ == '__main__':
    main()
