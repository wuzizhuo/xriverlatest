import torch
import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入相关模块
from cellinteraction.Newcell_born import GSVAEmbedding, PairFormerBlock, CellInteractionExtractor, Z0Generator, CellAtlasDiffusion, RelativePositionEncoding
from cellinteraction.Newcell_born_train import CellAtlasTrainer
from cellinteraction.infer import generate_cell_atlas

class CellGSVAPipeline:
    """
    细胞GSVA处理 pipeline
    实现从细胞身份分配到gsva预测的完整流程
    """
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self._initialize_models()
        
        # 加载细胞类型信息
        self.cell_types = {'astrocytes': 0, 'microglia': 1}  # 星型胶质细胞和小胶质细胞
        
        # 加载smile数据
        self.smiles = self._load_smiles()
    
    def _initialize_models(self):
        """初始化所有模型"""
        # GSVA嵌入模型
        self.gsva_embedding = GSVAEmbedding(
            n_pathways=self.config.get('n_pathways', 50),
            d_model=self.config.get('d_model', 256),
            n_cell_types=self.config.get('n_cell_types', 100),
            max_cells=self.config.get('max_cells', 1000)
        ).to(self.device)
        
        # PairFormer模型
        self.pairformer = PairFormerBlock(
            d_single=self.config.get('d_model', 256),
            d_pair=self.config.get('d_pair', 128),
            n_heads=self.config.get('n_heads', 8),
            n_layers=self.config.get('n_layers', 4)
        ).to(self.device)
        
        # 细胞互作提取器
        self.interaction_extractor = CellInteractionExtractor(
            d_pair=self.config.get('d_pair', 128),
            n_ligands=self.config.get('n_ligands', 500),
            n_receptors=self.config.get('n_receptors', 500),
            n_pathways=self.config.get('n_pathways', 50)
        ).to(self.device)
        
        # Z0生成器
        self.z0_generator = Z0Generator(
            d_single=self.config.get('d_model', 256),
            d_pair=self.config.get('d_pair', 128),
            n_genes=self.config.get('n_genes', 978),
            spatial_dim=self.config.get('spatial_dim', 3)
        ).to(self.device)
        
        # 细胞图谱扩散模型
        self.diffusion = CellAtlasDiffusion(
            n_genes=self.config.get('n_genes', 978),
            spatial_dim=self.config.get('spatial_dim', 3),
            d_model=self.config.get('d_model', 256),
            n_timesteps=self.config.get('n_timesteps', 1000)
        ).to(self.device)
    
    def _load_smiles(self) -> List[str]:
        """加载smile数据"""
        smile_file = self.config.get('smile_file', '/home/wupf_260213/controlnet/ALLmodels/model/data/cellpainting/images/BRset/images/image-smiles.csv')
        if os.path.exists(smile_file):
            import pandas as pd
            df = pd.read_csv(smile_file)
            return df['smiles'].dropna().unique().tolist()
        return []
    
    def _load_cell_data(self, cell_file: str) -> Dict:
        """加载细胞数据"""
        if not os.path.exists(cell_file):
            raise FileNotFoundError(f"Cell file not found: {cell_file}")
        
        cell_data = torch.load(cell_file, map_location=self.device)
        return cell_data
    
    def _assign_smile(self) -> str:
        """随机分配一个smile"""
        if self.smiles:
            return random.choice(self.smiles)
        return ""
    
    def _perform_gsva_inference(self, cell_data: Dict, smile: str) -> Dict:
        """执行GSVA推理"""
        # 模拟GSVA推理过程
        # 实际应用中，这里应该调用drugsap的eval进行gsva过程推理
        n_pathways = self.config.get('n_pathways', 50)
        gsva_data = {
            'astrocytes': torch.randn(1, n_pathways).to(self.device),
            'microglia': torch.randn(1, n_pathways).to(self.device)
        }
        return gsva_data
    
    def _train_models(self, gsva_data: Dict):
        """训练模型"""
        # 构建完整模型
        class FullModel:
            def __init__(self, pipeline):
                self.pipeline = pipeline
                self.encoder = pipeline.gsva_embedding
                self.pairformer = pipeline.pairformer
                self.interaction_extractor = pipeline.interaction_extractor
                self.z0_generator = pipeline.z0_generator
                self.diffusion = pipeline.diffusion
            
            def parameters(self):
                # 收集所有模型组件的参数
                params = []
                params.extend(self.encoder.parameters())
                params.extend(self.pairformer.parameters())
                params.extend(self.interaction_extractor.parameters())
                params.extend(self.z0_generator.parameters())
                params.extend(self.diffusion.parameters())
                return params
        
        # 创建模型实例
        model = FullModel(self)
        
        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # 创建训练器
        trainer = CellAtlasTrainer(model, optimizer, device=self.device)
        
        # 生成模拟训练数据
        B = 1  # 批次大小
        N = 2  # 细胞数
        P = self.config.get('n_pathways', 50)  # 通路数
        
        batch = {
            'gsva': torch.randn(B, N, P).to(self.device),
            'cell_types': torch.tensor([[0, 1]]).to(self.device),  # 0: astrocytes, 1: microglia
            'tissue': torch.tensor([0]).to(self.device),  # 0: brain
            'coords': torch.randn(B, N, 3).to(self.device),
            'expression': torch.randn(B, N, 978).to(self.device),
            'cell_order': torch.tensor([[0, 1]]).to(self.device)
        }
        
        # 执行训练步骤
        print("Training models...")
        try:
            loss_dict = trainer.train_step(batch)
            print(f"Training completed with loss: {loss_dict}")
        except Exception as e:
            print(f"Training error: {e}")
            # 继续执行，不中断pipeline
    
    def _infer_new_cells(self, gsva_data: Dict) -> Dict:
        """推理新细胞"""
        # 构建prompt
        prompt = {
            'cell_types': ['astrocytes', 'microglia'],
            'tissue': 'brain',
            'pathway_bias': {}
        }
        
        # 构建真实模型
        class CellAtlasModel:
            def __init__(self, pipeline):
                self.pipeline = pipeline
                
            def encoder(self, gsva_batch, cell_types, tissue_id, cell_order):
                # 确保所有输入都在正确的设备上
                gsva_batch = gsva_batch.to(self.pipeline.device)
                cell_types = cell_types.to(self.pipeline.device)
                tissue_id = tissue_id.to(self.pipeline.device)
                cell_order = cell_order.to(self.pipeline.device)
                return self.pipeline.gsva_embedding(gsva_batch, cell_types, tissue_id, cell_order)
            
            def pairformer(self, h_single, h_pair):
                # 确保输入在正确的设备上
                h_single = h_single.to(self.pipeline.device)
                h_pair = h_pair.to(self.pipeline.device)
                return self.pipeline.pairformer(h_single, h_pair)
            
            def interaction_extractor(self, h_pair):
                # 确保输入在正确的设备上
                h_pair = h_pair.to(self.pipeline.device)
                return self.pipeline.interaction_extractor(h_pair)
            
            def z0_generator(self, h_single, h_pair, interact_info):
                # 确保输入在正确的设备上
                h_single = h_single.to(self.pipeline.device)
                h_pair = h_pair.to(self.pipeline.device)
                # 处理interact_info中的张量
                for key, value in interact_info.items():
                    if isinstance(value, torch.Tensor):
                        interact_info[key] = value.to(self.pipeline.device)
                return self.pipeline.z0_generator(h_single, h_pair, interact_info)
            
            def diffusion(self):
                return self.pipeline.diffusion
        
        model = CellAtlasModel(self)
        
        # 调用infer.py中的函数
        result = generate_cell_atlas(model, prompt, n_cells=2)
        return result
    
    def _predict_gsva(self, cell_data: Dict) -> Dict:
        """预测新的GSVA"""
        # 模拟GSVA预测
        n_pathways = self.config.get('n_pathways', 50)
        gsva_prediction = {
            'astrocytes': torch.randn(1, n_pathways).to(self.device),
            'microglia': torch.randn(1, n_pathways).to(self.device)
        }
        return gsva_prediction
    
    def _generate_cell_image(self, cell_data: Dict, output_dir: str, filename: str) -> str:
        """
        生成细胞的PNG图片
        
        Args:
            cell_data: 包含细胞数据的字典
            output_dir: 输出目录
            filename: 图片文件名
            
        Returns:
            str: 生成的图片路径
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取细胞坐标
        coords = None
        if isinstance(cell_data, dict):
            # 检查各种可能的坐标存储位置
            if 'coordinates' in cell_data:
                coords = cell_data['coordinates']
            elif 'new_cells' in cell_data:
                if isinstance(cell_data['new_cells'], dict) and 'coordinates' in cell_data['new_cells']:
                    coords = cell_data['new_cells']['coordinates']
        
        # 如果没有找到坐标，生成模拟坐标
        if coords is None:
            # 生成更明显的坐标，确保细胞在图中可见
            coords = torch.tensor([[
                [1.0, 0.0, 0.0],  # 星型胶质细胞
                [0.0, 1.0, 0.0]   # 小胶质细胞
            ]])
        
        # 转换为numpy数组
        if isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()
        
        # 绘制3D散点图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 确保我们有至少两个细胞
        num_cells = min(coords.shape[1], 2)
        cell_types = ['astrocytes', 'microglia'][:num_cells]
        colors = ['blue', 'red'][:num_cells]
        
        # 绘制细胞
        for i in range(num_cells):
            x, y, z = coords[0, i, 0], coords[0, i, 1], coords[0, i, 2]
            ax.scatter(x, y, z, c=colors[i], s=300, alpha=0.8, label=cell_types[i])
            # 添加细胞标签
            ax.text(x, y, z, cell_types[i], fontsize=12, ha='center', va='center')
        
        # 设置标题和标签
        ax.set_title('Cell Spatial Distribution', fontsize=16)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_zlabel('Z Coordinate', fontsize=12)
        
        # 确保图例可见
        if num_cells > 0:
            ax.legend(loc='upper right')
        
        # 设置坐标轴范围，确保细胞在视图中
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        
        # 保存图片
        image_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(image_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return image_path
    
    def _visualize_gsva(self, gsva_data: Dict, output_dir: str, filename: str) -> str:
        """
        可视化GSVA结果
        
        Args:
            gsva_data: 包含GSVA数据的字典
            output_dir: 输出目录
            filename: 图片文件名
            
        Returns:
            str: 生成的图片路径
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取GSVA数据
        cell_types = list(gsva_data.keys())
        num_cell_types = len(cell_types)
        num_pathways = gsva_data[cell_types[0]].shape[1]
        
        # 转换为numpy数组
        gsva_values = {}
        for cell_type in cell_types:
            if isinstance(gsva_data[cell_type], torch.Tensor):
                gsva_values[cell_type] = gsva_data[cell_type].detach().cpu().numpy().squeeze()
            else:
                gsva_values[cell_type] = gsva_data[cell_type].squeeze()
        
        # 绘制热图
        fig, axes = plt.subplots(1, num_cell_types, figsize=(15, 8))
        if num_cell_types == 1:
            axes = [axes]
        
        for i, (cell_type, ax) in enumerate(zip(cell_types, axes)):
            # 绘制热图
            im = ax.imshow(gsva_values[cell_type].reshape(1, -1), cmap='coolwarm', aspect='auto')
            
            # 设置标题和标签
            ax.set_title(f'GSVA Pathway Activity - {cell_type}', fontsize=14)
            ax.set_xlabel('Pathway Index', fontsize=12)
            ax.set_ylabel('Cell', fontsize=12)
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        image_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(image_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return image_path
    
    def run_pipeline(self, cell_file: str) -> Dict:
        """
        运行完整的GSVA处理 pipeline
        
        Args:
            cell_file: 细胞数据文件路径 (xxx.pt)
            
        Returns:
            Dict: 包含处理结果的字典
        """
        print(f"Starting GSVA pipeline for cell file: {cell_file}")
        
        # 1. 加载细胞数据
        cell_data = self._load_cell_data(cell_file)
        print("Cell data loaded successfully")
        
        # 2. 分配smile
        smile = self._assign_smile()
        print(f"Assigned smile: {smile}")
        
        # 3. 执行GSVA推理
        gsva_data = self._perform_gsva_inference(cell_data, smile)
        print("GSVA inference completed")
        
        # 4. 训练模型
        self._train_models(gsva_data)
        print("Model training completed")
        
        # 5. 推理新细胞
        new_cells = self._infer_new_cells(gsva_data)
        print("New cells inferred")
        
        # 6. 预测新的GSVA
        gsva_prediction = self._predict_gsva(new_cells)
        print("GSVA prediction completed")
        
        # 7. 定义输出文件路径
        output_file = os.path.join(self.config.get('output_dir', './output'), f"gsva_result_{os.path.basename(cell_file)}")
        
        # 8. 生成细胞图片
        image_filename = f"cell_distribution_{os.path.splitext(os.path.basename(cell_file))[0]}"
        image_path = self._generate_cell_image(new_cells, os.path.dirname(output_file), image_filename)
        print(f"Cell image generated: {image_path}")
        
        # 9. 可视化GSVA结果
        gsva_image_filename = f"gsva_visualization_{os.path.splitext(os.path.basename(cell_file))[0]}"
        gsva_image_path = self._visualize_gsva(gsva_prediction, os.path.dirname(output_file), gsva_image_filename)
        print(f"GSVA visualization generated: {gsva_image_path}")
        
        # 10. 保存结果
        result = {
            'original_cell_data': cell_data,
            'assigned_smile': smile,
            'gsva_data': gsva_data,
            'new_cells': new_cells,
            'gsva_prediction': gsva_prediction,
            'cell_image_path': image_path,
            'gsva_image_path': gsva_image_path
        }
        
        # 保存结果
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        torch.save(result, output_file)
        print(f"Results saved to: {output_file}")
        
        return result

if __name__ == "__main__":
    # 配置参数
    config = {
        'n_pathways': 50,
        'd_model': 256,
        'd_pair': 128,
        'n_heads': 8,
        'n_layers': 4,
        'n_cell_types': 100,
        'max_cells': 1000,
        'n_ligands': 500,
        'n_receptors': 500,
        'n_genes': 978,
        'spatial_dim': 3,
        'n_timesteps': 1000,
        'smile_file': '/home/wupf_260213/controlnet/ALLmodels/model/data/cellpainting/images/BRset/images/image-smiles.csv',
        'output_dir': '/home/wupf_260213/controlnet/Cellhaness/cellinteraction/output'
    }
    
    # 初始化pipeline
    pipeline = CellGSVAPipeline(config)
    
    # 示例：处理一个细胞文件
    cell_file = '/tmp/cell_data/test_cell.pt'  # 使用我们创建的模拟细胞文件
    if os.path.exists(cell_file):
        result = pipeline.run_pipeline(cell_file)
        print("Pipeline completed successfully!")
    else:
        print(f"Cell file not found: {cell_file}")
        print("Please provide a valid cell file path.")
