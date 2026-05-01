import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import json

class PhenotypeExtractor:
    """从自然语言Prompt中提取结构化表型特征"""
    
    # 预定义的表型本体库（可扩展）
    CELL_TYPES = {
        '肝细胞癌': 'hepatocellular_carcinoma',
        '肝癌': 'hepatocellular_carcinoma', 
        '肝肿瘤': 'hepatocellular_carcinoma',
        '正常肝细胞': 'normal_hepatocyte',
        '肝星状细胞': 'hepatic_stellate',
        '小胶质细胞': 'microglia',
        '星型胶质细胞': 'astrocyte',
    }
    
    PATHWAY_PHENOTYPES = {
        # 增殖相关
        '高增殖': {'Hallmark_E2F_Targets': 2.5, 'Hallmark_G2M_Checkpoint': 2.0},
        '低增殖': {'Hallmark_E2F_Targets': -1.5, 'Hallmark_G2M_Checkpoint': -1.0},
        
        # 侵袭转移
        '侵袭性强': {'Hallmark_Epithelial_Mesenchymal_Transition': 2.5, 
                   'Hallmark_Invasion_Metastasis': 2.0},
        '侵袭性弱': {'Hallmark_Epithelial_Mesenchymal_Transition': -1.5},
        
        # 代谢重编程
        '糖酵解': {'Hallmark_Glycolysis': 2.5, 'Hallmark_Oxidative_Phosphorylation': -1.5},
        '氧化磷酸化': {'Hallmark_Glycolysis': -1.0, 'Hallmark_Oxidative_Phosphorylation': 2.0},
        
        # 免疫微环境
        '免疫热': {'Hallmark_Interferon_Gamma_Response': 2.0, 
                 'Hallmark_Inflammatory_Response': 1.5},
        '免疫冷': {'Hallmark_Interferon_Gamma_Response': -1.5},
        
        # 突变相关
        'TP53突变': {'Hallmark_P53_Pathway': -2.5, 'Hallmark_Apoptosis': -1.5},
        'KRAS突变': {'Hallmark_KRAS_Signaling_Up': 2.5, 'Hallmark_KRAS_Signaling_Dn': -2.0},
        'MYC扩增': {'Hallmark_MYC_Targets_V1': 2.5, 'Hallmark_MYC_Targets_V2': 2.5},
        
        # 应激反应
        '缺氧': {'Hallmark_Hypoxia': 2.5, 'Hallmark_Angiogenesis': 1.5},
        '内质网应激': {'Hallmark_Unfolded_Protein_Response': 2.5},
    }
    
    def __init__(self, pathway_names: List[str]):
        self.pathway_names = pathway_names
        self.n_pathways = len(pathway_names)
        
    def parse_prompt(self, prompt: str) -> Dict:
        """解析Prompt，提取所有表型特征"""
        prompt_lower = prompt.lower()
        
        features = {
            'cell_type': None,
            'pathway_modulations': {},  # 通路名 -> 调制值
            'confidence': 1.0
        }
        
        # 1. 识别细胞类型
        for cn, en in self.CELL_TYPES.items():
            if cn in prompt:
                features['cell_type'] = en
                break
                
        # 2. 识别通路表型
        for phenotype, modulations in self.PATHWAY_PHENOTYPES.items():
            if phenotype in prompt:
                for pw, val in modulations.items():
                    if pw in self.pathway_names:
                        # 累加多个表型的影响
                        features['pathway_modulations'][pw] = (
                            features['pathway_modulations'].get(pw, 0) + val
                        )
        
        # 3. 提取数值修饰（如"轻微"、"显著"、"极度"）
        if '轻微' in prompt or '轻度' in prompt:
            features['confidence'] = 0.5
        elif '显著' in prompt or '明显' in prompt:
            features['confidence'] = 1.5
        elif '极度' in prompt or '严重' in prompt:
            features['confidence'] = 2.0
            
        return features


class Prompt2GSVA(nn.Module):
    """
    神经网络：将结构化表型映射到GSVA通路活性谱
    输入: 表型特征向量
    输出: P维GSVA谱 (均值 + 方差，用于不确定性量化)
    """
    def __init__(self, n_phenotypes: int, n_pathways: int, 
                 hidden_dims: List[int] = [256, 512, 256]):
        super().__init__()
        
        # 表型编码器
        layers = []
        dims = [n_phenotypes] + hidden_dims
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.BatchNorm1d(dims[i+1]),
                nn.Dropout(0.2)
            ])
        self.encoder = nn.Sequential(*layers)
        
        # 均值预测头
        self.mu_head = nn.Linear(hidden_dims[-1], n_pathways)
        
        # 方差预测头（用于不确定性）
        self.logvar_head = nn.Linear(hidden_dims[-1], n_pathways)
        
        # 通路间相关性建模（协方差矩阵的低秩近似）
        self.corr_factor = nn.Parameter(torch.randn(n_pathways, 16) * 0.1)
        
    def forward(self, phenotype_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(phenotype_vec)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        
        # 构建协方差矩阵: Σ = diag(σ²) + FFᵀ
        std = torch.exp(0.5 * logvar)
        corr = self.corr_factor @ self.corr_factor.T
        
        return {
            'mu': mu,           # (batch, P)
            'std': std,         # (batch, P)
            'covariance': corr  # (P, P) 通路间相关性
        }


class GSVAGenerator:
    """
    完整的Prompt→GSVA生成器
    结合规则提取 + 神经网络预测
    """
    def __init__(self, 
                 model_path: Optional[str] = None,
                 pathway_names: Optional[List[str]] = None,
                 baseline_gsva: Optional[np.ndarray] = None):
        """
        baseline_gsva: 正常对照细胞的GSVA基线 (P,)
        """
        self.extractor = PhenotypeExtractor(pathway_names or [])
        self.n_pathways = len(pathway_names) if pathway_names else 50
        
        # 初始化或加载预训练模型
        self.model = Prompt2GSVA(
            n_phenotypes=100,  # 表型特征维度
            n_pathways=self.n_pathways
        )
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        
        self.baseline = baseline_gsva if baseline_gsva is not None else np.zeros(self.n_pathways)
        self.pathway_names = pathway_names or []
        
    def generate(self, prompt: str, 
                 temperature: float = 1.0,
                 n_samples: int = 1) -> Dict:
        """
        从Prompt生成GSVA谱
        
        Parameters:
        -----------
        prompt : str
            自然语言描述，如"生成肝肿瘤细胞，高增殖，TP53突变"
        temperature : float
            采样温度，控制多样性
        n_samples : int
            生成样本数（用于不确定性估计）
            
        Returns:
        --------
        dict: 包含GSVA谱、不确定性、解析信息
        """
        # 1. 解析Prompt
        features = self.extractor.parse_prompt(prompt)
        
        # 2. 构建表型向量（one-hot + 调制值）
        phenotype_vec = self._build_phenotype_vector(features)
        
        # 3. 神经网络预测
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.FloatTensor(phenotype_vec).unsqueeze(0))
            
        mu = pred['mu'].numpy()[0]
        std = pred['std'].numpy()[0] * temperature
        
        # 4. 结合规则先验（规则提取的通路调制）
        rule_prior = self._build_rule_prior(features)
        mu = 0.7 * mu + 0.3 * rule_prior  # 混合规则与神经网络
        
        # 5. 采样生成
        samples = []
        for _ in range(n_samples):
            noise = np.random.normal(0, std)
            gsva_sample = self.baseline + mu + noise
            samples.append(gsva_sample)
            
        return {
            'gsva_mean': self.baseline + mu,
            'gsva_std': std,
            'gsva_samples': np.array(samples),
            'pathway_names': self.pathway_names,
            'parsed_features': features,
            'prompt': prompt
        }
    
    def _build_phenotype_vector(self, features: Dict) -> np.ndarray:
        """将结构化特征转换为模型输入向量"""
        # 简化版：使用稀疏编码
        vec = np.zeros(100)
        # 编码细胞类型
        cell_type_idx = hash(features['cell_type']) % 20 if features['cell_type'] else 0
        vec[cell_type_idx] = 1.0
        # 编码通路调制强度
        for i, pw in enumerate(self.pathway_names[:50]):
            if pw in features['pathway_modulations']:
                vec[20 + i] = features['pathway_modulations'][pw]
        return vec
    
    def _build_rule_prior(self, features: Dict) -> np.ndarray:
        """从规则提取构建GSVA先验"""
        prior = np.zeros(self.n_pathways)
        for pw, val in features['pathway_modulations'].items():
            if pw in self.pathway_names:
                idx = self.pathway_names.index(pw)
                prior[idx] = val * features['confidence']
        return prior
import numpy as np
from scipy.optimize import minimize
from typing import Callable, Tuple

class GSVA2L1000Solver:
    """
    从GSVA通路活性迭代求解LINCS L1000基因表达谱
    核心：将GSVA作为黑盒函数，使用优化算法求解
    """
    
    def __init__(self,
                 l1000_database: np.ndarray,  # (N_samples, 978)
                 gsva_calculator: Callable,     # GSVA计算函数
                 gene_names: List[str],
                 pathway_names: List[str]):
        """
        l1000_database: LINCS L1000表达矩阵，用于构建统计约束
        """
        self.X_db = l1000_database
        self.gsva_fn = gsva_calculator
        self.gene_names = gene_names
        self.pathway_names = pathway_names
        
        # 预计算统计量
        self._compute_stats()
        
    def _compute_stats(self):
        """计算LINCS数据的统计约束"""
        self.gene_mean = self.X_db.mean(axis=0)
        self.gene_std = self.X_db.std(axis=0)
        # 基因间协方差（用于流形约束）
        self.gene_cov = np.cov(self.X_db.T)
        # PCA基（主成分）
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=50)
        self.pca.fit(self.X_db)
        
    def solve(self, 
              s_target: np.ndarray,
              method: str = 'hybrid',
              max_iter: int = 500,
              verbose: bool = True) -> Dict:
        """
        求解目标GSVA对应的L1000表达谱
        
        Parameters:
        -----------
        s_target : (P,) 目标GSVA通路活性
        method : str
            'direct' - 直接梯度优化
            'nn_init' - 神经网络初始化 + 微调
            'hybrid' - 最近邻初始化 + 迭代优化
            
        Returns:
        --------
        dict: 包含表达谱、收敛信息、质量评估
        """
        # 1. 初始化
        if method == 'hybrid':
            x0 = self._hybrid_initialization(s_target)
        else:
            x0 = np.zeros(978)
            
        # 2. 定义目标函数
        def objective(x):
            s_pred = self.gsva_fn(x)
            # 通路匹配损失
            pathway_loss = np.mean((s_pred - s_target) ** 2)
            # 生物学合理性正则化
            biology_loss = self._biology_regularizer(x)
            return pathway_loss + 0.1 * biology_loss
        
        # 3. 优化求解
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            jac='2-point',  # 数值梯度
            bounds=[(-5, 5)] * 978,  # z-score范围约束
            options={'maxiter': max_iter, 'disp': verbose}
        )
        
        x_opt = result.x
        
        # 4. 后处理：投影到数据流形
        x_final = self._project_to_manifold(x_opt)
        
        # 5. 评估
        s_final = self.gsva_fn(x_final)
        
        return {
            'expression_profile': x_final,
            'gsva_achieved': s_final,
            'gsva_target': s_target,
            'pathway_mse': np.mean((s_final - s_target) ** 2),
            'pathway_correlation': np.corrcoef(s_final, s_target)[0,1],
            'optimization_success': result.success,
            'n_iterations': result.nit
        }
    
    def _hybrid_initialization(self, s_target: np.ndarray) -> np.ndarray:
        """
        混合初始化策略：
        1. 在LINCS数据库中找GSVA最接近的样本
        2. 用其基因表达作为初始化
        """
        # 预计算数据库中所有样本的GSVA（假设已缓存）
        # 这里简化：随机采样找近似
        best_idx = 0
        best_dist = float('inf')
        
        # 为效率，随机采样100个候选
        candidates = np.random.choice(len(self.X_db), min(100, len(self.X_db)), replace=False)
        
        for idx in candidates:
            s_candidate = self.gsva_fn(self.X_db[idx])
            dist = np.linalg.norm(s_candidate - s_target)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
                
        return self.X_db[best_idx].copy()
    
    def _biology_regularizer(self, x: np.ndarray) -> float:
        """生物学合理性正则化项"""
        reg = 0.0
        
        # 1. 与LINCS数据分布的KL散度近似
        z_scores = (x - self.gene_mean) / (self.gene_std + 1e-8)
        reg += np.mean(z_scores ** 2)  # 防止偏离太远
        
        # 2. 稀疏性：L1000中通常只有少数基因显著变化
        reg += 0.01 * np.sum(np.abs(x) > 3)  # 惩罚过多异常值
        
        # 3. 基因间相关性结构
        # 投影到PCA空间，约束在主成分上
        pca_proj = self.pca.transform(x.reshape(1, -1))[0]
        reg += 0.001 * np.sum(pca_proj[10:] ** 2)  # 惩罚高阶成分
        
        return reg
    
    def _project_to_manifold(self, x: np.ndarray) -> np.ndarray:
        """将解投影到LINCS数据流形"""
        # 1. 范围裁剪
        x = np.clip(x, -5, 5)
        
        # 2. 在PCA空间重构（去噪）
        pca_proj = self.pca.transform(x.reshape(1, -1))[0]
        # 保留主要成分，去除噪声
        pca_proj[30:] = 0  # 截断高阶成分
        x_denoised = self.pca.inverse_transform(pca_proj.reshape(1, -1))[0]
        
        # 3. 混合：保持优化结果与数据流形的平衡
        alpha = 0.7
        x_final = alpha * x + (1 - alpha) * x_denoised
        
        return x_final


class DifferentiableGSVA(torch.autograd.Function):
    """
    可微分的GSVA近似实现
    使用soft-sort替代硬排序，使GSVA可反向传播
    """
    @staticmethod
    def forward(ctx, x, gene_sets, pathway_genes):
        """
        x: (batch, 978) 基因表达
        gene_sets: List[List[int]] 通路基因索引
        """
        ctx.save_for_backward(x)
        ctx.gene_sets = gene_sets
        
        # 标准GSVA计算（基于排序）
        s = []
        for gs in gene_sets:
            pathway_genes = x[:, gs]
            # 使用torchsort进行软排序
            ranks = torchsort.soft_rank(pathway_genes, regularization_strength=0.1)
            # KS统计量计算...
            ks_stat = compute_ks_statistic(ranks)
            s.append(ks_stat)
            
        return torch.stack(s, dim=1)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # 通过软排序的梯度自动计算
        # 实际实现需要torchsort库支持
        return grad_output, None, None
class Prompt2L1000Pipeline:
    """
    完整流水线: Prompt → GSVA → L1000
    """
    def __init__(self,
                 gsva_generator: GSVAGenerator,
                 l1000_solver: GSVA2L1000Solver,
                 output_formatter: Optional[Callable] = None):
        self.gsva_gen = gsva_generator
        self.l1000_solver = l1000_solver
        self.formatter = output_formatter or self._default_formatter
        
    def generate(self, 
                 prompt: str,
                 n_gsva_samples: int = 5,
                 temperature: float = 1.0,
                 return_intermediate: bool = False) -> Dict:
        """
        从Prompt生成L1000基因表达谱
        
        Parameters:
        -----------
        prompt : str
            自然语言描述细胞状态
        n_gsva_samples : int
            GSVA采样数（用于不确定性估计）
        temperature : float
            生成温度
        return_intermediate : bool
            是否返回中间结果（GSVA谱）
            
        Returns:
        --------
        dict: 包含L1000表达谱、GSVA谱、质量评估
        """
        # Step 1: Prompt → GSVA
        gsva_result = self.gsva_gen.generate(
            prompt, 
            temperature=temperature,
            n_samples=n_gsva_samples
        )
        
        # Step 2: 对每个GSVA样本求解L1000
        l1000_solutions = []
        for i, s_target in enumerate(gsva_result['gsva_samples']):
            sol = self.l1000_solver.solve(s_target, method='hybrid')
            l1000_solutions.append(sol)
            
        # Step 3: 选择最优解（通路吻合度最高）
        best_idx = np.argmin([s['pathway_mse'] for s in l1000_solutions])
        best_solution = l1000_solutions[best_idx]
        
        # Step 4: 格式化输出
        result = {
            'prompt': prompt,
            'gsva_target': gsva_result['gsva_mean'],
            'gsva_uncertainty': gsva_result['gsva_std'],
            'l1000_expression': best_solution['expression_profile'],
            'pathway_mse': best_solution['pathway_mse'],
            'pathway_correlation': best_solution['pathway_correlation'],
            'gene_names': self.l1000_solver.gene_names,
            'pathway_names': self.gsva_gen.pathway_names
        }
        
        if return_intermediate:
            result['gsva_samples'] = gsva_result['gsva_samples']
            result['all_l1000_solutions'] = l1000_solutions
            
        return result
    
    def _default_formatter(self, result: Dict) -> pd.DataFrame:
        """默认输出格式：LINCS L1000标准格式"""
        df = pd.DataFrame({
            'pr_gene_id': result['gene_names'],
            'z_score': result['l1000_expression'],
            'gsva_contribution': np.abs(result['l1000_expression'])  # 简化
        })
        return df


# ============================================
# 使用示例
# ============================================

def demo():
    """演示：生成肝肿瘤细胞的L1000表达谱"""
    
    # 初始化组件（实际使用时需要加载真实数据）
    pathway_names = [
        'Hallmark_E2F_Targets',
        'Hallmark_G2M_Checkpoint', 
        'Hallmark_Epithelial_Mesenchymal_Transition',
        'Hallmark_Glycolysis',
        'Hallmark_Hypoxia',
        'Hallmark_P53_Pathway',
        'Hallmark_Apoptosis',
        'Hallmark_Inflammatory_Response',
        'Hallmark_Interferon_Gamma_Response',
        'Hallmark_KRAS_Signaling_Up',
        # ... 更多通路
    ]
    
    # 1. 初始化GSVA生成器
    gsva_gen = GSVAGenerator(
        pathway_names=pathway_names,
        baseline_gsva=np.zeros(len(pathway_names))  # 正常肝细胞基线
    )
    
    # 2. 初始化L1000求解器（需要真实LINCS数据）
    # l1000_db = load_lincs_data()  # 加载LINCS L1000数据库
    # solver = GSVA2L1000Solver(l1000_db, gsva_calculator, gene_names, pathway_names)
    
    # 3. 构建流水线
    # pipeline = Prompt2L1000Pipeline(gsva_gen, solver)
    
    # 4. 生成
    prompt = "生成肝肿瘤细胞，高增殖，侵袭性强，TP53突变，缺氧微环境"
    # result = pipeline.generate(prompt, n_gsva_samples=10)
    
    print(f"Prompt: {prompt}")
    print("系统架构就绪，等待LINCS数据加载...")
    
    return gsva_gen


if __name__ == "__main__":
    demo()