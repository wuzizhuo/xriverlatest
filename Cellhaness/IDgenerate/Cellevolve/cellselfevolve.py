import os
import sys
import json
import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')

class CellAgent:
    """细胞智能体类 - 具有学习能力的细胞"""
    
    def __init__(self, cell_id, cell_type, initial_expression, drugs):
        self.cell_id = cell_id
        self.cell_type = cell_type
        self.expression = initial_expression.copy()
        self.drugs = drugs
        self.history = []
        self.fitness = 0.0
        self.best_expression = initial_expression.copy()
        self.best_fitness = 0.0
        self.drug_preferences = {drug: 0.5 for drug in drugs}  # 药物偏好
    
    def select_smile(self, strategy='exploration'):
        """根据策略选择smile"""
        if strategy == 'exploitation':
            best_drug = max(self.drug_preferences, key=self.drug_preferences.get)
            return best_drug
        elif strategy == 'exploration':
            weights = [self.drug_preferences[d] for d in self.drugs]
            weights = np.array(weights) / sum(weights)
            return np.random.choice(self.drugs, p=weights)
        else:
            return random.choice(self.drugs)
    
    def update_drug_preference(self, drug, reward):
        """根据反馈更新药物偏好"""
        learning_rate = 0.1
        self.drug_preferences[drug] = min(1.0, max(0.0, 
            self.drug_preferences[drug] + learning_rate * reward))
    
    def apply_gsva_ldm(self, smile, gsva_model):
        """通过GSVA-LDM生成新的细胞状态"""
        perturbation = gsva_model.predict(self.expression, smile)
        new_expression = self.expression + perturbation
        new_expression += np.random.normal(0, 0.005, size=self.expression.shape)
        new_expression = np.maximum(new_expression, 0)
        
        if new_expression.sum() > 0:
            new_expression = new_expression / new_expression.sum()
        
        return new_expression
    
    def update_lr(self, lr_model):
        """在LR空间中更新细胞状态"""
        updated = lr_model.update(self.expression)
        return updated
    
    def compute_fitness(self, target_interaction, current_interaction):
        """计算适应度"""
        if target_interaction.size == 0 or current_interaction.size == 0:
            return 0.0
        
        target_flat = target_interaction.flatten()
        current_flat = current_interaction.flatten()
        
        norm_target = np.linalg.norm(target_flat)
        norm_current = np.linalg.norm(current_flat)
        
        if norm_target == 0 or norm_current == 0:
            return 0.0
        
        similarity = np.dot(target_flat, current_flat) / (norm_target * norm_current)
        self.fitness = similarity
        
        if similarity > self.best_fitness:
            self.best_fitness = similarity
            self.best_expression = self.expression.copy()
        
        return similarity

class RealGSVAModel:
    """基于真实通路的GSVA-LDM模型"""
    
    def __init__(self, n_genes, gene_names):
        self.n_genes = n_genes
        self.gene_names = gene_names
        
        # 真实的Hallmark通路基因集合（来自MSigDB）
        self.pathways = {
            'inflammatory_response': ['IL6', 'IL6R', 'IL6ST', 'IFNG', 'IFNGR1', 'IFNGR2', 
                                     'TNF', 'TNFAIP3', 'IL1B', 'IL1R1', 'NFKB1', 'RELA'],
            'interferon_gamma_response': ['IFNG', 'IFNGR1', 'IFNGR2', 'IRF1', 'STAT1', 
                                          'MX1', 'OAS1', 'OAS2', 'OAS3', 'GBP1', 'GBP2'],
            'interferon_alpha_response': ['IFNA1', 'IFNA2', 'IFNAR1', 'IFNAR2', 'IRF7', 
                                          'STAT2', 'MX1', 'OAS1', 'EIF2AK2', 'ISG15'],
            'tumor_necrosis_factor_signaling': ['TNF', 'TNRSF1A', 'TNRSF1B', 'TRAF2', 
                                                'TRAF6', 'NFKB1', 'RELA', 'JUN', 'FOS'],
            'chemokine_signaling': ['CXCL1', 'CXCL2', 'CXCL8', 'CXCL10', 'CXCL12', 
                                    'CXCR3', 'CXCR4', 'CCL2', 'CCL3', 'CCR2', 'CCR5'],
            'il2_stat5_signaling': ['IL2', 'IL2RA', 'IL2RB', 'JAK1', 'JAK3', 'STAT5A', 
                                    'STAT5B', 'BCL2', 'MYC', 'CDKN1A'],
            'il6_jak_stat3_signaling': ['IL6', 'IL6R', 'IL6ST', 'JAK1', 'JAK2', 'STAT3', 
                                         'SOCS3', 'MYC', 'BCL2L1', 'CCND1'],
            'complement': ['C1QA', 'C1QB', 'C1QC', 'C3', 'C4A', 'C4B', 'C5', 'C6', 
                          'C7', 'C8A', 'C8B', 'C9'],
            'apoptosis': ['CASP3', 'CASP8', 'CASP9', 'BAX', 'BCL2', 'FAS', 'FASLG', 
                          'TNFRSF1A', 'TP53', 'CYCS'],
            'cell_cycle': ['CDK1', 'CDK2', 'CDK4', 'CCNB1', 'CCNA2', 'CDC20', 
                          'CCNE1', 'PCNA', 'MCM2', 'MCM3'],
        }
        
        self.perturbation_matrices = {}
        for pathway, genes in self.pathways.items():
            mask = np.zeros(n_genes)
            for gene in genes:
                if gene in gene_names:
                    mask[gene_names.index(gene)] = 1.0
            self.perturbation_matrices[pathway] = mask
    
    def _smile_to_pathway(self, smile):
        """将smile映射到通路"""
        smile_hash = hash(smile)
        pathways = list(self.pathways.keys())
        return pathways[smile_hash % len(pathways)]
    
    def predict(self, expression, smile):
        """根据smile和当前表达生成扰动"""
        pathway = self._smile_to_pathway(smile)
        mask = self.perturbation_matrices[pathway]
        
        perturbation = expression * mask * 0.1
        perturbation += np.random.randn(self.n_genes) * 0.02
        
        return perturbation

class RealLRModel:
    """基于真实受配体数据库的LR-LDM模型"""
    
    def __init__(self, n_genes, gene_names):
        self.n_genes = n_genes
        self.gene_names = gene_names
        
        # 真实的人类受配体对（来自CellChatDB）
        self.lr_pairs = [
            # 细胞因子-细胞因子受体
            ('IL6', 'IL6R', 1.0), ('IL6', 'IL6ST', 0.8),
            ('IL1B', 'IL1R1', 1.0), ('IL1B', 'IL1R2', 0.6),
            ('IL1A', 'IL1R1', 1.0), ('IL1A', 'IL1R2', 0.6),
            ('IL2', 'IL2RA', 1.0), ('IL2', 'IL2RB', 0.9), ('IL2', 'IL2RG', 0.8),
            ('IL4', 'IL4R', 1.0), ('IL4', 'IL2RG', 0.7),
            ('IL5', 'IL5RA', 1.0), ('IL5', 'IL2RG', 0.7),
            ('IL7', 'IL7R', 1.0), ('IL7', 'IL2RG', 0.7),
            ('IL9', 'IL9R', 1.0), ('IL9', 'IL2RG', 0.7),
            ('IL10', 'IL10RA', 1.0), ('IL10', 'IL10RB', 0.9),
            ('IL12A', 'IL12RB1', 1.0), ('IL12B', 'IL12RB2', 0.9),
            ('IL13', 'IL13RA1', 1.0), ('IL13', 'IL4R', 0.8),
            ('IL15', 'IL15RA', 1.0), ('IL15', 'IL2RB', 0.9), ('IL15', 'IL2RG', 0.8),
            ('IL17A', 'IL17RA', 1.0), ('IL17A', 'IL17RC', 0.9),
            ('IL17F', 'IL17RA', 1.0), ('IL17F', 'IL17RC', 0.9),
            ('IL18', 'IL18R1', 1.0), ('IL18', 'IL18RAP', 0.8),
            ('IL21', 'IL21R', 1.0), ('IL21', 'IL2RG', 0.7),
            ('IFNG', 'IFNGR1', 1.0), ('IFNG', 'IFNGR2', 0.9),
            ('IFNA1', 'IFNAR1', 1.0), ('IFNA1', 'IFNAR2', 0.9),
            ('TNF', 'TNFRSF1A', 1.0), ('TNF', 'TNFRSF1B', 0.7),
            ('TNFSF10', 'TNFRSF10A', 1.0), ('TNFSF10', 'TNFRSF10B', 0.9),
            
            # 趋化因子-趋化因子受体
            ('CXCL1', 'CXCR2', 1.0), ('CXCL1', 'CXCR1', 0.8),
            ('CXCL2', 'CXCR2', 1.0), ('CXCL2', 'CXCR1', 0.8),
            ('CXCL8', 'CXCR1', 1.0), ('CXCL8', 'CXCR2', 0.9),
            ('CXCL10', 'CXCR3', 1.0),
            ('CXCL11', 'CXCR3', 1.0),
            ('CXCL12', 'CXCR4', 1.0), ('CXCL12', 'CXCR7', 0.7),
            ('CXCL13', 'CXCR5', 1.0),
            ('CCL2', 'CCR2', 1.0),
            ('CCL3', 'CCR1', 1.0), ('CCL3', 'CCR5', 0.9),
            ('CCL4', 'CCR1', 1.0), ('CCL4', 'CCR5', 0.9),
            ('CCL5', 'CCR1', 1.0), ('CCL5', 'CCR3', 0.9), ('CCL5', 'CCR5', 0.8),
            ('CCL19', 'CCR7', 1.0),
            ('CCL21', 'CCR7', 1.0),
            ('XCL1', 'XCR1', 1.0),
            
            # 免疫检查点
            ('CD28', 'CD80', 1.0), ('CD28', 'CD86', 0.9),
            ('CTLA4', 'CD80', 1.0), ('CTLA4', 'CD86', 0.9),
            ('PDCD1', 'CD274', 1.0), ('PDCD1', 'PDCD1LG2', 0.9),
            ('LAG3', 'HLA-DRA', 1.0), ('LAG3', 'HLA-DRB1', 0.8),
            ('TIGIT', 'CD155', 1.0), ('TIGIT', 'CD112', 0.9),
            ('TIM3', 'HAVCR2', 1.0),
            
            # TNF超家族
            ('CD40LG', 'CD40', 1.0),
            ('FASLG', 'FAS', 1.0),
            ('CD70', 'CD27', 1.0),
            ('CD30LG', 'CD30', 1.0),
            ('OX40LG', 'TNFRSF4', 1.0),
            ('41BBLG', 'TNFRSF9', 1.0),
            ('GITRL', 'TNFRSF18', 1.0),
            
            # 生长因子
            ('EGF', 'EGFR', 1.0),
            ('TGFB1', 'TGFBR1', 1.0), ('TGFB1', 'TGFBR2', 0.9),
            ('VEGFA', 'KDR', 1.0), ('VEGFA', 'FLT1', 0.9),
            ('PDGFA', 'PDGFRA', 1.0), ('PDGFA', 'PDGFRB', 0.8),
            ('FGF2', 'FGFR1', 1.0), ('FGF2', 'FGFR2', 0.8),
            
            # 整合素
            ('ICAM1', 'ITGB2', 1.0), ('ICAM1', 'ITGA4', 0.8),
            ('VCAM1', 'ITGA4', 1.0), ('VCAM1', 'ITGB1', 0.9),
            ('SELL', 'SELP', 1.0), ('SELL', 'SELPLG', 0.9),
        ]
    
    def update(self, expression):
        """更新表达以增强LR信号"""
        new_expression = expression.copy()
        
        for ligand, receptor, strength in self.lr_pairs:
            if ligand in self.gene_names:
                idx = self.gene_names.index(ligand)
                new_expression[idx] = np.minimum(1.0, new_expression[idx] * (1.0 + 0.05 * strength))
            
            if receptor in self.gene_names:
                idx = self.gene_names.index(receptor)
                new_expression[idx] = np.minimum(1.0, new_expression[idx] * (1.0 + 0.05 * strength))
        
        new_expression += np.random.normal(0, 0.003, size=self.n_genes)
        new_expression = np.maximum(0, new_expression)
        
        if new_expression.sum() > 0:
            new_expression = new_expression / new_expression.sum()
        
        return new_expression

class CellPopulation:
    """细胞群体类"""
    
    def __init__(self, cells):
        self.cells = cells
        self.n_cells = len(cells)
    
    def compute_interaction_matrix(self, lr_pairs, gene_names):
        """计算细胞间互作矩阵"""
        cell_types = sorted(set([c.cell_type for c in self.cells]))
        n_cell_types = len(cell_types)
        type_to_idx = {t: i for i, t in enumerate(cell_types)}
        
        interaction_matrix = np.zeros((n_cell_types, n_cell_types))
        
        cells_by_type = {}
        for cell in self.cells:
            if cell.cell_type not in cells_by_type:
                cells_by_type[cell.cell_type] = []
            cells_by_type[cell.cell_type].append(cell)
        
        avg_expression = {}
        for cell_type, cells in cells_by_type.items():
            exprs = np.array([c.expression for c in cells])
            avg_expression[cell_type] = exprs.mean(axis=0)
        
        for ligand, receptor, _ in lr_pairs:
            if ligand not in gene_names or receptor not in gene_names:
                continue
            
            ligand_idx = gene_names.index(ligand)
            receptor_idx = gene_names.index(receptor)
            
            for source_type in cell_types:
                for target_type in cell_types:
                    if source_type == target_type:
                        continue
                    
                    source_expr = avg_expression[source_type][ligand_idx]
                    target_expr = avg_expression[target_type][receptor_idx]
                    
                    interaction_matrix[type_to_idx[source_type], 
                                     type_to_idx[target_type]] += source_expr * target_expr
        
        if interaction_matrix.sum() > 0:
            interaction_matrix = interaction_matrix / interaction_matrix.sum()
        
        return interaction_matrix, cell_types
    
    def evolve(self, gsva_model, lr_model, target_interaction, lr_pairs, gene_names, step):
        """执行一代进化"""
        if step < 10:
            strategy = 'exploration'
        else:
            strategy = 'exploitation' if random.random() > 0.3 else 'exploration'
        
        rewards = []
        
        for cell in self.cells:
            smile = cell.select_smile(strategy)
            old_expression = cell.expression.copy()
            
            cell.expression = cell.apply_gsva_ldm(smile, gsva_model)
            cell.expression = cell.update_lr(lr_model)
            
            expr_change = np.linalg.norm(cell.expression - old_expression)
            cell.history.append({
                'smile': smile,
                'expression_change': expr_change
            })
        
        current_interaction, cell_types = self.compute_interaction_matrix(lr_pairs, gene_names)
        
        for cell in self.cells:
            old_fitness = cell.fitness
            new_fitness = cell.compute_fitness(target_interaction, current_interaction)
            
            reward = new_fitness - old_fitness
            
            if len(cell.history) > 0:
                last_smile = cell.history[-1]['smile']
                cell.update_drug_preference(last_smile, reward)
            
            rewards.append(reward)
        
        best_cell = max(self.cells, key=lambda c: c.fitness)
        if best_cell.fitness > 0.7:
            for cell in self.cells:
                if cell.fitness < 0.5 * best_cell.fitness:
                    cell.expression = best_cell.best_expression.copy() * 0.9 + cell.expression * 0.1
        
        return current_interaction, cell_types, np.mean(rewards)

class CellSelfEvolver:
    """细胞自进化主类"""
    
    def __init__(self, initial_cells, target_interaction, drugs, gene_names):
        self.population = CellPopulation(initial_cells)
        self.target_interaction = target_interaction
        self.drugs = drugs
        self.gene_names = gene_names
        
        n_genes = len(gene_names)
        self.gsva_model = RealGSVAModel(n_genes, gene_names)
        self.lr_model = RealLRModel(n_genes, gene_names)
        
        self.lr_pairs = self.lr_model.lr_pairs
        self.evolution_history = []
    
    def run_evolution(self, max_steps=100, tolerance=0.95):
        """运行进化过程"""
        print("=" * 70)
        print("CELL SELF-EVOLUTION WITH REAL DATA")
        print("=" * 70)
        print(f"Initial population: {self.population.n_cells} cells")
        print(f"Target interaction shape: {self.target_interaction.shape}")
        print(f"Number of drugs: {len(self.drugs)}")
        print(f"Number of LR pairs: {len(self.lr_pairs)}")
        print(f"Max steps: {max_steps}")
        print(f"Convergence tolerance: {tolerance}")
        print("=" * 70)
        
        best_similarity = 0.0
        
        for step in range(max_steps):
            print(f"\n--- Step {step + 1}/{max_steps} ---")
            
            current_interaction, cell_types, avg_reward = self.population.evolve(
                self.gsva_model, 
                self.lr_model, 
                self.target_interaction,
                self.lr_pairs,
                self.gene_names,
                step
            )
            
            target_flat = self.target_interaction.flatten()
            current_flat = current_interaction.flatten()
            
            norm_target = np.linalg.norm(target_flat)
            norm_current = np.linalg.norm(current_flat)
            
            similarity = np.dot(target_flat, current_flat) / (norm_target * norm_current) if (norm_target > 0 and norm_current > 0) else 0.0
            
            if similarity > best_similarity:
                best_similarity = similarity
            
            self.evolution_history.append({
                'step': step + 1,
                'similarity': similarity,
                'best_similarity': best_similarity,
                'avg_fitness': np.mean([c.fitness for c in self.population.cells]),
                'avg_reward': avg_reward,
                'interaction_matrix': current_interaction.copy()
            })
            
            print(f"  Similarity to target: {similarity:.4f}")
            print(f"  Best similarity: {best_similarity:.4f}")
            print(f"  Average fitness: {np.mean([c.fitness for c in self.population.cells]):.4f}")
            print(f"  Average reward: {avg_reward:.4f}")
            
            if similarity >= tolerance:
                print(f"\n✓ Converged at step {step + 1}!")
                print(f"  Final similarity: {similarity:.4f}")
                break
            
            if (step + 1) % 10 == 0:
                print("\n  Cell type distribution:")
                type_counts = {}
                for cell in self.population.cells:
                    type_counts[cell.cell_type] = type_counts.get(cell.cell_type, 0) + 1
                
                for cell_type, count in type_counts.items():
                    print(f"    {cell_type}: {count}")
                
                print("\n  Top drug preferences:")
                avg_preferences = {d: 0.0 for d in self.drugs}
                for cell in self.population.cells:
                    for d, pref in cell.drug_preferences.items():
                        avg_preferences[d] += pref / self.population.n_cells
                
                for drug, pref in sorted(avg_preferences.items(), key=lambda x: -x[1])[:3]:
                    print(f"    {drug}: {pref:.3f}")
        
        print("\n" + "=" * 70)
        print("EVOLUTION COMPLETE")
        print("=" * 70)
        
        return self.evolution_history, current_interaction, cell_types

class RealDataLoader:
    """加载真实单细胞数据"""
    
    def __init__(self):
        self.data_dir = '/tmp/real_sc_data'
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load_pbmc3k(self):
        """加载PBMC 3K真实单细胞数据"""
        try:
            import scanpy as sc
            
            adata_path = os.path.join(self.data_dir, 'pbmc3k.h5ad')
            if os.path.exists(adata_path):
                print("Loading PBMC 3K data...")
                adata = sc.read_h5ad(adata_path)
                
                print(f"✓ Raw data: {adata.n_obs} cells, {adata.n_vars} genes")
                
                # 预处理
                sc.pp.filter_cells(adata, min_genes=200)
                sc.pp.filter_genes(adata, min_cells=3)
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                
                # 细胞类型注释（简化版）
                marker_genes = {
                    'CD4+ T': ['IL7R', 'CCR7'],
                    'CD8+ T': ['CD8A', 'CD8B'],
                    'B': ['MS4A1', 'CD79A'],
                    'Monocyte': ['CD14', 'LYZ'],
                    'NK': ['GNLY', 'NKG7'],
                }
                
                # 根据标记基因表达分配细胞类型
                cell_types = []
                for i in range(adata.n_obs):
                    max_score = 0
                    assigned_type = 'Unknown'
                    for cell_type, markers in marker_genes.items():
                        score = sum(adata[i, marker].X[0, 0] if hasattr(adata[i, marker].X, 'shape') else adata[i, marker].X for marker in markers if marker in adata.var_names)
                        if score > max_score:
                            max_score = score
                            assigned_type = cell_type
                    cell_types.append(assigned_type)
                
                adata.obs['cell_type'] = cell_types
                
                # 筛选受配体相关基因
                lr_genes = set()
                for ligand, receptor, _ in RealLRModel(1, []).lr_pairs:
                    lr_genes.add(ligand)
                    lr_genes.add(receptor)
                
                # 选择与受配体相关的基因
                common_genes = [g for g in adata.var_names if g in lr_genes]
                if len(common_genes) < 20:
                    # 如果匹配太少，使用高变基因
                    sc.pp.highly_variable_genes(adata, n_top_genes=200)
                    adata = adata[:, adata.var.highly_variable]
                else:
                    adata = adata[:, common_genes]
                
                # 标准化表达矩阵
                sc.pp.scale(adata, max_value=10)
                
                print(f"✓ After filtering: {adata.n_obs} cells, {adata.n_vars} genes")
                
                return adata
            
            else:
                print("PBMC 3K data not found, downloading...")
                self.download_pbmc3k()
                return self.load_pbmc3k()
        
        except Exception as e:
            print(f"Failed to load real data: {e}")
            print("Using simulated data instead...")
            return None
    
    def download_pbmc3k(self):
        """下载PBMC 3K数据"""
        import scanpy as sc
        adata = sc.datasets.pbmc3k()
        adata.write_h5ad(os.path.join(self.data_dir, 'pbmc3k.h5ad'))
        
        # 保存表达矩阵和基因列表
        np.save(os.path.join(self.data_dir, 'pbmc3k_expression.npy'), adata.X.toarray())
        pd.DataFrame({'gene': adata.var_names.tolist()}).to_csv(
            os.path.join(self.data_dir, 'pbmc3k_genes.csv'), index=False)
        pd.DataFrame({'sample': adata.obs_names.tolist()}).to_csv(
            os.path.join(self.data_dir, 'pbmc3k_samples.csv'), index=False)
        
        print("✓ PBMC 3K downloaded successfully")
    
    def get_real_drugs(self):
        """获取真实药物SMILE"""
        drugs = [
            # 抗炎药物
            'CC(=O)OC1=CC=CC=C1C(=O)O',           # Aspirin (阿司匹林)
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',     # Ibuprofen (布洛芬)
            'C1=CC=CC=C1C(=O)O',                 # Benzoic acid (苯甲酸)
            'C1CCCCC1C(=O)O',                    # Cyclohexanecarboxylic acid
            
            # 免疫调节剂
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',     # Caffeine (咖啡因)
            'CCOC(=O)C(C)CC1=CC=C(C=C1)O',       # Phenoxyacetic acid derivative
            
            # 细胞因子抑制剂
            'CN(C)CC1=CC=C(C=C1)O',              # Phenol derivative
            'C1=CC=C(C=C1)CC(=O)O',              # Phenylacetic acid
            
            # 免疫检查点抑制剂相关
            'C1CCN(CC1)C(=O)C1=CC=CC=C1',        # Carbanilide
            'CN(C)C(=O)C1=CC=C(C=C1)NC(=O)N',    # Urea derivative
            
            # 类固醇类
            'C1=CC=C2C(=C1)C(=O)CC2',            # Indan-1-one
            
            # 抗体药物相关结构
            'C(=O)(O)CC1=CC=C(C=C1)N',           # 2-Aminophenylacetic acid
        ]
        return drugs
    
    def compute_real_interaction_matrix(self, adata):
        """计算真实的细胞间互作矩阵"""
        lr_model = RealLRModel(adata.n_vars, adata.var_names.tolist())
        lr_pairs = lr_model.lr_pairs
        
        cell_types = adata.obs['cell_type'].unique().tolist()
        n_cell_types = len(cell_types)
        type_to_idx = {t: i for i, t in enumerate(cell_types)}
        
        interaction_matrix = np.zeros((n_cell_types, n_cell_types))
        
        for ligand, receptor, _ in lr_pairs:
            if ligand not in adata.var_names or receptor not in adata.var_names:
                continue
            
            ligand_expr = adata[:, ligand].X.toarray() if hasattr(adata[:, ligand].X, 'toarray') else adata[:, ligand].X
            receptor_expr = adata[:, receptor].X.toarray() if hasattr(adata[:, receptor].X, 'toarray') else adata[:, receptor].X
            
            for i, source_type in enumerate(cell_types):
                for j, target_type in enumerate(cell_types):
                    if i == j:
                        continue
                    
                    source_mask = adata.obs['cell_type'] == source_type
                    target_mask = adata.obs['cell_type'] == target_type
                    
                    if source_mask.sum() == 0 or target_mask.sum() == 0:
                        continue
                    
                    source_ligand = ligand_expr[source_mask].mean()
                    target_receptor = receptor_expr[target_mask].mean()
                    
                    interaction_matrix[i, j] += source_ligand * target_receptor
        
        if interaction_matrix.sum() > 0:
            interaction_matrix = interaction_matrix / interaction_matrix.sum()
        
        return interaction_matrix, cell_types

def main():
    print("=" * 70)
    print("CELL SELF-EVOLUTION WITH REAL DATA")
    print("=" * 70)
    
    # 加载真实数据
    data_loader = RealDataLoader()
    adata = data_loader.load_pbmc3k()
    
    if adata is not None:
        print("\n✓ Using REAL single-cell data")
        print(f"  Dataset: PBMC 3K")
        print(f"  Cells: {adata.n_obs}")
        print(f"  Genes: {adata.n_vars}")
        print(f"  Cell types: {adata.obs['cell_type'].nunique()}")
        print(adata.obs['cell_type'].value_counts())
        
        # 计算真实的互作矩阵作为目标
        print("\nComputing target interaction matrix from real data...")
        target_interaction, target_cell_types = data_loader.compute_real_interaction_matrix(adata)
        print(f"✓ Target interaction matrix shape: {target_interaction.shape}")
        print(f"  Cell types: {target_cell_types}")
        
        # 获取真实药物
        drugs = data_loader.get_real_drugs()
        print(f"\n✓ Loaded {len(drugs)} real drug SMILEs")
        
        # 从真实数据中采样细胞
        n_cells = 100
        print(f"\nSampling {n_cells} cells from real data...")
        
        # 平衡采样每种细胞类型
        sampled_indices = []
        for cell_type in target_cell_types:
            type_indices = np.where(adata.obs['cell_type'] == cell_type)[0]
            n_sample = min(len(type_indices), n_cells // len(target_cell_types))
            sampled_indices.extend(np.random.choice(type_indices, n_sample, replace=False))
        
        # 如果采样不足，随机补充
        if len(sampled_indices) < n_cells:
            remaining = n_cells - len(sampled_indices)
            all_indices = np.arange(adata.n_obs)
            remaining_indices = np.setdiff1d(all_indices, sampled_indices)
            sampled_indices.extend(np.random.choice(remaining_indices, remaining, replace=False))
        
        sampled_adata = adata[sampled_indices].copy()
        
        # 创建细胞智能体
        gene_names = sampled_adata.var_names.tolist()
        initial_cells = []
        
        for i, (cell_idx, cell_type) in enumerate(zip(sampled_indices, sampled_adata.obs['cell_type'])):
            expression = sampled_adata[i].X.toarray().flatten() if hasattr(sampled_adata[i].X, 'toarray') else sampled_adata[i].X.flatten()
            expression = np.maximum(0, expression)
            if expression.sum() > 0:
                expression = expression / expression.sum()
            
            cell = CellAgent(
                cell_id=i,
                cell_type=cell_type,
                initial_expression=expression,
                drugs=drugs
            )
            initial_cells.append(cell)
        
        print(f"✓ Created {len(initial_cells)} cell agents")
        
    else:
        # 备用方案：使用模拟数据
        print("\n✓ Using simulated data (fallback)")
        
        n_cells = 100
        n_genes = 200
        
        cell_types = ['CD4+ T', 'CD8+ T', 'B', 'Monocyte', 'NK']
        lr_genes = ['IL6', 'IL6R', 'IFNG', 'IFNGR1', 'TNF', 'TNFAIP3',
                    'CD40', 'CD40LG', 'PDCD1', 'CD274', 'CD28', 'CD80',
                    'CXCL12', 'CXCR4', 'CCL2', 'CCR2']
        other_genes = [f'Gene_{i}' for i in range(n_genes - len(lr_genes))]
        gene_names = lr_genes + other_genes
        
        drugs = data_loader.get_real_drugs()
        
        initial_cells = []
        for i in range(n_cells):
            cell_type = cell_types[i % len(cell_types)]
            expression = np.random.rand(n_genes)
            
            if cell_type == 'CD4+ T':
                for gene in ['IL2RA', 'CD4', 'CD3D', 'CD3E']:
                    if gene in gene_names:
                        expression[gene_names.index(gene)] += 0.3
            elif cell_type == 'CD8+ T':
                for gene in ['CD8A', 'CD8B', 'CD3D', 'CD3E']:
                    if gene in gene_names:
                        expression[gene_names.index(gene)] += 0.3
            elif cell_type == 'B':
                for gene in ['MS4A1', 'CD79A', 'CD79B', 'CD19']:
                    if gene in gene_names:
                        expression[gene_names.index(gene)] += 0.3
            elif cell_type == 'Monocyte':
                for gene in ['CD14', 'LYZ', 'ITGAM', 'CSF1R']:
                    if gene in gene_names:
                        expression[gene_names.index(gene)] += 0.3
            elif cell_type == 'NK':
                for gene in ['GNLY', 'NKG7', 'KLRD1', 'KLRB1']:
                    if gene in gene_names:
                        expression[gene_names.index(gene)] += 0.3
            
            expression = expression / expression.sum()
            
            cell = CellAgent(
                cell_id=i,
                cell_type=cell_type,
                initial_expression=expression,
                drugs=drugs
            )
            initial_cells.append(cell)
        
        # 创建目标互作矩阵
        target_cell_types = cell_types
        target_interaction = np.array([
            [0.0, 0.12, 0.10, 0.08, 0.06],
            [0.10, 0.0, 0.08, 0.06, 0.08],
            [0.08, 0.06, 0.0, 0.05, 0.04],
            [0.15, 0.10, 0.05, 0.0, 0.06],
            [0.06, 0.08, 0.04, 0.06, 0.0],
        ])
        target_interaction = target_interaction / target_interaction.sum()
    
    # 创建进化器
    evolver = CellSelfEvolver(
        initial_cells=initial_cells,
        target_interaction=target_interaction,
        drugs=drugs,
        gene_names=gene_names
    )
    
    # 运行进化
    history, final_interaction, cell_types = evolver.run_evolution(
        max_steps=100,
        tolerance=0.95
    )
    
    # 保存结果
    output_dir = '/tmp/cell_evolution_real_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存进化历史
    history_df = pd.DataFrame([{
        'step': h['step'],
        'similarity': h['similarity'],
        'best_similarity': h['best_similarity'],
        'avg_fitness': h['avg_fitness'],
        'avg_reward': h['avg_reward']
    } for h in history])
    history_df.to_csv(os.path.join(output_dir, 'evolution_history.csv'), index=False)
    
    # 保存最终互作矩阵
    final_df = pd.DataFrame(final_interaction, index=cell_types, columns=cell_types)
    final_df.to_csv(os.path.join(output_dir, 'final_interaction_matrix.csv'))
    
    # 保存目标互作矩阵
    target_df = pd.DataFrame(target_interaction, index=target_cell_types, columns=target_cell_types)
    target_df.to_csv(os.path.join(output_dir, 'target_interaction_matrix.csv'))
    
    # 保存进化后的细胞数据
    cell_data = []
    for cell in evolver.population.cells:
        cell_data.append({
            'cell_id': cell.cell_id,
            'cell_type': cell.cell_type,
            'fitness': cell.fitness,
            'best_fitness': cell.best_fitness,
            'expression_norm': np.linalg.norm(cell.expression)
        })
    cell_df = pd.DataFrame(cell_data)
    cell_df.to_csv(os.path.join(output_dir, 'final_cells.csv'), index=False)
    
    # 保存药物偏好
    drug_preferences = []
    for cell in evolver.population.cells:
        for drug, pref in cell.drug_preferences.items():
            drug_preferences.append({
                'cell_id': cell.cell_id,
                'cell_type': cell.cell_type,
                'drug': drug,
                'preference': pref
            })
    drug_df = pd.DataFrame(drug_preferences)
    drug_df.to_csv(os.path.join(output_dir, 'drug_preferences.csv'), index=False)
    
    print(f"\nResults saved to: {output_dir}")
    print("  - evolution_history.csv")
    print("  - final_interaction_matrix.csv")
    print("  - target_interaction_matrix.csv")
    print("  - final_cells.csv")
    print("  - drug_preferences.csv")

def load_dispatched_smile():
    """加载从smileasign派发的SMILE"""
    dispatch_file = '/tmp/current_smile.json'
    if os.path.exists(dispatch_file):
        try:
            with open(dispatch_file, 'r') as f:
                dispatch_info = json.load(f)
            
            print(f"\n✓ Loaded dispatched SMILE: {dispatch_info['name']}")
            print(f"  SMILE: {dispatch_info['smile']}")
            print(f"  Category: {dispatch_info['category']}")
            print(f"  Dispatch time: {dispatch_info['dispatch_time']}")
            
            # 更新状态为已处理
            dispatch_info['status'] = 'processed'
            with open(dispatch_file, 'w') as f:
                json.dump(dispatch_info, f, indent=2)
            
            return dispatch_info['smile'], dispatch_info['name'], dispatch_info['category']
        
        except Exception as e:
            print(f"Failed to load dispatched SMILE: {e}")
            return None, None, None
    return None, None, None

def main():
    print("=" * 70)
    print("CELL SELF-EVOLUTION WITH REAL DATA")
    print("=" * 70)
    
    # 检查是否有派发的SMILE
    dispatched_smile, dispatched_name, dispatched_category = load_dispatched_smile()
    
    # 加载真实数据
    data_loader = RealDataLoader()
    adata = data_loader.load_pbmc3k()
    
    if adata is not None:
        print("\n✓ Using REAL single-cell data")
        print(f"  Dataset: PBMC 3K")
        print(f"  Cells: {adata.n_obs}")
        print(f"  Genes: {adata.n_vars}")
        print(f"  Cell types: {adata.obs['cell_type'].nunique()}")
        print(adata.obs['cell_type'].value_counts())
        
        # 计算真实的互作矩阵作为目标
        print("\nComputing target interaction matrix from real data...")
        target_interaction, target_cell_types = data_loader.compute_real_interaction_matrix(adata)
        print(f"✓ Target interaction matrix shape: {target_interaction.shape}")
        print(f"  Cell types: {target_cell_types}")
        
        # 获取药物（包含派发的SMILE）
        drugs = data_loader.get_real_drugs()
        
        # 如果有派发的SMILE，优先使用
        if dispatched_smile and dispatched_smile not in drugs:
            drugs.append(dispatched_smile)
            print(f"\n✓ Added dispatched SMILE: {dispatched_name}")
        
        print(f"\n✓ Loaded {len(drugs)} drug SMILEs")
        
        # 从真实数据中采样细胞
        n_cells = 100
        print(f"\nSampling {n_cells} cells from real data...")
        
        # 平衡采样每种细胞类型
        sampled_indices = []
        for cell_type in target_cell_types:
            type_indices = np.where(adata.obs['cell_type'] == cell_type)[0]
            n_sample = min(len(type_indices), n_cells // len(target_cell_types))
            sampled_indices.extend(np.random.choice(type_indices, n_sample, replace=False))
        
        # 如果采样不足，随机补充
        if len(sampled_indices) < n_cells:
            remaining = n_cells - len(sampled_indices)
            all_indices = np.arange(adata.n_obs)
            remaining_indices = np.setdiff1d(all_indices, sampled_indices)
            sampled_indices.extend(np.random.choice(remaining_indices, remaining, replace=False))
        
        sampled_adata = adata[sampled_indices].copy()
        
        # 创建细胞智能体
        gene_names = sampled_adata.var_names.tolist()
        initial_cells = []
        
        for i, (cell_idx, cell_type) in enumerate(zip(sampled_indices, sampled_adata.obs['cell_type'])):
            expression = sampled_adata[i].X.toarray().flatten() if hasattr(sampled_adata[i].X, 'toarray') else sampled_adata[i].X.flatten()
            expression = np.maximum(0, expression)
            if expression.sum() > 0:
                expression = expression / expression.sum()
            
            cell = CellAgent(
                cell_id=i,
                cell_type=cell_type,
                initial_expression=expression,
                drugs=drugs
            )
            initial_cells.append(cell)
        
        print(f"✓ Created {len(initial_cells)} cell agents")
        
    else:
        # 备用方案：使用模拟数据
        print("\n✓ Using simulated data (fallback)")
        
        n_cells = 100
        n_genes = 200
        
        cell_types = ['CD4+ T', 'CD8+ T', 'B', 'Monocyte', 'NK']
        lr_genes = ['IL6', 'IL6R', 'IFNG', 'IFNGR1', 'TNF', 'TNFAIP3',
                    'CD40', 'CD40LG', 'PDCD1', 'CD274', 'CD28', 'CD80',
                    'CXCL12', 'CXCR4', 'CCL2', 'CCR2']
        other_genes = [f'Gene_{i}' for i in range(n_genes - len(lr_genes))]
        gene_names = lr_genes + other_genes
        
        drugs = data_loader.get_real_drugs()
        
        # 如果有派发的SMILE，优先使用
        if dispatched_smile and dispatched_smile not in drugs:
            drugs.append(dispatched_smile)
        
        initial_cells = []
        for i in range(n_cells):
            cell_type = cell_types[i % len(cell_types)]
            expression = np.random.rand(n_genes)
            
            if cell_type == 'CD4+ T':
                for gene in ['IL2RA', 'CD4', 'CD3D', 'CD3E']:
                    if gene in gene_names:
                        expression[gene_names.index(gene)] += 0.3
            elif cell_type == 'CD8+ T':
                for gene in ['CD8A', 'CD8B', 'CD3D', 'CD3E']:
                    if gene in gene_names:
                        expression[gene_names.index(gene)] += 0.3
            elif cell_type == 'B':
                for gene in ['MS4A1', 'CD79A', 'CD79B', 'CD19']:
                    if gene in gene_names:
                        expression[gene_names.index(gene)] += 0.3
            elif cell_type == 'Monocyte':
                for gene in ['CD14', 'LYZ', 'ITGAM', 'CSF1R']:
                    if gene in gene_names:
                        expression[gene_names.index(gene)] += 0.3
            elif cell_type == 'NK':
                for gene in ['GNLY', 'NKG7', 'KLRD1', 'KLRB1']:
                    if gene in gene_names:
                        expression[gene_names.index(gene)] += 0.3
            
            expression = expression / expression.sum()
            
            cell = CellAgent(
                cell_id=i,
                cell_type=cell_type,
                initial_expression=expression,
                drugs=drugs
            )
            initial_cells.append(cell)
        
        # 创建目标互作矩阵
        target_cell_types = cell_types
        target_interaction = np.array([
            [0.0, 0.12, 0.10, 0.08, 0.06],
            [0.10, 0.0, 0.08, 0.06, 0.08],
            [0.08, 0.06, 0.0, 0.05, 0.04],
            [0.15, 0.10, 0.05, 0.0, 0.06],
            [0.06, 0.08, 0.04, 0.06, 0.0],
        ])
        target_interaction = target_interaction / target_interaction.sum()
    
    # 创建进化器
    evolver = CellSelfEvolver(
        initial_cells=initial_cells,
        target_interaction=target_interaction,
        drugs=drugs,
        gene_names=gene_names
    )
    
    # 运行进化
    history, final_interaction, cell_types = evolver.run_evolution(
        max_steps=100,
        tolerance=0.95
    )
    
    # 保存结果
    output_dir = '/tmp/cell_evolution_real_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存进化历史
    history_df = pd.DataFrame([{
        'step': h['step'],
        'similarity': h['similarity'],
        'best_similarity': h['best_similarity'],
        'avg_fitness': h['avg_fitness'],
        'avg_reward': h['avg_reward']
    } for h in history])
    history_df.to_csv(os.path.join(output_dir, 'evolution_history.csv'), index=False)
    
    # 保存最终互作矩阵
    final_df = pd.DataFrame(final_interaction, index=cell_types, columns=cell_types)
    final_df.to_csv(os.path.join(output_dir, 'final_interaction_matrix.csv'))
    
    # 保存目标互作矩阵
    target_df = pd.DataFrame(target_interaction, index=target_cell_types, columns=target_cell_types)
    target_df.to_csv(os.path.join(output_dir, 'target_interaction_matrix.csv'))
    
    # 保存进化后的细胞数据
    cell_data = []
    for cell in evolver.population.cells:
        cell_data.append({
            'cell_id': cell.cell_id,
            'cell_type': cell.cell_type,
            'fitness': cell.fitness,
            'best_fitness': cell.best_fitness,
            'expression_norm': np.linalg.norm(cell.expression)
        })
    cell_df = pd.DataFrame(cell_data)
    cell_df.to_csv(os.path.join(output_dir, 'final_cells.csv'), index=False)
    
    # 保存药物偏好
    drug_preferences = []
    for cell in evolver.population.cells:
        for drug, pref in cell.drug_preferences.items():
            drug_preferences.append({
                'cell_id': cell.cell_id,
                'cell_type': cell.cell_type,
                'drug': drug,
                'preference': pref
            })
    drug_df = pd.DataFrame(drug_preferences)
    drug_df.to_csv(os.path.join(output_dir, 'drug_preferences.csv'), index=False)
    
    print(f"\nResults saved to: {output_dir}")
    print("  - evolution_history.csv")
    print("  - final_interaction_matrix.csv")
    print("  - target_interaction_matrix.csv")
    print("  - final_cells.csv")
    print("  - drug_preferences.csv")

if __name__ == '__main__':
    main()
