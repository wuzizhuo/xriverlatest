import os
import sys
import numpy as np
import pandas as pd
import GEOparse
import warnings
warnings.filterwarnings('ignore')

def read_geo_soft_file(gse_id):
    """读取GEO soft文件并提取表达数据"""
    print(f"Reading {gse_id}...")
    
    try:
        gse = GEOparse.get_GEO(geo=gse_id, destdir='/tmp/auto_sc_data')
        
        # 检查样本
        print(f"\nTotal samples: {len(gse.gsms)}")
        
        # 尝试读取每个样本
        for gsm_name, gsm in list(gse.gsms.items())[:5]:
            print(f"\nSample: {gsm_name}")
            
            if hasattr(gsm, 'table'):
                table = gsm.table
                print(f"  Table columns: {table.columns.tolist()}")
                print(f"  Table shape: {table.shape}")
                
                if not table.empty:
                    print(f"  First few rows:")
                    print(table.head())
            else:
                print("  No table found")
        
        # 尝试获取表达矩阵
        print("\nAttempting to extract expression matrix...")
        
        all_samples = []
        sample_names = []
        
        for gsm_name, gsm in gse.gsms.items():
            if hasattr(gsm, 'table') and gsm.table is not None:
                table = gsm.table
                
                # 尝试不同的列名组合
                if 'ID_REF' in table.columns and 'VALUE' in table.columns:
                    expr_df = table[['ID_REF', 'VALUE']].set_index('ID_REF')
                    all_samples.append(expr_df)
                    sample_names.append(gsm_name)
                elif 'Gene Symbol' in table.columns:
                    expr_df = table.set_index('Gene Symbol')
                    all_samples.append(expr_df)
                    sample_names.append(gsm_name)
                elif len(table.columns) >= 2:
                    # 使用前两列
                    expr_df = table.iloc[:, [0, 1]]
                    expr_df.columns = ['gene', 'value']
                    expr_df = expr_df.set_index('gene')
                    all_samples.append(expr_df)
                    sample_names.append(gsm_name)
        
        if len(all_samples) > 0:
            print(f"\nMerging {len(all_samples)} samples...")
            merged_df = pd.concat(all_samples, axis=1, join='inner')
            merged_df.columns = sample_names
            
            expression_matrix = merged_df.values.T
            genes = merged_df.index.tolist()
            
            print(f"\n✓ Success!")
            print(f"  Expression matrix: {expression_matrix.shape}")
            print(f"  Cells: {expression_matrix.shape[0]}")
            print(f"  Genes: {expression_matrix.shape[1]}")
            
            # 保存
            output_dir = '/tmp/real_sc_data'
            os.makedirs(output_dir, exist_ok=True)
            
            np.save(os.path.join(output_dir, f'{gse_id}_expression.npy'), expression_matrix)
            pd.DataFrame(genes).to_csv(os.path.join(output_dir, f'{gse_id}_genes.csv'), index=False, header=['gene'])
            pd.DataFrame(sample_names).to_csv(os.path.join(output_dir, f'{gse_id}_samples.csv'), index=False, header=['sample'])
            
            return expression_matrix, genes, sample_names
        
        print("\n✗ Could not extract expression data")
        return None, None, None
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def main():
    print("=" * 70)
    print("EXTRACTING EXPRESSION DATA FROM GEO SOFT FILES")
    print("=" * 70)
    
    # 尝试读取已下载的数据集
    datasets_to_try = [
        'GSE134809',  # Brain data
        'GSE131928',  # Brain
        'GSE135927',  # Brain
    ]
    
    for gse_id in datasets_to_try:
        print(f"\n{'=' * 70}")
        expr, genes, samples = read_geo_soft_file(gse_id)
        
        if expr is not None:
            continue
        
        print(f"\n{'=' * 70}")
        print("SUCCESS!")
        print(f"{'=' * 70}")
        print(f"\nData saved to /tmp/real_sc_data/")
        print(f"  {gse_id}_expression.npy")
        print(f"  {gse_id}_genes.csv")
        print(f"  {gse_id}_samples.csv")
        
        # 如果成功，返回
        return expr, genes, samples, gse_id
    
    print("\n" + "=" * 70)
    print("Could not extract expression data from any dataset")
    print("=" * 70)

if __name__ == '__main__':
    main()
