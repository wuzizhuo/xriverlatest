import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def download_scanpy_datasets():
    """使用Scanpy内置功能下载真实数据集"""
    print("=" * 70)
    print("DOWNLOADING REAL DATA USING SCANPY")
    print("=" * 70)
    
    output_dir = '/tmp/real_sc_data'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        import scanpy as sc
        print("\nScanpy version:", sc.__version__)
        
        # 尝试下载多个内置数据集
        datasets = [
            {
                'name': 'pbmc3k',
                'description': 'PBMC 3K (2700 cells, human immune)'
            },
            {
                'name': 'pbmc68k',
                'description': 'PBMC 68K (large immune dataset)'
            },
            {
                'name': 'b10k',
                'description': 'Brain 10K (mouse brain cells)'
            },
            {
                'name': 'mouse_brain',
                'description': 'Mouse brain cells'
            }
        ]
        
        for dataset in datasets:
            print(f"\n{'=' * 70}")
            print(f"Attempting: {dataset['name']}")
            print(f"Description: {dataset['description']}")
            print(f"{'=' * 70}")
            
            try:
                # 尝试不同的下载方法
                if dataset['name'] == 'pbmc3k':
                    print("Downloading pbmc3k...")
                    adata = sc.datasets.pbmc3k()
                elif dataset['name'] == 'pbmc68k':
                    print("Downloading pbmc68k...")
                    adata = sc.datasets.pbmc68k_reduced()
                elif dataset['name'] == 'b10k':
                    print("Downloading b10k...")
                    adata = sc.datasets.b10k()
                elif dataset['name'] == 'mouse_brain':
                    print("Downloading mouse_brain...")
                    adata = sc.datasets.mouse_brain()
                else:
                    print(f"Unknown dataset: {dataset['name']}")
                    continue
                
                print(f"✓ Successfully downloaded!")
                print(f"  Cells: {adata.n_obs}")
                print(f"  Genes: {adata.n_vars}")
                print(f"  Shape: {adata.shape}")
                
                # 保存数据
                adata.write_h5ad(os.path.join(output_dir, f"{dataset['name']}.h5ad"))
                
                # 保存表达矩阵
                if hasattr(adata.X, 'toarray'):
                    expr = adata.X.toarray()
                else:
                    expr = adata.X
                
                np.save(os.path.join(output_dir, f"{dataset['name']}_expression.npy"), expr)
                pd.DataFrame(adata.var_names).to_csv(os.path.join(output_dir, f"{dataset['name']}_genes.csv"), index=False, header=['gene'])
                pd.DataFrame(adata.obs_names).to_csv(os.path.join(output_dir, f"{dataset['name']}_samples.csv"), index=False, header=['sample'])
                
                print(f"✓ Saved to: {output_dir}")
                print(f"  - {dataset['name']}.h5ad")
                print(f"  - {dataset['name']}_expression.npy")
                print(f"  - {dataset['name']}_genes.csv")
                print(f"  - {dataset['name']}_samples.csv")
                
                # 返回第一个成功的数据集
                return adata, dataset['name']
                
            except Exception as e:
                print(f"✗ Failed: {e}")
                continue
        
        print("\n" + "=" * 70)
        print("Could not download any dataset")
        print("=" * 70)
        return None, None
        
    except ImportError:
        print("✗ Scanpy not installed")
        print("Please install: pip install scanpy")
        return None, None
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    adata, dataset_name = download_scanpy_datasets()
    
    if adata is not None:
        print("\n" + "=" * 70)
        print("✓ REAL DATA DOWNLOADED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nDataset: {dataset_name}")
        print(f"Cells: {adata.n_obs}")
        print(f"Genes: {adata.n_vars}")
        print("\nData saved to: /tmp/real_sc_data/")
        print("\nNext steps:")
        print("1. Run cell communication analysis")
        print("2. Generate SMILE using LRLDM")
    else:
        print("\n" + "=" * 70)
        print("✗ Download failed")
        print("=" * 70)

if __name__ == '__main__':
    main()
