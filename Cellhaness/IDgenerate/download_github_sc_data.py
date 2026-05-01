import os
import sys
import numpy as np
import pandas as pd
import requests
import tarfile
import warnings
warnings.filterwarnings('ignore')

def download_from_github():
    """从GitHub下载公开的单细胞数据集"""
    print("=" * 70)
    print("DOWNLOADING REAL SINGLE-CELL DATA FROM GITHUB")
    print("=" * 70)
    
    output_dir = '/tmp/real_sc_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # 尝试下载多个公开数据集
    datasets = [
        {
            'name': 'pbmc3k',
            'url': 'https://raw.githubusercontent.com/theislab/scanpy_usage/master/180209_celltypist/data/pbmc3k_raw.h5ad',
            'description': 'PBMC 3K dataset (human immune cells)'
        },
        {
            'name': 'brain_cells',
            'url': 'https://github.com/chanzuckerberg/DIP-seq/raw/master/data/brain_cells.h5ad',
            'description': 'Brain cells dataset'
        },
        {
            'name': 'mouse_brain',
            'url': 'https://github.com/broadinstitute/SingleCellPortal/raw/master/datasets/mouse_brain/mouse_brain.h5ad',
            'description': 'Mouse brain dataset'
        }
    ]
    
    for dataset in datasets:
        print(f"\nAttempting to download: {dataset['name']}")
        print(f"  Description: {dataset['description']}")
        print(f"  URL: {dataset['url']}")
        
        try:
            response = requests.get(dataset['url'], stream=True, timeout=60)
            
            if response.status_code == 200:
                file_path = os.path.join(output_dir, f"{dataset['name']}.h5ad")
                
                print(f"  Downloading...")
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"  Downloaded to {file_path}")
                
                # 尝试读取数据
                try:
                    import scanpy as sc
                    adata = sc.read_h5ad(file_path)
                    print(f"  Successfully read: {adata.n_obs} cells, {adata.n_vars} genes")
                    
                    # 保存为npy格式以便后续处理
                    np.save(os.path.join(output_dir, f"{dataset['name']}_expression.npy"), adata.X)
                    pd.DataFrame(adata.var_names).to_csv(os.path.join(output_dir, f"{dataset['name']}_genes.csv"), index=False, header=['gene'])
                    pd.DataFrame(adata.obs_names).to_csv(os.path.join(output_dir, f"{dataset['name']}_samples.csv"), index=False, header=['sample'])
                    
                    expr_file = os.path.join(output_dir, f"{dataset['name']}_expression.npy")
                    print(f"  Saved expression matrix: {expr_file}")
                    
                    return adata
                    
                except ImportError:
                    print("  Scanpy not available, skipping data preview")
                    return None
                except Exception as e:
                    print(f"  Error reading data: {e}")
                    return None
                    
            else:
                print(f"  Failed (status code: {response.status_code})")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    
    # 检查下载的文件
    files = os.listdir(output_dir)
    print(f"\nFiles in {output_dir}:")
    for f in files:
        print(f"  - {f}")
    
    return output_dir

def main():
    output_dir = download_from_github()
    
    # 检查是否有成功下载的数据
    if 'pbmc3k_expression.npy' in os.listdir(output_dir):
        print("\n" + "=" * 70)
        print("REAL DATA DOWNLOADED SUCCESSFULLY!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Run cell communication analysis")
        print("2. Generate SMILE using LRLDM")
    else:
        print("\n" + "=" * 70)
        print("Download failed")
        print("=" * 70)

if __name__ == '__main__':
    main()
