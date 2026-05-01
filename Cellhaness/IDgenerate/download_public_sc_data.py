import os
import sys
import numpy as np
import pandas as pd
import requests
import tarfile
import gzip
import shutil

class PublicSingleCellDownloader:
    """从公开数据源下载真实单细胞RNA-seq数据"""
    
    def __init__(self):
        self.data_dir = '/tmp/real_sc_data'
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_from_broad(self):
        """从Broad Institute Single Cell Portal下载数据"""
        print("Downloading from Broad Single Cell Portal...")
        
        # 脑卒中相关数据集
        datasets = [
            {
                'name': 'stroke_brain',
                'url': 'https://singlecell.broadinstitute.org/single_cell/study/SCP1000/human-stroke-brain',
                'download_url': 'https://storage.googleapis.com/scp-broad-public/SCP1000/SCP1000_human_stroke_brain.tar.gz'
            },
            {
                'name': 'normal_brain', 
                'url': 'https://singlecell.broadinstitute.org/single_cell/study/SCP384/human-adult-brain',
                'download_url': 'https://storage.googleapis.com/scp-broad-public/SCP384/SCP384_human_adult_brain.tar.gz'
            }
        ]
        
        results = {}
        
        for dataset in datasets:
            try:
                print(f"\nDownloading {dataset['name']}...")
                response = requests.get(dataset['download_url'], stream=True)
                
                if response.status_code == 200:
                    file_path = os.path.join(self.data_dir, f"{dataset['name']}.tar.gz")
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    print(f"Downloaded to {file_path}")
                    
                    # 解压
                    with tarfile.open(file_path, 'r:gz') as tar:
                        tar.extractall(os.path.join(self.data_dir, dataset['name']))
                    print(f"Extracted to {os.path.join(self.data_dir, dataset['name'])}")
                    
                    results[dataset['name']] = os.path.join(self.data_dir, dataset['name'])
                else:
                    print(f"Failed to download {dataset['name']} (status: {response.status_code})")
                
            except Exception as e:
                print(f"Error downloading {dataset['name']}: {e}")
        
        return results
    
    def download_from_figshare(self):
        """从Figshare下载数据"""
        print("\nDownloading from Figshare...")
        
        datasets = [
            {
                'name': 'stroke_sc',
                'url': 'https://figshare.com/articles/dataset/Single-cell_RNA-seq_of_human_stroke_brain/14317762',
                'download_url': 'https://figshare.com/ndownloader/files/26266142'
            }
        ]
        
        results = {}
        
        for dataset in datasets:
            try:
                print(f"Downloading {dataset['name']}...")
                response = requests.get(dataset['download_url'], stream=True)
                
                if response.status_code == 200:
                    file_path = os.path.join(self.data_dir, f"{dataset['name']}.zip")
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Downloaded to {file_path}")
                    results[dataset['name']] = file_path
                else:
                    print(f"Failed to download {dataset['name']}")
                    
            except Exception as e:
                print(f"Error downloading {dataset['name']}: {e}")
        
        return results
    
    def download_from_zenodo(self):
        """从Zenodo下载数据"""
        print("\nDownloading from Zenodo...")
        
        datasets = [
            {
                'name': 'mouse_brain',
                'url': 'https://zenodo.org/record/5123934',
                'download_url': 'https://zenodo.org/api/files/5123934/brain.h5ad'
            }
        ]
        
        results = {}
        
        for dataset in datasets:
            try:
                print(f"Downloading {dataset['name']}...")
                response = requests.get(dataset['download_url'], stream=True)
                
                if response.status_code == 200:
                    file_path = os.path.join(self.data_dir, f"{dataset['name']}.h5ad")
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Downloaded to {file_path}")
                    results[dataset['name']] = file_path
                else:
                    print(f"Failed to download {dataset['name']}")
                    
            except Exception as e:
                print(f"Error downloading {dataset['name']}: {e}")
        
        return results
    
    def download_example_datasets(self):
        """下载示例数据集"""
        print("\nDownloading example datasets from known sources...")
        
        # 下载一个已知可用的单细胞数据集
        url = "https://raw.githubusercontent.com/theislab/scanpy_usage/master/180209_celltypist/data/pbmc3k_raw.h5ad"
        
        try:
            print("Downloading PBMC dataset...")
            response = requests.get(url, stream=True)
            
            if response.status_code == 200:
                file_path = os.path.join(self.data_dir, "pbmc3k.h5ad")
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded to {file_path}")
                return file_path
            else:
                print(f"Failed to download PBMC dataset")
                return None
                
        except Exception as e:
            print(f"Error downloading: {e}")
            return None

def main():
    print("=" * 70)
    print("Downloading REAL Single-Cell Data from Public Sources")
    print("=" * 70)
    
    downloader = PublicSingleCellDownloader()
    
    # 尝试多种下载源
    broad_results = downloader.download_from_broad()
    figshare_results = downloader.download_from_figshare()
    zenodo_results = downloader.download_from_zenodo()
    example_data = downloader.download_example_datasets()
    
    # 检查下载结果
    all_results = {**broad_results, **figshare_results, **zenodo_results}
    
    if example_data:
        all_results['example_pbmc'] = example_data
    
    if all_results:
        print("\n" + "=" * 70)
        print("Download Summary:")
        print("=" * 70)
        
        for name, path in all_results.items():
            print(f"{name}: {path}")
        
        print("\nTotal downloaded: {} datasets".format(len(all_results)))
        
        # 尝试读取并显示数据信息
        try:
            import scanpy as sc
            
            for name, path in all_results.items():
                if path.endswith('.h5ad'):
                    try:
                        adata = sc.read_h5ad(path)
                        print(f"\n{name}: {adata.n_obs} cells, {adata.n_vars} genes")
                    except:
                        pass
                        
        except ImportError:
            print("\nScanpy not installed, skipping data preview")
            
    else:
        print("\n" + "=" * 70)
        print("No datasets downloaded successfully")
        print("=" * 70)
        print("\nPlease consider:")
        print("1. Downloading data manually from GEO: https://www.ncbi.nlm.nih.gov/geo/")
        print("2. Using R with GEOquery package")
        print("3. Uploading your own h5ad/mtx files")

if __name__ == '__main__':
    main()
