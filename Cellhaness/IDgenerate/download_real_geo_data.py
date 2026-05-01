import os
import sys
import numpy as np
import pandas as pd
import requests
import tarfile
import gzip
import shutil

class TrueGEOSingleCellDownloader:
    """从GEO下载真实的单细胞RNA-seq数据"""
    
    def __init__(self):
        self.data_dir = '/tmp/geo_true_data'
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_gse_supplementary(self, gse_id):
        """下载GEO补充文件"""
        print(f"Attempting to download supplementary files for {gse_id}...")
        
        # GEO补充文件URL格式
        base_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_id[:-3]}nnn/{gse_id}/suppl/"
        
        try:
            # 列出目录内容
            response = requests.get(base_url)
            if response.status_code != 200:
                print(f"Could not access supplementary files for {gse_id}")
                return None
            
            # 查找可能的文件
            content = response.text
            potential_files = []
            
            for line in content.split('\n'):
                if '.tar.gz' in line or '.h5ad' in line or '.mtx' in line or '.csv' in line:
                    for part in line.split():
                        if part.endswith(('.tar.gz', '.h5ad', '.mtx.gz', '.csv.gz')):
                            potential_files.append(part)
            
            if not potential_files:
                print(f"No suitable files found for {gse_id}")
                return None
            
            # 下载第一个合适的文件
            file_to_download = potential_files[0]
            file_url = base_url + file_to_download
            local_path = os.path.join(self.data_dir, file_to_download)
            
            print(f"Downloading {file_url}...")
            response = requests.get(file_url, stream=True)
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded to {local_path}")
            
            # 解压文件
            if local_path.endswith('.tar.gz'):
                with tarfile.open(local_path, 'r:gz') as tar:
                    tar.extractall(self.data_dir)
                print(f"Extracted tar.gz file")
            
            elif local_path.endswith('.gz'):
                with gzip.open(local_path, 'rb') as f_in:
                    out_path = local_path[:-3]
                    with open(out_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"Extracted gz file to {out_path}")
            
            return self.data_dir
            
        except Exception as e:
            print(f"Error downloading supplementary files: {e}")
            return None
    
    def download_from_cellxgene(self, dataset_id):
        """从Cellxgene下载数据集"""
        print(f"Attempting to download from Cellxgene: {dataset_id}")
        
        try:
            import anndata
            import scanpy as sc
            
            # 使用cellxgene_census或直接下载
            # 这里我们尝试下载公开可用的数据集
            url = f"https://api.cellxgene.cziscience.com/dp/v1/dataset/{dataset_id}/download"
            
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                file_path = os.path.join(self.data_dir, f"{dataset_id}.h5ad")
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"Downloaded to {file_path}")
                
                # 读取数据
                adata = sc.read_h5ad(file_path)
                print(f"Successfully loaded dataset: {adata.n_obs} cells, {adata.n_vars} genes")
                
                return adata
            
        except Exception as e:
            print(f"Error downloading from Cellxgene: {e}")
            return None
    
    def download_stroke_dataset(self):
        """下载脑卒中单细胞数据集"""
        print("\n=== Downloading Stroke Single-Cell Dataset ===")
        
        # 尝试多个脑卒中相关数据集
        stroke_datasets = [
            'GSE199075',  # Stroke single-cell
            'GSE166388',  # Stroke RNA-seq
            'GSE184554',  # Brain stroke
        ]
        
        for gse_id in stroke_datasets:
            result = self.download_gse_supplementary(gse_id)
            if result:
                # 尝试读取数据
                adata = self._try_read_data(result, gse_id)
                if adata is not None:
                    print(f"Successfully loaded stroke dataset: {gse_id}")
                    return adata
        
        # 如果都失败，尝试从cellxgene下载
        cellxgene_id = "b0e51f42-1e95-4d85-b635-5a63d4802f44"  # 可能的脑数据集
        adata = self.download_from_cellxgene(cellxgene_id)
        if adata is not None:
            return adata
        
        raise ValueError("Could not download real stroke data. Please provide a dataset file.")
    
    def download_normal_brain_dataset(self):
        """下载正常脑组织数据集"""
        print("\n=== Downloading Normal Brain Dataset ===")
        
        normal_datasets = [
            'GSE190604',  # Normal human brain
            'GSE157278',  # Brain tissue
            'GSE183838',  # Human brain cells
        ]
        
        for gse_id in normal_datasets:
            result = self.download_gse_supplementary(gse_id)
            if result:
                adata = self._try_read_data(result, gse_id)
                if adata is not None:
                    print(f"Successfully loaded normal brain dataset: {gse_id}")
                    return adata
        
        # 尝试cellxgene
        cellxgene_id = "ef403303-9528-411d-9d10-284fd774956a"
        adata = self.download_from_cellxgene(cellxgene_id)
        if adata is not None:
            return adata
        
        raise ValueError("Could not download real normal brain data. Please provide a dataset file.")
    
    def _try_read_data(self, data_dir, gse_id):
        """尝试读取下载的数据"""
        try:
            import anndata
            import scanpy as sc
            
            # 查找可能的数据文件
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith('.h5ad'):
                        return sc.read_h5ad(os.path.join(root, file))
                    elif file.endswith('.mtx'):
                        # 尝试读取mtx格式
                        try:
                            mtx_path = os.path.join(root, file)
                            adata = sc.read_mtx(mtx_path)
                            # 尝试查找基因名和细胞名
                            for f in files:
                                if 'genes' in f.lower() or 'features' in f.lower():
                                    genes = pd.read_csv(os.path.join(root, f), header=None)
                                    adata.var_names = genes[0].tolist()
                                if 'cells' in f.lower() or 'barcodes' in f.lower():
                                    cells = pd.read_csv(os.path.join(root, f), header=None)
                                    adata.obs_names = cells[0].tolist()
                            return adata
                        except:
                            continue
                    elif file.endswith('.csv'):
                        df = pd.read_csv(os.path.join(root, file), index_col=0)
                        return anndata.AnnData(X=df.values.T, var=pd.DataFrame(index=df.index))
            
        except Exception as e:
            print(f"Error reading data: {e}")
            return None
        
        return None

def main():
    print("=" * 70)
    print("Downloading REAL Single-Cell Data from GEO")
    print("=" * 70)
    
    downloader = TrueGEOSingleCellDownloader()
    
    try:
        # 下载脑卒中数据
        stroke_adata = downloader.download_stroke_dataset()
        print(f"\nStroke dataset loaded: {stroke_adata.n_obs} cells, {stroke_adata.n_vars} genes")
        
        # 下载正常脑数据
        normal_adata = downloader.download_normal_brain_dataset()
        print(f"\nNormal brain dataset loaded: {normal_adata.n_obs} cells, {normal_adata.n_vars} genes")
        
        # 保存数据
        stroke_adata.write_h5ad('/tmp/geo_true_data/stroke_data.h5ad')
        normal_adata.write_h5ad('/tmp/geo_true_data/normal_brain_data.h5ad')
        
        print("\n" + "=" * 70)
        print("Successfully downloaded REAL data!")
        print("=" * 70)
        print(f"Stroke data saved to: /tmp/geo_true_data/stroke_data.h5ad")
        print(f"Normal brain data saved to: /tmp/geo_true_data/normal_brain_data.h5ad")
        
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nAlternative options:")
        print("1. Provide your own h5ad/mtx files to /tmp/geo_true_data/")
        print("2. Use R with GEOquery to download and convert data")
        print("3. Download manually from https://www.ncbi.nlm.nih.gov/geo/")

if __name__ == '__main__':
    main()
