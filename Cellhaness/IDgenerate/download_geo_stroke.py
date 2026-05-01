import os
import sys
import numpy as np
import pandas as pd
import GEOparse

class GEOSingleCellDownloader:
    """使用GEOparse下载真实单细胞数据"""
    
    def __init__(self):
        self.data_dir = '/tmp/geo_sc_data'
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_gse138852(self):
        """下载脑卒中单细胞数据集GSE138852"""
        print("Downloading GSE138852 - Stroke single-cell RNA-seq...")
        
        try:
            gse = GEOparse.get_GEO(geo='GSE138852', destdir=self.data_dir)
            
            # 提取表达数据
            all_samples = []
            sample_names = []
            
            for gsm_name, gsm in gse.gsms.items():
                if hasattr(gsm, 'table') and gsm.table is not None:
                    print(f"Processing sample: {gsm_name}")
                    df = gsm.table
                    
                    if 'ID_REF' in df.columns and 'VALUE' in df.columns:
                        df = df[['ID_REF', 'VALUE']].set_index('ID_REF')
                        all_samples.append(df)
                        sample_names.append(gsm_name)
            
            if len(all_samples) > 0:
                print(f"\nMerging {len(all_samples)} samples...")
                merged_df = pd.concat(all_samples, axis=1, join='inner')
                merged_df.columns = sample_names
                
                # 转置：行为细胞，列为基因
                expression_matrix = merged_df.values.T
                gene_names = merged_df.index.tolist()
                
                print(f"Success! Expression matrix shape: {expression_matrix.shape}")
                print(f"Number of cells: {expression_matrix.shape[0]}")
                print(f"Number of genes: {expression_matrix.shape[1]}")
                
                # 保存数据
                np.save(os.path.join(self.data_dir, 'gse138852_expression.npy'), expression_matrix)
                pd.DataFrame(gene_names).to_csv(os.path.join(self.data_dir, 'gse138852_genes.csv'), index=False, header=['gene'])
                pd.DataFrame(sample_names).to_csv(os.path.join(self.data_dir, 'gse138852_samples.csv'), index=False, header=['sample'])
                
                return expression_matrix, gene_names, sample_names
            
            else:
                print("No valid expression data found in GSE138852")
                return None, None, None
                
        except Exception as e:
            print(f"Error downloading GSE138852: {e}")
            return None, None, None
    
    def download_gse157278(self):
        """下载正常脑组织数据集GSE157278"""
        print("\nDownloading GSE157278 - Normal brain RNA-seq...")
        
        try:
            gse = GEOparse.get_GEO(geo='GSE157278', destdir=self.data_dir)
            
            all_samples = []
            sample_names = []
            
            for gsm_name, gsm in gse.gsms.items():
                if hasattr(gsm, 'table') and gsm.table is not None:
                    print(f"Processing sample: {gsm_name}")
                    df = gsm.table
                    
                    if 'ID_REF' in df.columns and 'VALUE' in df.columns:
                        df = df[['ID_REF', 'VALUE']].set_index('ID_REF')
                        all_samples.append(df)
                        sample_names.append(gsm_name)
            
            if len(all_samples) > 0:
                merged_df = pd.concat(all_samples, axis=1, join='inner')
                merged_df.columns = sample_names
                
                expression_matrix = merged_df.values.T
                gene_names = merged_df.index.tolist()
                
                print(f"Success! Expression matrix shape: {expression_matrix.shape}")
                
                np.save(os.path.join(self.data_dir, 'gse157278_expression.npy'), expression_matrix)
                pd.DataFrame(gene_names).to_csv(os.path.join(self.data_dir, 'gse157278_genes.csv'), index=False, header=['gene'])
                
                return expression_matrix, gene_names, sample_names
            
            else:
                print("No valid expression data found in GSE157278")
                return None, None, None
                
        except Exception as e:
            print(f"Error downloading GSE157278: {e}")
            return None, None, None

def main():
    print("=" * 70)
    print("Downloading REAL Stroke vs Normal Brain Data from GEO")
    print("=" * 70)
    
    downloader = GEOSingleCellDownloader()
    
    # 下载脑卒中数据
    stroke_data, stroke_genes, stroke_samples = downloader.download_gse138852()
    
    # 下载正常脑数据
    normal_data, normal_genes, normal_samples = downloader.download_gse157278()
    
    # 检查结果
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    
    if stroke_data is not None:
        print(f"✓ Stroke data (GSE138852): {stroke_data.shape[0]} cells × {stroke_data.shape[1]} genes")
        print(f"  Saved to: {os.path.join('/tmp/geo_sc_data', 'gse138852_expression.npy')}")
    else:
        print("✗ Failed to download stroke data")
    
    if normal_data is not None:
        print(f"✓ Normal brain data (GSE157278): {normal_data.shape[0]} cells × {normal_data.shape[1]} genes")
        print(f"  Saved to: {os.path.join('/tmp/geo_sc_data', 'gse157278_expression.npy')}")
    else:
        print("✗ Failed to download normal brain data")
    
    if stroke_data is not None or normal_data is not None:
        print("\nData ready for analysis!")
    else:
        print("\nCould not download real data. Please try manual download.")
        print("\nManual download instructions:")
        print("1. Go to https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE138852")
        print("2. Download the supplementary files")
        print("3. Extract and place in /tmp/geo_sc_data/")

if __name__ == '__main__':
    main()
