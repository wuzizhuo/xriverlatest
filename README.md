# XRiverLatest - 细胞自进化模拟系统

## 项目简介

XRiverLatest是一个基于真实单细胞数据的细胞自进化模拟系统，实现了细胞作为智能体通过药物SMILE进行自我更新的过程。

## 核心功能

### 1. 细胞自进化模拟 (cellselfevolve.py)
- 基于真实PBMC 3K单细胞数据
- 使用CellChatDB人类受配体数据库
- 集成DrugBank真实药物SMILE
- 实现GSVA-LDM和LR-LDM模型

### 2. SMILE自动派发系统 (smileasign.py)
- 每6小时自动派发药物SMILE
- 支持多种派发策略（随机、轮询）
- 可配置药物库和派发规则

## 安装

```bash
pip install numpy pandas scanpy schedule
```

## 使用方法

### 启动SMILE派发服务
```bash
python Cellhaness/IDgenerate/smileasign.py start
```

### 运行细胞自进化
```bash
python Cellhaness/IDgenerate/cellselfevolve.py
```

## 配置文件

配置文件位于 `Cellhaness/IDgenerate/cellselfevolve.json`，包含：
- 药物库（抗炎药、免疫调节剂等）
- 派发规则
- 进化参数

## 数据来源

- 单细胞数据：PBMC 3K (Scanpy)
- 受配体数据库：CellChatDB
- 药物SMILE：DrugBank

## 作者

wuzizhuo
