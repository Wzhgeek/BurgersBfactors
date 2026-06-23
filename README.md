# Pcode — Burgers 动力学 B-factor 预测

## 快速开始

```bash
conda create -n pcode python=3.11 && conda activate pcode
pip install -r requirements.txt

# Step1: 从 xyzb 生成 Aij 矩阵（首次运行必需）
python run_step1.py --dataset 33small

# 运行单个蛋白（烟雾测试）
python run.py --dataset 33small --protein 1Q9B --smoke

# 运行单个蛋白（完整扫描）
python run.py --dataset 33small --protein 1Q9B

# 跨蛋白汇总
python summarize.py --dataset 33small
```

## 目录结构

```
Pcode/
├── run.py               # 主流程：模拟 → 特征 → 回归 → 绘图
├── run_step1.py         # 预处理：xyzb/PDB → Aij/distance/binary
├── summarize.py         # 跨蛋白结果汇总
├── config.yaml          # 所有参数
├── requirements.txt
├── src/                 # 核心模块
├── slurm/               # SLURM 提交脚本
├── code_data/           # xyzb 输入数据（按数据集分目录）
├── protein/             # PDB 文件（可选）
└── result/              # 输出（按数据集/蛋白分目录）
```

## 新数据集

起点是 **xyzb 文件**（每行 `x y z bfactor`）。放到 `code_data/<dataset>/` 下：

```bash
mkdir -p code_data/mydata
cp /path/to/*_ca.xyzb code_data/mydata/

# 生成 Aij
python run_step1.py --dataset mydata

# 运行
python run.py --dataset mydata --protein XXXX
```

也可从 PDB 出发：
```bash
python run_step1.py --dataset mydata --from-pdb
```

输出自动保存到 `result/<dataset>/`。

## SLURM 部署

```bash
# 单个蛋白
sbatch slurm/submit_protein.sh 33small 1Q9B

# 批量全部
bash slurm/submit_all.sh
```

## 参数

编辑 `config.yaml` 修改：ν、ε 范围、回归器、并行核数、特征类型等。
