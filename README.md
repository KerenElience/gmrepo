# GMRepo肠道菌群疾病鉴别

## 介绍
基于深度学习，利用gmrepo公共数据库建立肠道菌群相对丰度与相关疾病之间的模型

## 使用

### 下载与环境配置
```shell
git clone https://www.github.com/KerenElience/GutBacteria_RiskDiagnosis.git
pip install requirement.txt
python main.py
```


## 流程
提取中国地区，实验类型为Metagenomics的样本对应的菌群丰度以及phenotype

### 数据清洗
- 过滤低丰度特征，过滤低样本量疾病
- CLR变换

### 基于级联式模型进行疾病分类
- SMOTE过采样，确保样本量均衡
- SA算法/BeamSearch计算最优疾病组合模型
- 贪心策略解决难以区分疾病
- Optuna对各个子模型进行调优
- 串联所有子模型

### 基于菌属丰度排名顺序的Transformer疾病分类模型
- 对[MGM](https://github.com/HUST-NingKang-Lab/MGM)项目模型进行迁移学习


