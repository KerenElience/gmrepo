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
数据清洗
提取中国地区，实验类型为Metagenomics的样本对应的菌群丰度以及phenotype

标签编码
RandomForst + xgboost -- baseline
MLP -- base model
Transformer提升 -- last model
