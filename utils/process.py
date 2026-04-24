import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import gmean
from scipy.spatial.distance import pdist, squareform
from typing import Literal, Union, Mapping

class DataProcess():
    def __init__(self, meta: str):
        header = ["run_id", "disease", "ncbi_taxon_id", "relative_abundance", "phenotype"]
        self.meta = pd.read_csv(meta, usecols=header)
        self.meta = self.meta[(self.meta['ncbi_taxon_id'] != -1) & (~self.meta["disease"].isin(["D000086382"]))] ##trip bacteria and COV19

        self.encoder = LabelEncoder()
        self.label = None
    
    def process(self, clean_process: Mapping = {"disease":30, "relative_abundance": 80, "disease": 70}):
        # trip rare phenotype/disease
        df = self.meta.copy()
        if clean_process is not None:
            for k, v in clean_process.items():
                df = self.cleaning(df, k, v)

        data = df.pivot_table(index = "run_id",
                              columns="ncbi_taxon_id",
                              values = "relative_abundance",
                              aggfunc="sum",
                              sort=False,
                              fill_value=0)
        if self.label is None:
            self.label = df.drop_duplicates("run_id").set_index("run_id")["disease"]
        print(f"样本数: {len(data)}, 疾病数: {len(self.label.unique())}")
        return data

    def cleaning(self, df: pd.DataFrame, by: Literal["disease", "relative_abundance"] = "disease", threshold: Union[int, float] = 30):
        if by == "disease":
            class_counts = df.drop_duplicates("run_id").value_counts("disease")
            valid_classes = class_counts[class_counts >= threshold].index
            df = df[df['disease'].isin(valid_classes)].reset_index(drop=True)
        elif by == "relative_abundance":
            total_abundance = df.groupby("run_id").agg({"relative_abundance": "sum"})
            valid_abundance = total_abundance[total_abundance >= threshold].dropna().index
            df = df[df["run_id"].isin(valid_abundance) ].reset_index(drop=True)
        else:
            pass
        return df

    def filtration(self, df: pd.DataFrame, prevalence: float = 0.1, threshold: float = 1e-3):
        initial_features = df.shape[1]
        prevalence_scores = (df > threshold).mean(axis=0)
        keep_features = prevalence_scores[prevalence_scores >= prevalence].index
        df_filtered = df[keep_features]
        print(f"特征过滤: {initial_features} -> {df_filtered.shape[1]} (保留率: {df_filtered.shape[1]/initial_features:.2%})")
        return df_filtered

    def transform(self, data: pd.DataFrame, pseudocount: float = 1e-4):
        """
        clr transform
        """
        df = data.copy()
        df = np.where(df == 0, pseudocount, df)
        geo_means = gmean(df, axis = 1)
        clr_df = np.log(df) - np.log(geo_means).reshape(-1, 1)
        return clr_df
    
    def l1_l2(self, x, y, l1: float = None, l2: float = None):
        if l1 is None or l2 is None:
            en_cv = ElasticNetCV(alphas=[0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 1.0],
                                l1_ratio=[0.001, 0.01, 0.05, 0.1, 0.2, 0.5], 
                                cv = 5)
            l1, l2 = en_cv.l1_ratio_, en_cv.alpha_
        

    def exec(self, clean_process: Mapping = {"disease":30, "relative_abundance": 80, "disease": 70}):
        """
        Preprocessing and transform data
        """
        data = self.process(clean_process)
        data = self.filtration(data)
        trans_data = self.transform(data)
        y = self.label.reindex(data.index)
        y = self.encoder.fit_transform(y)
        if len(data) != len(y):
            raise ValueError(f"Please make sure your label{len(y)} and sample{len(data)} is same")
        return trans_data, y

def calc_dist_matrix(x, y, n_components: int):
    """
    Calculate distance matrix for group the disease.

    - x: [samples, features]
    - y: [samples,], like lable or encoded label.

    Return: dist_matrix `np.ndarray`
    """
    lda = LDA(n_components=n_components)

    reduced = lda.fit_transform(x, y)
    disease_centers = {}
    for disease in np.unique(y):
        # 提取该疾病在降维空间的所有样本
        X_disease = reduced[y == disease]
        # 计算中心点 (均值向量)
        center = np.mean(X_disease, axis=0)
        disease_centers[disease] = center

    centers_matrix = np.array([disease_centers[d] for d in np.unique(y)])
    dist_matrix = squareform(pdist(centers_matrix, metric='euclidean'))
    return dist_matrix

