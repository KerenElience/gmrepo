import os
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv(override=True)
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_NAME = os.getenv("DB_NAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def get_db_connection():
    """
    Connect GMRepo
    """
    db_url = (f"""mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{3306}/{DB_NAME}""")
    try:
        engine = create_engine(db_url, pool_size=10,
                               max_overflow=20,
                               pool_timeout=300,
                               echo=False)
        print("数据库连接成功")
        return engine
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return None

def extract_region_data(country: str, savepath: str):
    """
    Extract specifical country metagenomics genus abundance data.
    """
    run_id_to_disease = """
    SELECT 
        p.run_id, 
        s.experiment_type, 
        s.country, 
        s.longitude, 
        s.latitude, 
        s.phenotype, 
        s.disease,
        sa.ncbi_taxon_id,
        sa.relative_abundance
    FROM project_to_run_id_to_phenotypes AS p
    LEFT JOIN sample_to_run_info AS s
        ON p.run_id = s.run_id
    LEFT JOIN species_abundance AS sa
        ON p.run_id = sa.accession_id
    WHERE s.country = %(country)s 
    AND s.experiment_type = 'Metagenomics'
    AND sa.taxon_rank_level = "genus";
    """

    connection = get_db_connection()
    if connection is None:
        raise RuntimeError(f"No found DataBase named {DB_NAME}, please download from https://2909682517.github.io/gmrepodocumentation/usage/downloaddatafromgmrepo/#download-processed-data-from-gmrepo")
    try:
        data = pd.read_sql(run_id_to_disease, connection, params={"country": country})
        data.to_csv(savepath, index = None)
    except Exception as e:
        print(e)

def explore_data(metadata: pd.DataFrame):
    """
    - run_id
    - disease
    - phenotype
    - relative_abundance
    - ncbi_taxon_id
    """
    phenotype = metadata.drop_duplicates(subset="run_id").value_counts("phenotype").reset_index()
    phenotype.loc[len(phenotype)] = ["disease" , phenotype[phenotype["phenotype"]!="Health"]["count"].sum()]
    phenotype = phenotype.iloc[[0, -1], :].reset_index(drop=True)
    fig, ax = plt.subplots(1, 2, clear = True, figsize = (12, 9), dpi = 360)
    ## plot different phenotype counts
    ax[0].set_title("Phenotype Counts")
    ax[0].set_xlabel("phenotpye")
    ax[0].spines[['top', 'right']].set_color(None)
    ax[0].bar(phenotype["phenotype"], phenotype["count"])
    
    ## plot different sample's bacteria genus counts
    genus = metadata.groupby("run_id").agg({"ncbi_taxon_id": len}).reset_index().sort_values("ncbi_taxon_id")
    ax[1].set_title("Genus Counts Hist")
    ax[1].set_xlabel("sample's genus count")
    ax[1].spines[['top', 'right']].set_color(None)
    ax[1].hist(genus["ncbi_taxon_id"])
    plt.savefig("./meta.png")
    plt.close()

if __name__ == "__main__":
    savefold = os.getenv("META_SAVEPATH")
    savepath = os.path.join(savefold, "meta.csv")
    if not os.path.exists(savefold):
        os.makedirs(savefold, exist_ok=True)
    if not os.path.exists(savepath):
        extract_region_data("China", savepath)
    metadata = pd.read_csv(savepath)
    explore_data(metadata)