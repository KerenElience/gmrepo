from utils.process import DataProcess
from src.randomforest import RFModel, RFOptimizer
from sklearn.model_selection import train_test_split



if __name__ == "__main__":
    prcd = DataProcess("E:/code/gmrepo/resource/meta.csv")
    x, y = prcd.exec()
    cls_name = prcd.encoder.classes_
    x_train, x_test, y_train, y_test = train_test_split(
                x, y, random_state=42, test_size=0.2 , shuffle=True, stratify=y
            )
    # optimizer = RFOptimizer(x_train, y_train)
    # best_params = optimizer.run()
    rf_model = RFModel(x_train, x_test, y_train, y_test, cls_name,)
    rf_model.train()
    rf_model.eval()
    # rf_model.save("E:/code/gmrepo/rfclassifier.pkl")