from sklearn.metrics import confusion_matrix

def get_cm(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    return cm