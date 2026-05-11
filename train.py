from gmrepo.src.models.multi_disease_classifier import RFModel
from gmrepo.src.models.gbdt import XGBModel
from utils.optimizer import Optimizer

class SubModel():
    """
    Training sub disease model.
    """
    def __init__(self, insolutions, data, label, encoder, classifier_model, optimizer):
        self.insolutions = insolutions

        self.data = data
        self.label = label
        self.encoder = encoder
        self.model = classifier_model
        self.optim = optimizer
        self.models = {}

    def get_group_data(self, group):
        mask = self.label.isin(group)
        mask_label = self.label[mask]
        x_group = self.x[mask]
        y_group = self.encoder.fit_transform(mask_label)
        return x_group, y_group
    
    