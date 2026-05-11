class BaseModule():
    def __init__(self, model_type: str = None, params: dict = None):
        self.model_type = model_type
        self.default_params = params

    def train(self, x, y):
        pass

    def eval(self, x, y):
        pass