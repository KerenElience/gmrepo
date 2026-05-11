class BaseModule():
    def __init__(self, param: dict = None, model = None):
        self.params = param
        self.model = model

    def train(self, x, y):
        pass

    def eval(self, x, y):
        pass