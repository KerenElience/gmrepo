class BaseModule():
    def __init__(self, param: dict = None):
        self.params = param

    def train(self, x, y):
        pass

    def eval(self, x, y):
        pass