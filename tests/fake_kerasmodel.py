class FakeKerasModel():

    def __init__(self, name: str):
        self.name = name

    def predict(self, np_array, verbose: int = 0):
        # Simulate a prediction output
        return [[10, 20, 30, 40]] 

    