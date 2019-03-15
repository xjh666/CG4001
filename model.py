from joblib import dump, load

class Model:
    def __init__(self, model_path):
        print('Loading model from ' +  model_path)
        self.model = load(model_path)
        print('Successfully loaded the model: ', self.model)

    def predict(self, input_data):
        return self.model.predict(input_data)
