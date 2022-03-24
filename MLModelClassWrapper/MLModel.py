from .IO import *

def row_(x):
    return x.reshape(1, -1)

def scale_(X, m, s):
    m = row_(m)
    s = row_(s)
    if len(X.shape)==2:
        pass
    else:
        X = row_(X)
    return (X - m) / s

def descale_(X, m, s):
    m = row_(m)
    s = row_(s)
    if len(X.shape)==2:
        pass
    else:
        X = row_(X)
    return X*s + m

def scaled_prediction(func, scale, ms):
    if scale:
        X_mean = ms["X_mean"]
        X_std = ms["X_std"]
        y_mean = ms["y_mean"]
        y_std = ms["y_std"]
        def f(X):
            X_s = scale_(X, X_mean, X_std)
            y_s = func(X_s)
            y = descale_(y_s, y_mean, y_std)
            return y    
        return f
    else:
        return func

class MLModel():
    def __init__(self, modelfile, predict_fn="none", scale=False, ms="none"):
        self.model = loadfile(modelfile)        
        self.type_ = str(type(self.model)).split("'")[1].split(".")[0]
        if self.type_ in ["xgboost", "sklearn"]:
            self.predict = scaled_prediction(self.model.predict, scale, ms)
        elif self.type_ in ["torch", "pytorch", "tensorflow"]:
            self.predict = scaled_prediction(self.model, scale, ms)
        else:
            if predict_fn == "none":
                raise "Unable to find model type. Please provide prediction function name as predict_fn kwarg."
            else:
                if predict_fn in dir(self.model):
                    self.predict = scaled_prediction(getattr(self.model, predict_fn), scale, ms)
                else:
                    raise "Unable to determine prediction function."                    