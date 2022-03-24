import pandas as pd
import pickle
import json
import numpy as np

def loadfile(filename): 
    return pickle.load(open(filename, "rb")) 

def savefile(filename, *args):
    return pickle.dump(open(filename, "w+"), args) 

def loadjson(filename):
    return json.load(open(filename, "rb"))

def savejson(filename, *args):
    return json.dump(open(filename, "w+"), args)

def loaddata(xfile, yfile, mean_std_file="none"):
    X = pd.read_csv(xfile)    
    y = pd.read_csv(yfile)
    if mean_std_file=="none":
        return X, y, {}
    else:
        ms = loadjson(mean_std_file)
        mean = np.array(ms["means"]).reshape(1, -1)
        std = np.array(ms["stds"]).reshape(1, -1)
        X_col = X.shape[1]
        X_mean = mean[:1, :X_col]
        X_std = std[:1, :X_col]
        y_mean = mean[:1, X_col:]
        y_std = std[:1, X_col:]
        return X, y, {"X_mean": X_mean, "X_std": X_std, "y_mean": y_mean, "y_std":y_std}        


#