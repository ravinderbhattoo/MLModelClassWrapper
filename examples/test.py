import sys
sys.path.append("..")
#==========================================
import matplotlib.pyplot as plt
from MLModelClassWrapper import *
from sklearn.metrics import r2_score

X, y, ms = loaddata("./test_X.csv", "./test_y.csv", mean_std_file="./mean_std.json")

# X = X.values
# y = y.values

# models = MLModel("./model.pkl", scale=True, ms=ms)
# y_ = models.predict(X).reshape(y.shape)

# print(r2_score(y, y_))

# plt.scatter(y, y_)
# plt.show()

import requests

url = 'http://127.0.0.1:5000/app/Dielectric_Constant/'
myobj = {'X': [i.tolist() for i in X.values]}


url2 = 'http://127.0.0.1:5000/app/allmodels/'
myobj2 = {'X': [i.tolist() for i in X.values], 'models':['Dielectric_Constant', 'Dielectric_Constant']}

x = requests.post(url, json = myobj)

print(x.json().keys())

x2 = requests.post(url2, json = myobj2)

print(x2.json().keys())


