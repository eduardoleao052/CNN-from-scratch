import numpy as np
import pandas as pd
import os

train = pd.read_csv('data/mnist_train.csv')
test = pd.read_csv('data/mnist_test.csv')
xl = train.iloc[:,1:]
yl = train.iloc[:,0]
xt = test.iloc[:,1:]
yt = test.iloc[:,0]


os.chdir('/workspaces/neural_network/data')

xl.to_csv("xl.csv",index=False)
xt.to_csv("xt.csv",index=False)
yl.to_csv("yl.csv",index=False)
yt.to_csv("yt.csv",index=False)