import pickle
import os
import xgboost as xgb
#print(os.getcwd())
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import numpy as np

data_dict = pickle.load(open('./Data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

print(data.shape)