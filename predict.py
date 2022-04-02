from tqdm import tqdm
import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from scipy import signal 
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from tensorflow.keras.models  import Sequential
from tensorflow.keras.optimizers import SGD, Adam, Adadelta
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, Dropout, BatchNormalization, Dropout
import tensorflow as tf
import numpy as np
from os.path import exists
from datetime import datetime

# Run only if features.csv is not found or when implementing new features
# Read MAT files

def clean_data(array):
    transp_data = np.transpose(array, axes=[0,2,1])
    # unnorm_data = transp_data.reshape(*transp_data.shape, 1)
    cleaned_data = transp_data  / transp_data.max()
    
    return cleaned_data
    

test_mats = []
for dirname, _, filenames in os.walk('/project/kaggle_decmeg_2014/input/test/data'):
    for filename in filenames:
        print("Loading: ", os.path.join(dirname, filename))
        test_mats.append(loadmat(os.path.join(dirname, filename)))
# Merging data
print('Merging data')
test_data = clean_data(test_mats[0]['X'])

for i in tqdm(range(1, len(test_mats))):
    reshaped_input = clean_data(test_mats[i]['X'])
    print(reshaped_input.shape)
    test_data = np.concatenate((test_data, reshaped_input), axis=0)
    

del test_mats # Free RAM pls 
print('Done')

# for i in tqdm(range(0, len(train_data))):


now = datetime.now()
model = tf.keras.models.load_model(f'/project/kaggle_decmeg_2014/model/saved_models/{now.strftime("%Y_%m_%d")}')
preds = model.predict(test_data)

print(preds)

