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
    
if not exists('/project/kaggle_decmeg_2014/input/train/data/train_data.npy'):
    train_mats = []
    for dirname, _, filenames in os.walk('/project/kaggle_decmeg_2014/input/train/data'):
        for filename in filenames:
            print("Loading: ", os.path.join(dirname, filename))
            train_mats.append(loadmat(os.path.join(dirname, filename)))
    # Merging data
    print('Merging data')
    train_data = clean_data(train_mats[0]['X'])
    print(train_data.shape)
    train_lables = train_mats[0]['y']
    for i in tqdm(range(1, len(train_mats))):
        reshaped_input = clean_data(train_mats[i]['X'])
        print(reshaped_input.shape)
        train_data = np.concatenate((train_data, reshaped_input), axis=0)
        train_lables = np.concatenate((train_lables, train_mats[i]['y']), axis=0)
    
    del train_mats # Free RAM pls 
    print('Done')
    
    print(len(train_data))
    # for i in tqdm(range(0, len(train_data))):
        
    np.save('/project/kaggle_decmeg_2014/input/train/data/train_data.npy', train_data)
    np.save('/project/kaggle_decmeg_2014/input/train/data/train_lables.npy', train_lables)
else:
    train_data = np.load('/project/kaggle_decmeg_2014/input/train/data/train_data.npy')
    train_lables = np.load('/project/kaggle_decmeg_2014/input/train/data/train_lables.npy')

test_mats = []
test_indeces = []
for dirname, _, filenames in os.walk('/project/kaggle_decmeg_2014/input/test/data'):
    for filename in filenames:
        print("Loading: ", os.path.join(dirname, filename))
        import re
        tests = loadmat(os.path.join(dirname, filename))
        string = os.path.join(dirname, filename)
        prefix = './input/test/data/test_subject'
        suffix = '.mat'
        result = re.search(f'{prefix}(.*){suffix}', string)
        number = int(result.group(1))
        index_list = [ number*1000 + i + 1 for i in range(len(tests)) ]
        test_mats.append(tests)
        test_indeces.append(index_list)
# Merging data
print('Merging data')
test_data = clean_data(test_mats[0]['X'])

for i in tqdm(range(1, len(test_mats))):
    reshaped_input = clean_data(test_mats[i]['X'])
    print(reshaped_input.shape)
    test_data = np.concatenate((test_data, reshaped_input), axis=0)
    

del test_mats # Free RAM pls 
print('Done')

#define model layers
input_shape = (1, 306, 375, 1)

# #create model
model = Sequential([
    Conv1D(25, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(375,306)),
    Dropout(0.5),
    BatchNormalization(),
    MaxPool1D(5, strides=5, padding='valid'),
    Conv1D(50, kernel_size=11, strides=1, padding='same', activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    MaxPool1D(5, strides=5, padding='valid'),
    Conv1D(100, kernel_size=15, strides=3, padding='same', activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    MaxPool1D(5, strides=5, padding='valid'),
    Flatten(),
    Dropout(0.5),
    BatchNormalization(),
    Dense(10, activation='softmax'),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=3e-4),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
now = datetime.now()
model.save(f'/project/kaggle_decmeg_2014/model/saved_models/{now.strftime("%Y_%m_%d")}')

hist = model.fit(train_data, train_lables, epochs=300, 
                    validation_split=0.2)


preds = model.predict(test_data)
np.save('/project/kaggle_decmeg_2014/input/test/data/preds.npy', preds)
print(preds)

submission_dict = {'Id':test_indeces,'Prediction':preds}
submission_df = pd.DataFrame(submission_dict, columns=['Id','Prediction'])
submission_df.to_csv('/project/kaggle_decmeg_2014/model/submission.csv', index=False)
with open('/project/kaggle_decmeg_2014/model/train_history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
