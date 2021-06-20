import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import librosa
from os import listdir
import numpy as np
from tensorflow.keras.utils import to_categorical

model = Sequential()
model.add(Dense(256, input_dim=7, activation='relu'))
model.add(keras.layers.Dropout(0.25))
model.add(Dense(224, activation='relu'))
model.add(keras.layers.Dropout(0.25))
model.add(Dense(112, activation='softmax'))
model.add(Dense(16))
model.add(Dense(2, activation='softmax'))
model.compile(loss='mae', metrics=['accuracy'], optimizer='adam')


ses = []
for i in listdir(r'C:\Users\darak\Desktop\Mali'):
    sound, fr=librosa.load(fr'C:\Users\darak\Desktop\Mali\{i}')
    describe = pd.DataFrame(sound)[:275000].describe()
    a = [float(describe.loc['mean'].values), float(describe.loc['std'].values), float(describe.loc['min'].values), float(describe.loc['25%'].values), float(describe.loc['50%'].values), float(describe.loc['75%'].values), float(describe.loc['max'].values)]
    ses.append(a)

ses = np.array(ses).reshape(5, 7)
data = pd.DataFrame(ses)
data['malimi?'] = 1

sesothers = []
for i in listdir(r'C:\Users\darak\Desktop\Others'):
    sound, fr = librosa.load(fr'C:\Users\darak\Desktop\Others\{i}')
    describe=pd.DataFrame(sound)[:275000].describe()
    q = [float(describe.loc['mean'].values), float(describe.loc['std'].values), float(describe.loc['min'].values), float(describe.loc['25%'].values), float(describe.loc['50%'].values), float(describe.loc['75%'].values), float(describe.loc['max'].values)]
    sesothers.append(q)

data2 = pd.DataFrame(sesothers)
data2['malimi?'] = 0
data = data.append(data2)
for i in data.drop(columns=['malimi?']).columns:
    data[i] = data[i]/max(data[i])
x_train = data.drop(columns='malimi?')
y_train = data['malimi?']

y_train = to_categorical(y_train)

model.fit(x_train, y_train, epochs=100)

a = model.predict()