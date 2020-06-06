# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 22:03:57 2019

It is a part of kaggle digit competetion
https://www.kaggle.com/c/digit-recognizer
download full files from above
files here are chopped to keep file size small

@author: Gaurav
"""
#%%
'''load all libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

#%%
'''set common style for the notebook'''
plt.style.use('seaborn')

#%%
'''data preprocessing'''
#Load the data

df_sample= pd.DataFrame(pd.read_csv('sample_submission.csv'))
df_train= pd.DataFrame(pd.read_csv('train.csv'))
df_evaluate= pd.DataFrame(pd.read_csv('test.csv'))
df_datagen= pd.DataFrame(pd.read_csv('cnn_mnist_datagen.csv'))

#Data Visualisation
sns.countplot(df_train.iloc[:,0])
print(df_train.iloc[:,0].value_counts())

#Check for missing data
null= df_train.isnull().any().describe()

#Setup test train split
x_train, x_test, y_train, y_test = train_test_split(df_train.iloc[:,1:].values,
                                                    df_train.iloc[:,0].values,
                                                    test_size=0.2,
                                                    random_state=41)

#Reshape input image
x_train= x_train.reshape(-1,28,28,1)
x_test= x_test.reshape(-1,28,28,1)
x_evaluate= df_evaluate.values.reshape(-1,28,28,1)

#see some example
plt.imshow(x_train[15,:,:,0])

#Normalise the data
x_train= x_train.reshape(-1,28,28,1)/255
x_test= x_test.reshape(-1,28,28,1)/255
x_evaluate= x_evaluate/255

#to categorical
y_train= to_categorical(y_train, num_classes=10)
y_test= to_categorical(y_test, num_classes=10)

#%%
'''Keras Model'''
#Model Arhitecture
model= Sequential([Conv2D(64,5,strides=1, padding='same',activation='relu',input_shape=[28,28,1]),
                    MaxPooling2D(2),
                    Conv2D(128,5,strides=1, padding='same',activation='relu'),
                    Dropout(0.2),
                    MaxPooling2D(2),
                    Flatten(),
                    Dense(10, activation='relu'),
                    Dense(10, activation='softmax')
                    ])

#Model compile
model.compile(optimizer=RMSprop(),
              loss='categorical_crossentropy',
              metrics=['acc'])
#model callback
lr = ReduceLROnPlateau(monitor='val_loss',
                       patience=3,
                       factor=0.5,
                       min_delta=1e-4,
                       min_lr=1e-5)
#Data augumentation
datagen= ImageDataGenerator(rotation_range=10,
                            zoom_range=0.1,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            )

datagen.fit(x_train)

#Model train
history= model.fit_generator(datagen.flow(x_train, y_train),
                   steps_per_epoch=300,
                   epochs=5,
                   validation_data=(x_test,y_test),
                   callbacks=[lr]
                   )

#%%
'''evaluation of model'''
#plotting losses and acuracy
sns.lineplot(np.arange(len(history.history['acc'])),
             history.history['acc'],
             label='accuracy', color='k')
sns.lineplot(np.arange(len(history.history['val_acc'])),
             history.history['val_acc'],
             label='validation accuray', color='b')
plt.xlabel('epoch')
plt.ylabel('accuracy')

#%%
'''kaggle submission'''
#model prediction
y_evaluate= model.predict(x_evaluate)
y_evaluate= np.argmax(y_evaluate, axis=1)

df_sample.iloc[:,1] = y_evaluate
df_sample.to_csv('kaggle190914.csv', index=False)

#%%
#confusion matrix
y_pred = np.argmax(model.predict(x_test), axis=1)
conf_mat = np.round(confusion_matrix(np.argmax(y_test, axis=1), y_pred),0)
sns.heatmap(conf_mat, cmap='Blues', annot=True, fmt='d')
plt.xticks(np.arange(10))
plt.xlabel('predicted label')
plt.yticks(np.arange(10))
plt.ylabel('actual label')