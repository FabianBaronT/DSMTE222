import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import regularizers
import pickle
import os
import numpy as np
import gc
from Funciones import *

gc.collect()
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#filtros = [32, 64, 128]
#densas = [128, 256, 512]
#drop = [0, 0.2, 0.5]
filtros = [128]
densas = [128]
drop = [0.5]

x = pickle.load(open("x.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

x = x/255.0
y = tf.keras.utils.to_categorical(y,3)

def entrenar():
    for filt in filtros:
            for densa in densas:
                for d in drop:
                    NAME = "RedConvRE_F{}_D{}_dropout{}".format(filt, densa, d)
    #                NAME = "ModeloFinal"
                    tensorboard = TensorBoard(log_dir='log/{}'.format(NAME))
                    modelo = Sequential()
                    modelo.add(Conv2D(filt, (5, 5), strides=(4,4), input_shape=x.shape[1:], activation='relu'))
                    modelo.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

                    modelo.add(Conv2D(filt, (3, 3), padding='same', activation='relu'))
                    modelo.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

                    modelo.add(Flatten())

                    modelo.add(Dense(densa, activation='relu'))
                    modelo.add(Dropout(d))

                    modelo.add(Dense(3, activation='softmax'))


                    modelo.compile(loss="categorical_crossentropy",
                                   optimizer="adam",
                                   metrics=['accuracy'])

                    modelo.fit(x, y, batch_size=32, epochs=30,validation_split=0.15, callbacks=[tensorboard])
                    #modelo.save("models/{}".format(NAME))
                    modelo.save(r"C:\Users\Faby\Desktop\Remake/RedFinal.h5")
                    modelo.summary()
