import os
import pickle
import cv2
from keras.utils import load_img
from tqdm import tqdm
import random as rn
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, array_to_img, load_img

CATEGORIAS = ["carro", "moto", "cicla"]
Ancho = 200
Alto = 100

def New_datos():
    DG_data = 'Carro_Aumento'
    if not os.path.exists(DG_data):
        os.mkdir(DG_data)


    data_path = r"C:\Users\Faby\Desktop\Remake\Imagenes1\cicla"
    data_new = r"C:\Users\Faby\Desktop\Remake\Imagenes\cicla"

    data_gen = ImageDataGenerator(
        rotation_range=0,
        zoom_range=0.1,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
    )

    for img_file in os.listdir(data_path):
        img_path = os.path.join(data_path, img_file)
        img = load_img(img_path)  # carga la imagen como un objeto PIL
        x = img_to_array(img)  # convierte la imagen en un arreglo de numpy
        x = x.reshape((1,) + x.shape)  # cambia la forma del arreglo para su procesamiento
        i = 0
        for batch in data_gen.flow(x, batch_size=1, save_to_dir=data_new, save_prefix='aug', save_format='jpg'):
            i += 1
            if i > 5:  # genera 5 imágenes aumentadas por cada imagen original
                break

def Gen_datos():
    data = []
    for categoria in CATEGORIAS:
        path = os.path.join(DATADIR, categoria)
        valor = CATEGORIAS.index(categoria)
        listdir = os.listdir(path)
        for i in tqdm(range(len(listdir)),desc = categoria):
            imagen_nombre = listdir[i]
            imagen_ruta = os.path.join(path, imagen_nombre)
            imagen_ruta = cv2.imread(imagen_ruta)
            imagen_gris = cv2.cvtColor(imagen_ruta, cv2.COLOR_BGR2GRAY)
            imagen_resized = cv2.resize(imagen_gris, (Ancho,Alto))
            data.append([imagen_resized, valor])
            #print(imagen_nombre)
            #cv2.imshow('sda',imagen_resized)
            #print(valor)
            cv2.waitKey(0)
    rn.shuffle(data)
    x = []
    y = []

    for par in data:
        x.append(par[0])
        y.append(par[1])

    x = np.array(x).reshape(-1,Ancho,Alto,1)

    pickle_out = open("x.pickle", "wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()



if __name__ == "__main__":
    DATADIR = "C:\\Users\\Faby\\Desktop\\Remake\\Imagenes"
    Gen_datos()
    #New_datos()
