import cv2
import numpy as np
from datetime import datetime, timedelta


Ancho = 200
Alto = 100
CATEGORIAS = ["carro", "moto", "cicla"]

def labelmap(w):
    if np.all(w == np.array([1, 0, 0])):
        return 0
    elif np.all(w == np.array([0, 1, 0])):
        return 1
    elif np.all(w == np.array([0, 0, 1])):
        return 2
    else:
        pos = np.argmax(max(w))
        return pos

def prepare(dir):
    img = cv2.imread(dir)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.resize(img1, (Ancho, Alto))
    return img2.reshape(-1, Ancho, Alto)

def cron():
    now = datetime.now()
    new = now + timedelta(seconds=10)
    return new








