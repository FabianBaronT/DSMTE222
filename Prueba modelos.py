from Red import *
img_pathc = r"C:\Users\Faby\Desktop\Imagenes U Originales\Imagenes\cicla"
img_pathm = r"C:\Users\Faby\Desktop\Imagenes U Originales\Imagenes\moto"
img_pathca = r"C:\Users\Faby\Desktop\Imagenes U Originales\Imagenes\carro"
lista = []
#for filt in filtros:
#    for densa in densas:
#        for d in drop:
cont = 0
            #NAME = "RedConv_F{}_D{}_dropout{}".format(filt, densa, d)
            #dirNAME = r"C:\Users\Faby\Desktop\Remake\models\{}".format(NAME)
dirNAME =  r"C:\Users\Faby\Desktop\Remake\models\RedConvRE_F128_D128_dropout0.5"
predi = tf.keras.models.load_model(dirNAME)
for img in os.listdir(img_pathc):
    dir = r"C:\Users\Faby\Desktop\Imagenes U Originales\Imagenes\cicla\{}".format(img)
    pe = CATEGORIAS[labelmap(predi.predict([prepare(dir)]))]
    if pe == 'cicla':
        cont = cont + 1
print(cont)
for img in os.listdir(img_pathm):
    dir = r"C:\Users\Faby\Desktop\Imagenes U Originales\Imagenes\moto\{}".format(img)
    pe = CATEGORIAS[labelmap(predi.predict([prepare(dir)]))]
    if pe == 'moto':
        cont = cont + 1
print(cont)
for img in os.listdir(img_pathca):
    dir = r"C:\Users\Faby\Desktop\Imagenes U Originales\Imagenes\carro\{}".format(img)
    pe = CATEGORIAS[labelmap(predi.predict([prepare(dir)]))]
    if pe == 'carro':
        cont = cont + 1


print(cont)
