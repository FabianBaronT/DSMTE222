import cv2
import multiprocessing
import os
from Funciones import *
import tensorflow as tf
import requests

deteccion = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=50)#400, 50
pred = tf.keras.models.load_model("ModeloFinal.h5")

def leer_video(video_path, frame_queue):
    n = datetime.now() - timedelta(seconds=20)
    id = 0
    listad = [0 , 0 , 0]
    listacx = []
    global i
    cap = cv2.VideoCapture(video_path)
    #cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        zona = cv2.resize(frame, (1280, 720))
        mascara = deteccion.apply(zona)
        _, mascara = cv2.threshold(mascara, 254, 255, cv2.THRESH_BINARY)
        contornos, _ = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detecciones = []
        for cont in contornos:

            area = cv2.contourArea(cont)
            x, y, ancho, alto = cv2.boundingRect(cont)
            if area > 40000 and ancho > 450 and alto > 150:

                cx = int((x + ancho)/2)
                listacx.append(cx)
                if listad[-1] < listad[-2]:
                    sentido = 'sumar'
                elif listad[-1] > listad[-2]:
                    sentido = 'restar'

                detecciones.append([x, y, ancho, alto])
                d = sum(listacx[-10:]) / 10
                listad.append(d)
                if 550 < cx < 580:
                    direccion = "imagen{}.jpg".format(id)
                    m = datetime.now()
                    if direccion not in os.listdir(r"C:\Users\Faby\Desktop\Multiprocesos_2P_CN") and m > n:
                        frame = cv2.flip(frame, 0)
                        frame_queue.put((frame, sentido))
                        n = cron()
                        id = id + 1
                    else:
                        break
        zona = cv2.flip(zona, 0)
        cv2.imshow("Carretera", zona)
        key = cv2.waitKey(5)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

# Función para guardar las imágenes
def guardar_imagenes(output_folder, frame_queue, num_frames):
    os.makedirs(output_folder, exist_ok=True)
    for i in range(num_frames):
        frame, sentido = frame_queue.get()
        filename = os.path.join(output_folder, f"frame_{i:04d}.png")
        frame = cv2.resize(frame, (Ancho, Alto))
        cv2.imwrite(filename, frame)

        imag1 = prepare(filename)
        imag2 = pred.predict(imag1)
        imag3 = labelmap(imag2)
        veh = CATEGORIAS[imag3]
        url_id = "https://backend-fabian-production.up.railway.app/parking/{}/{}".format(sentido, veh)
        print(url_id)
        requests.get(url_id)

if __name__ == "__main__":
    video_path = "videoSalidaC2.mp4"
    output_folder = r"C:\Users\Faby\Desktop\Multiprocesos_2P_CN"
    num_frames_to_save = 100

    frame_queue = multiprocessing.Queue()

    # Crear procesos para leer el video y guardar las imágenes
    leer_video_process = multiprocessing.Process(target=leer_video, args=(video_path, frame_queue))
    guardar_imagenes_process = multiprocessing.Process(target=guardar_imagenes, args=(output_folder, frame_queue, num_frames_to_save))

    # Iniciar los procesos
    leer_video_process.start()
    guardar_imagenes_process.start()

    # Esperar a que ambos procesos terminen
    leer_video_process.join()
    guardar_imagenes_process.join()

    print("Procesamiento de video completado.")
