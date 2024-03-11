import cv2
import sys
import numpy as np
from camera import camera
import urllib.request
import calibrationzed2
import argparse
import pyzed.sl as sl

'''
Usar argparse para pasarle el nombre del directorio donde se van a estar guardando las imagenes de los plantines.

ejemplo:
$ python capturar_imagenes.py --dir 20_03_08
$ python capturar_imagenes.py --dir 20_03_08(2)
'''

parser = argparse.ArgumentParser(description='Guardar imágenes de cámaras en un directorio especificado.')
parser.add_argument('--folder', type=str, default='', help='Nombre del directorio donde se guardarán las imágenes.')

args = parser.parse_args()

FECHA = args.folder if args.folder else "capturas"

cam1 = camera(0)
cam2 = camera(1)
k = 0


while True:
    img1 = cam1.get_image(scale_percentage=100)
    img2 = cam2.get_image(scale_percentage=100)

    
    cv2.imshow('Camera1', img1)
    cv2.imshow('Camera2', img2)
    cv2.waitKey(10)
    
    fila = k//6 + 1
    columna = k%6 + 1
    
    if cv2.waitKey(0) & 0xFF == ord('t'):
        confirmation = input("Do you want to store these images? [yes/no]")  
        
        if confirmation == 'y':
            cv2.imwrite(f"{FECHA}/vertical/rgb/{fila}_{columna}.jpg",img1)
            cv2.imwrite(f"{FECHA}/horizontal/rgb/{fila}_{columna}.jpg",img2)
            cv2.waitKey(1)
            k += 1  
            print("save it!!!")

cv2.destroyAllWindows()