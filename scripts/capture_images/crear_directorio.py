
'''
Escribir un código que se encarge de crear una carpeta tipo "/[FECHA]" (eg. 24_03_08 - 2024 Marzo 08). 
Esta tiene que tener dos carpetas más, horizontal, vertical, params y un archivo altura.csv.

/[FECHA]
- /horizontal/rgb
- /vertical/rgb
- /params
- alturas.csv

Antes de crear el conjunto de carpetas se debe verificar si existe una carpeta con el mismo nombre 
en el directorio actual. Usar la libreria OS
'''
import os
from datetime import datetime

fecha_actual = datetime.now().strftime("%y_%m_%d") 

directorio_actual = os.path.dirname(os.path.abspath(__file__))

carpeta_base = fecha_actual
carpeta_principal = os.path.join(directorio_actual, carpeta_base)

indice = 2
while os.path.exists(carpeta_principal):
    carpeta_principal = os.path.join(directorio_actual, f"{carpeta_base}({indice})")
    indice += 1

os.makedirs(carpeta_principal, exist_ok=True)
os.makedirs(os.path.join(carpeta_principal, "horizontal", "rgb"), exist_ok=True)
os.makedirs(os.path.join(carpeta_principal, "vertical", "rgb"), exist_ok=True)
os.makedirs(os.path.join(carpeta_principal, "params"), exist_ok=True)

with open(os.path.join(carpeta_principal, "altura.csv"), "w") as f:
    pass

print("Estructura de carpetas creada:", carpeta_principal)


