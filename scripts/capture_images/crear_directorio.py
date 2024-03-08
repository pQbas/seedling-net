
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
