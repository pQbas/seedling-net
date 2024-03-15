import cv2
import numpy as np

# Ruta de la imagen
image_file = "/24_03_13/horizontal/rgb/1_1.jpg"

# Parámetros de distorsión
k1 = -4.04861084e-01
k2 = -1.39165795
p1 = 2.03628652e-03
p2 = -1.61717690e-03

# Parámetros intrínsecos
fx = 977.91106995
fy = 975.46476456
cx = 342.58569442
cy = 211.95226346

# Cargar la imagen
image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
rows, cols = image.shape

# Crear una matriz para la imagen corregida de la misma dimensión
image_undistort = np.zeros((rows, cols), dtype=np.uint8)

# Calcular la imagen corregida
for v in range(rows):
    for u in range(cols):
        x = (u - cx) / fx
        y = (v - cy) / fy
        r = np.sqrt(x * x + y * y)
        x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x)
        y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y
        u_distorted = fx * x_distorted + cx
        v_distorted = fy * y_distorted + cy

        # Asignar valor con interpolación más cercana
        if 0 <= u_distorted < cols and 0 <= v_distorted < rows:
            image_undistort[v, u] = image[int(v_distorted), int(u_distorted)]
        else:
            image_undistort[v, u] = 0

# Mostrar las imágenes
cv2.imshow("distorted", image)
cv2.imshow("undistorted", image_undistort)
cv2.waitKey(0)
cv2.destroyAllWindows()
