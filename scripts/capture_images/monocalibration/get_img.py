import cv2

cap = cv2.VideoCapture(0)

num = 0

while cap.isOpened():
    # Capturar fotograma de la cámara
    success, img = cap.read()
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1) 
    img = cv2.warpAffine(img, M, (cols, rows))
    # Mostrar la imagen
    cv2.imshow('Img', img)

    # Esperar la tecla 's' para guardar la imagen
    k = cv2.waitKey(5)
    if k == 27:  # Presionar ESC para salir del bucle
        break
    elif k == ord('s'):  # Presionar 's' para guardar la imagen
        cv2.imwrite(f'scripts/capture_images/monocalibration/imagenes/img{num}.png', img)
        print("Imagen guardada como img" + str(num) + ".png")
        num += 1

# Liberar recursos y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
