import os
import pyzed.sl as sl
import cv2

def main():
    # Crear un objeto Camera
    zed = sl.Camera()

    # Crear un objeto InitParameters y configurar los parámetros de la cámara
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.AUTO  # Usar modo de video HD720 o HD1200, según el tipo de cámara.
    init_params.camera_fps = 30  # Establecer fps en 30

    # Abrir la cámara
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Error al abrir la cámara: " + repr(err) + ". Salir del programa.")
        exit()

    # Inicializar ventana para mostrar la transmisión en tiempo real
    cv2.namedWindow("Transmisión en tiempo real ZED", cv2.WINDOW_NORMAL)

    # Capturar imágenes con la cámara derecha y guardarlas
    i = 0
    runtime_parameters = sl.RuntimeParameters()
    while True:
        # Capturar una imagen, se debe proporcionar un objeto RuntimeParameters a grab()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Una nueva imagen está disponible si grab() devuelve SUCCESS
            image = sl.Mat()
            zed.retrieve_image(image, sl.VIEW.RIGHT)  # Capturar imagen con cámara derecha
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Obtener la marca de tiempo en el momento de la captura de la imagen
            image_data = image.get_data()  # Obtener datos de la imagen

            # Guardar la imagen si se presiona la tecla 's'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                image_path = os.path.join('images', f'imagen_{i}.png')  # Ruta de la imagen a guardar
                cv2.imwrite(image_path, image_data)  # Guardar la imagen en la carpeta especificada
                print(f"Imagen guardada en {image_path}")
                i += 1

            # Mostrar la transmisión en tiempo real en la ventana
            cv2.imshow("Transmisión en tiempo real ZED", cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))

        else:
            print("Error al capturar la imagen")

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cerrar la cámara y destruir la ventana
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
