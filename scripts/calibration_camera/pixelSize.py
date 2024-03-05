import cv2
import numpy as np
import pickle


def plot_circle(center, image):

    # Specify the circle parameters
    radius = 10  # Radius of the circle
    color = (0, 0, 255)  # Color of the circle in BGR format
    thickness = 2  # Thickness of the circle outline

    return cv2.circle(image, center, radius, color, thickness)

def undistord_with_remaping(frame, cameraMatrix, dist):
    img = frame
    h,  w = img.shape[:2]
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

    # Undistort with Remapping
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst


def compute_pixel_size(calibration_pattern_size, object_width_cm):
    # Assuming the chessboard pattern dimensions (number of corners)
    pattern_rows, pattern_cols = calibration_pattern_size

    # Create object points in real-world coordinates
    objp = np.zeros((pattern_rows * pattern_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)

    # Prepare arrays to store object points and image points from calibration images
    objpoints = []  # 3D points in real-world coordinates
    imgpoints = []  # 2D points in image plane

    # Capture calibration images until enough valid chessboard patterns are found
    capture = cv2.VideoCapture(2)  # Replace 0 with the appropriate camera index if using multiple cameras
    pattern_found_count = 0
    max_pattern_count = 100

    cameraMatrix, dist = None, None
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    
    # Specify the file paths of the pickle files
    calibration_file_path = "./calibration.pkl"
    cameraMatrix_file_path = "./cameraMatrix.pkl"
    dist_file_path = "./dist.pkl"

    # Open and load the calibration.pkl file
    with open(calibration_file_path, 'rb') as file:
        cameraMatrix, dist = pickle.load(file)

    # Open and load the cameraMatrix.pkl file
    with open(cameraMatrix_file_path, 'rb') as file:
        cameraMatrix = pickle.load(file)

    # Open and load the dist.pkl file
    with open(dist_file_path, 'rb') as file:
        dist = pickle.load(file)

    while True:
        ret, frame = capture.read()
        frame = undistord_with_remaping(frame, cameraMatrix, dist)

        # Convert image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (pattern_cols, pattern_rows), None)

        # If chessboard pattern is found, add object points and image points
        if ret:
            pattern_found_count += 1
            objpoints.append(objp)
            imgpoints.append(corners)

            # Specify the circle parameters
            center_1 = (int(corners[0][0][0]), int(corners[0][0][1]))  # Center coordinates (x, y)
            plot_circle(center_1, frame)

            center_2 = (int(corners[3][0][0]), int(corners[3][0][1]))
            plot_circle(center_2, frame)

            distance_pixels = np.sqrt((center_1[0] - center_2[0])**2 + (center_1[1] - center_2[1])**2)
            print(distance_pixels/94.36)

            # Draw and display the corners
            # cv2.drawChessboardCorners(frame, (pattern_cols, pattern_rows), corners, ret)
            cv2.imshow('Chessboard', frame)
        
        k = cv2.waitKey(10)

        if k == 27:
            break


    capture.release()
    cv2.destroyAllWindows()

    # # Calibrate the camera
    # ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # # Compute pixel size using the object width in centimeters
    focal_length = cameraMatrix[0, 0]
    pixel_size = object_width_cm / focal_length

    return pixel_size

# Specify the size of the calibration chessboard pattern (number of corners)
calibration_pattern_size = (6, 4)

# # Specify the width of a known object in centimeters
object_width_cm = 9.4

# Compute the pixel size
pixel_size = compute_pixel_size(calibration_pattern_size, object_width_cm=10)

# Print the computed pixel size
print("Pixel size:", pixel_size, "mm/pixel")
