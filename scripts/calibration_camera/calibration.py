import numpy as np
import cv2 as cv
import glob
import pickle
import argparse
import os
import sys
import warnings 
# Suppress the warning message
warnings.filterwarnings("ignore", category=UserWarning, module="cv2")


def get_image_list(directory_path):
    if not os.path.exists(directory_path):
        print(f"Warning: The directory {directory_path} doesn't exist.")
        return []

    image_list = []
    file_list = os.listdir(directory_path)

    for file in file_list:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(directory_path, file)
            image_list.append(image_path)

    return image_list

def verify_directory(directory_path):
    if not os.path.exists(directory_path):
        print(f"Warning: The directory {directory_path} doesn't exist.")
        sys.exit()
    else:
        print(f"The directory {directory_path} exists.")


def verify_create_directory(directory_path):
    if not os.path.exists(directory_path):
        # The directory doesn't exist, so we create it
        os.makedirs(directory_path)
        print(f"The directory {directory_path} has been created.")
    else:
        # The directory already exists
        print(f"The directory {directory_path} already exists.")


def check_image_folder(directory_path):
    if not os.path.exists(directory_path):
        print(f"Warning: The directory {directory_path} doesn't exist.")
        return False

    file_list = os.listdir(directory_path)
    image_count = sum(1 for file in file_list if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')))

    if image_count == 0:
        print(f"The directory {directory_path} doesn't contain any images.")
        return False
    else:
        print(f"The directory {directory_path} contains {image_count} image(s).")
        return True


def main(folder_images=None, folder_results=None, size_of_chessboard_squares_mm=None):

    ################ VERIFY THE EXISTANCE OF THE FOLDER #############################
    PATH_IMAGES=folder_images
    PATH_SAVE_RESULTS=folder_results

    verify_directory(PATH_IMAGES)
    check_image_folder(PATH_IMAGES)
    verify_create_directory(PATH_SAVE_RESULTS)

    ################ CHESSBORAD PARAMETERS #############################
    size_of_chessboard_squares_mm = float(size_of_chessboard_squares_mm) #size_of_chessboard_squares_mm = 31

    chessboardSize = (6,4)
    frameSize = (640,480)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    objp = objp * size_of_chessboard_squares_mm

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = get_image_list(PATH_IMAGES)
    
    for image in images:
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret == True:

            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(10)

    cv.destroyAllWindows()

    # ############## CALIBRATION #######################################################

    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
    print(cameraMatrix)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    
    pickle.dump((cameraMatrix, dist), open( os.path.join(PATH_SAVE_RESULTS, "calibration.pkl") , "wb" ))
    pickle.dump(cameraMatrix, open(  os.path.join(PATH_SAVE_RESULTS, "cameraMatrix.pkl") , "wb" ))
    pickle.dump(dist, open(os.path.join(PATH_SAVE_RESULTS, "dist.pkl") , "wb" ))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--images', 
                        type=str, 
                        default='images/', 
                        help='Path of folder with images taken using the camera to be calibrated')
    
    parser.add_argument('--results', 
                        type=str, 
                        default='results/', 
                        help='Path where to save the intrinsic parameters of camera')
    
    parser.add_argument('--box_size', 
                        type=str, 
                        default='results/', 
                        help='Size in mm of the box of the chessboard')

    opt = parser.parse_args()
  
    main(folder_images=opt.images, folder_results=opt.results, size_of_chessboard_squares_mm=opt.box_size)
