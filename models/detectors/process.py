
import cv2
import os
import numpy as np



def load_images_from_folder(folder, scale):
    images_list_ = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),scale)
        if img is not None:
            images_list_.append(img)
    return images_list_



def image_processing(img_, img_depth_):
    gray_ = cv2.inRange(img_depth_,(0),(10))

    width = int(img_.shape[1])
    height = int(img_.shape[0])
    dim = (width, height)
    gray_resized = cv2.resize(gray_, dim, interpolation = cv2.INTER_AREA)
    canvas = np.zeros((height,width),dtype=np.uint8)


    gray_ = gray_resized.copy()
    kernel = np.ones((3,3), np.uint8)
    for i in range(2):    
        gray_ = cv2.erode(gray_, kernel, iterations=3)
        gray_ = cv2.dilate(gray_, kernel, iterations=1)


    gray_negative = cv2.bitwise_not(gray_)
    gray_eroded = cv2.erode(gray_negative, kernel, iterations=4)
    gray_eroded_negative = cv2.bitwise_not(gray_eroded)

    ret, mask_ = cv2.threshold(gray_eroded_negative,100,255,cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(mask_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            cv2.drawContours(canvas,[contour],-1,(255),-1)
    
    ret, mask_2 = cv2.threshold(canvas,100,255,cv2.THRESH_BINARY)

    return mask_2




def applyDepthOverImage(image, depth_image):
    image = image[478:720,303:1049,:]
    gray = depth_image[455:624,376:893]
    #gray2 = cv2.inRange(gray,(0),(10))

    mask = image_processing(image, gray)
    res = cv2.bitwise_and(image,image,mask = mask)
    return res


def postProcess_h(detections):
    
    post_process_detections = []

    for seedling in detections:
        frame = seedling.copy()
        
        # linear filter on hsv-space
        low_H, low_S, low_V = [30, 40, 40]
        high_H, high_S, high_V = [180, 255, 255]
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        
        # dilatation process
        dilatation_size = 3
        dilation_shape = cv2.MORPH_ELLIPSE
        element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
        dilatation_result = cv2.dilate(frame_threshold, element)

        # find max area
        ret, thresh = cv2.threshold(dilatation_result, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #frame_contours = cv2.drawContours(frame.copy(), contours, -1, (0,255,0), 3)

        max = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max:
                frame_contours_max = cnt
                max = area

        # get the bbox of the max-area
        x1,y1,x2,y2 = cv2.boundingRect(frame_contours_max)
        bbox_seedling = cv2.rectangle(frame.copy(), (x1,y1), (x2,y2), (0,255,0),1)
        max_contour = cv2.drawContours(frame.copy(), [frame_contours_max], -1, (0,255,0), 3)
        
        
        post_process_detections.append(frame.copy()[y1:y2,:,:])
        #post_process_detections.append(frame.copy())

    return post_process_detections




def postProcess_v(detections):
    
    post_process_detections = []

    for seedling in detections:
        frame = seedling.copy()
        
        # # linear filter on hsv-space
        # low_H, low_S, low_V = [37, 60, 53]
        # high_H, high_S, high_V = [180, 255, 255]
        # frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        
        # # dilatation process
        # dilatation_size = 3
        # dilation_shape = cv2.MORPH_ELLIPSE
        # element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
        # dilatation_result = cv2.dilate(frame_threshold, element)

        # # find max area
        # ret, thresh = cv2.threshold(dilatation_result, 127, 255, 0)
        # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # #frame_contours = cv2.drawContours(frame.copy(), contours, -1, (0,255,0), 3)

        # max = 0
        # for cnt in contours:
        #     area = cv2.contourArea(cnt)
        #     if area > max:
        #         frame_contours_max = cnt
        #         max = area

        # # get the bbox of the max-area
        # x1,y1,x2,y2 = cv2.boundingRect(frame_contours_max)
        # bbox_seedling = cv2.rectangle(frame.copy(), (x1,y1), (x2,y2), (0,255,0),1)
        # max_contour = cv2.drawContours(frame.copy(), [frame_contours_max], -1, (0,255,0), 3)
        
        
        post_process_detections.append(frame.copy())

    return post_process_detections


