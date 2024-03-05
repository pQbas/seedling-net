# Importing libraries
import sys
import cv2
import os
from process import load_images_from_folder, image_processing, applyDepthOverImage, postProcess_h, postProcess_v
import numpy as np


sys.path.insert(1, 'detectors')
sys.path.insert(2, 'detectors/yolov7')
from detector import Detector
detector_h = Detector('yolo7', weights='./weights/yolov7-hseed.pt', data='./weights/opt.yaml', device='cuda:0')  # eg: 'yolov7' 'maskrcnn'
detector_v = Detector('yolo7', weights='./weights/yolov7-vseed.pt', data='./weights/opt.yaml', device='cuda:0')  # eg: 'yolov7' 'maskrcnn'


images_horizontal = load_images_from_folder('./data/gallery_24_03_23/horizontal/rgb', 1)
images_vertical = load_images_from_folder('./data/gallery_24_03_23/vertical/rgb', 1)
depth_map_vertical = load_images_from_folder('./data/gallery_24_03_23/vertical/mask', 0)


#f = open ('ratios.txt','w')
vertical_width_list = []
vertical_height_list = []
vertical_area_list = []
horizontal_width_list = []
horizontal_height_list = []
horizontal_area_list = []

idx = 0
n_x = 0

for img_h, img_v, depth_v in zip(images_horizontal, images_vertical, depth_map_vertical):

 
    print(f'---------{n_x}-----------')
    
    # normalization
    # img_hsv = cv2.cvtColor(img_v, cv2.COLOR_BGR2HSV)
    # img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
    # img_v = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    #cv2.imshow('original_image_vertical',img_v)
    #cv2.imshow('original_image',depth_v)

    # img_v = applyDepthOverImage(img_v, depth_v)

    cv2.imshow('',img_v)
    cv2.waitKey(0)

    #cv2.imshow('Post_processing_vertical',img_v)

    prediction_v = detector_v.predict(img_v)
    if prediction_v is not None:
        prediction_v = sorted(prediction_v, key=lambda obj: obj.bbox[0])
    else:
        print("No_prediction")
        continue

    h1,w1,c1 = img_v.shape


    # pre-processing of horizontal images
    #cv2.imshow('original_image_horizontal',img_h)
    #img_h = img_h[280:500, 200:700,:]
    cv2.imshow('Post_processing_horizontal',img_h)
    cv2.waitKey(0)

    # # normalization
    # img_hsv = cv2.cvtColor(img_h, cv2.COLOR_RGB2HSV)
    # img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
    # img_h = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    h2,w2,c2 = img_h.shape
    img_h = cv2.resize(img_h, (int(w2/0.5748385471282702), int(h2*(1/0.5748385471282702))))
    #print(img_h.shape)
    prediction_h = detector_h.predict(img_h)
    if prediction_h is not None:
        prediction_h = sorted(prediction_h, key=lambda obj: obj.bbox[0])
    else:
        continue
    

    # detection to the seedlings on vertical view
    vertical_detections = []
    if prediction_v is not None:
        img_v_copy = img_v.copy()
        for pred in prediction_v:
            x1, y1, x2, y2 = pred.bbox
            # cv2.rectangle(img_v, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
            if pred.cls.item() == 0:
                h,w,c = img_v.shape
                mask = pred.mask*255
                mask = cv2.resize(mask, (w,h), 1, 1, interpolation=cv2.INTER_LINEAR)[int(y1):int(y2), int(x1):int(x2)]
                seedling = img_v[int(y1):int(y2), int(x1):int(x2),:]
                seedling_masked = cv2.bitwise_and(seedling, seedling, mask=mask)
                vertical_detections.append(seedling_masked)
                
                # plotting the results
                cv2.rectangle(img_v_copy, (int(x1),int(y1)),(int(x2),int(y2)) , (0,0,255), 2 )
                # cv2.waitKey(0)

        # plotting the result
        cv2.imshow('vertical-view', img_v_copy)
        cv2.moveWindow('vertical-view', 100, 10)


    # detection to the seedlings on horizontal view
    horizontal_detections = []
    if prediction_h is not None:
        img_h_copy = img_h.copy()
        #print(img_h.shape)
        for pred in prediction_h:
            id = 0
            x1, y1, x2, y2 = pred.bbox
            if pred.cls.item() == 0:        
                h,w,c = img_h.shape
                mask = pred.mask*255
                mask = cv2.resize(mask, (w,h), 1, 1, interpolation=cv2.INTER_LINEAR)[int(y1):int(y2), int(x1):int(x2)]
                seedling = img_h[int(y1):int(y2), int(x1):int(x2),:]
                seedling_masked = cv2.bitwise_and(seedling, seedling, mask=mask)
                horizontal_detections.append(seedling_masked)
                cv2.rectangle(img_h_copy, (int(x1),int(y1)),(int(x2),int(y2)) , (0,0,255), 2 )

        # plotting the result
        cv2.imshow('horizontal-view', img_h_copy)
        cv2.moveWindow('horizontal-view', 1000, 10)
    
    
    # processing to the horizontal detections to get the leaf area
    seedlings_v = vertical_detections #postProcess_v()
    seedlings_h = horizontal_detections #postProcess_h()

    
    '''---------horizontal-----------'''
    wh_list = []
    data_h = []
    w_ = 0
    for id, seed_h in enumerate(seedlings_h):


        seed_h = postProcess_h([seed_h])[0]
        h,w,c = seed_h.shape
        no_zeros = np.count_nonzero(seed_h[:,:,0]==0)
        leaf_area_h = h*w - no_zeros
        data_h.append([h,w,leaf_area_h])

        # plotting the results
        cv2.imshow(f'h-{id}', seed_h)
        cv2.moveWindow(f'h-{id}', 90 + w_ + w, 800)

        w_ = w_ + w + 500
        wh_list.append(w)



    '''---------vertical-----------'''
    wv_list = []
    data_v = []
    w_ = 0
    for id, seed_v in enumerate(seedlings_v):
        
        h,w,c = seed_v.shape
        no_zeros = np.count_nonzero(seed_v[:,:,0]==0)
        leaf_area_v = h*w - no_zeros
        data_v.append([h,w,leaf_area_v])
        
        # plotting the results
        cv2.imshow(f'v-{id}', seed_v)
        cv2.moveWindow(f'v-{id}', 10 + w_ + w, 400)
        w_ = w_ + w + 400
        wv_list.append(w)

    
    ''' getting the features '''
    
    
    if len(wh_list) == len(wv_list):

        ''' used to save the features used to make the classification '''
        for v_data,h_data in zip(data_v,data_h):
            hv,wv,areav = v_data
            hh,wh,areah = h_data

            vertical_height_list.append(hv)
            vertical_width_list.append(wv)
            vertical_area_list.append(areav)

            horizontal_height_list.append(hh)
            horizontal_width_list.append(wh)
            horizontal_area_list.append(areah)


        
        ''' to calibrate the images '''
        #print('wh',wh_list)
        #print('wv',wv_list)
        #print('dif',np.array(wh_list) / np.array(wv_list))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #for ratio in (np.array(wh_list) / np.array(wv_list)):
        #    f.write(f'{ratio}\n')

    n_x += 1

    #cv2.imwrite(f'../gallery/non-date/chops/{idx}.jpg', seedling_masked)
    #idx += 1
                
    #result = hsv_filter(seedling_masked)
    #horizontal_detections.append(seedling_masked)
    cv2.imshow('1',seedling_masked)
    #cv2.imshow('result',result)
    cv2.imshow('1',img_v)
    cv2.waitKey(0)

#f.close()

# import pandas as pd

# data = {
#     'vertical_height': vertical_height_list,
#     'vertical_width': vertical_width_list,
#     'vertical_area': vertical_area_list,
#     'horizontal_height': horizontal_height_list,
#     'horizontal_width': horizontal_width_list,
#     'horizontal_area': horizontal_area_list,
#     'class': 1
# }

# df = pd.DataFrame(data)
# df.to_csv('./data/17-01-23_seedlings_class_good')
# print(df)
