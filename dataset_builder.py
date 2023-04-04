import sys
sys.path.insert(1, 'detectors')
sys.path.insert(2, 'detectors/yolov7')
from detector import Detector
import matplotlib.pyplot as plt
import cv2
import pandas


def getFeatures(detector, img):
    img_pred = detector.predict(img)
    H, W, C = img.shape

    # verify if the prediction was succesful
    if img_pred is None or len(img_pred) == 0:
        return None

    if img_pred[0].area/(H*W) > 0.14:
        return None

    # get the height, width, area. Normalize them and save it in a list
    x1,y1,x2,y2 = img_pred[0].bbox

    area_normalized = img_pred[0].area/(H*W)
    heigth_normalized = (y2-y1)/H
    width_normalized = (x2-x1)/W

    return area_normalized, heigth_normalized, width_normalized


def getData(path, df, detector_h, detector_v, show=False):
    col = range(1,7)
    row = range(1,13)

    height1 = []
    width1 = []
    area1 = []

    height2 = []
    width2 = []
    area2 = []

    quality = []

    horizontal_non_detected, vertical_non_detected = 0,0
    fig = plt.figure(figsize=(30, 30))
    
    idx = 0
    for i in col:
        for j in row:
            # read the corresponding seedling and apply yolov7 to obtain their mask, area, bbox
            h1,w1,a1,h2,w2,a2 = 0,0,0,0,0,0

            if detector_h != None:
                img_h = plt.imread(f"{path}/horizontal/rgb/seedlings_{j}_{i}.jpg")
                result1 = getFeatures(detector_h, img_h)
                
            if detector_v != None:
                img_v = plt.imread(f"{path}/vertical/rgb/seedlings_{j}_{i}.jpg")
                depth = plt.imread(f"{path}/vertical/depth/seedlings_{j}_{i}.jpg")
                ret, thresh1 = cv2.threshold(depth, 120, 255, cv2.THRESH_BINARY)
                img_v_maksed = cv2.bitwise_and(img_v,img_v,mask = thresh1)
                result2 = getFeatures(detector_v, img_v_maksed)
            
            # assign a quality according the long of the max leaf
            max_leaf = df[f"col{i}"][f"row{j}"]
            quality_status = 0
            if max_leaf == -1:
                continue
            elif max_leaf > 80:
                quality_status = 1
            
            # verify if the predictions are not NoneType object
            if result1 is not None and result2 is not None:
                height1.append(result1[0])
                width1.append(result1[1])
                area1.append(result1[2])

                height2.append(result2[0])
                width2.append(result2[1])
                area2.append(result2[2])
                
                quality.append(quality_status)
            
            else:
                continue

            idx = idx + 1
            if show != False:
                # to plot something
                img = cv2.putText(img=img_h, 
                                text=f"{max_leaf}mm", 
                                org=(200,400), 
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                fontScale=2, color=(255,0,0), 
                                thickness=2, 
                                lineType=cv2.LINE_AA)
                
                fig.add_subplot(12, 6, idx)
                plt.imshow(img)


    # build a dataframe with the data recolected
    dataset = {
        'quality': quality,
        'area_hor': area1,
        'height_hor': height1,
        'width_hor': width1,
        'area_ver': area2,
        'height_ver': height2,
        'widht_ver': width2
    }
    
    #print(horizontal_non_detected, vertical_non_detected)
    return dataset

if __name__ == '__main__':
    detector_h = Detector('yolo7', weights='./weights/yolov7-hseed.pt', data='./weights/opt.yaml', device='cuda:0')  # eg: 'yolov7' 'maskrcnn'
    detector_v = Detector('yolo7', weights='./weights/yolov7-vseed.pt', data='./weights/opt.yaml', device='cuda:0')  # eg: 'yolov7' 'maskrcnn'
    
    df = pandas.read_csv('./data/gallery_17_03_23/Plantines_17_03_23 - Sheet1.csv')
    data_17_03_23 = pandas.DataFrame(getData('./data/gallery_17_03_23', df, detector_h=detector_h, detector_v=detector_v))

    df = pandas.read_csv('./data/gallery_24_03_23/Plantines_24_03_23 - Sheet1.csv')
    data_24_03_23 = pandas.DataFrame(getData('./data/gallery_24_03_23',df, detector_h=detector_h, detector_v=detector_v))

    df = pandas.read_csv('data/gallery_03_03_23_tray1/Plantines_03_02_23_tray1 - Hoja 1.csv')
    data_03_03_23_tray1 = pandas.DataFrame(getData('./data/gallery_03_03_23_tray1',df, detector_h=detector_h, detector_v=detector_v))

    df = pandas.read_csv('data/gallery_03_03_23_tray2/Plantines_03_02_23_tray2 - Hoja 1.csv')
    data_03_03_23_tray2 = pandas.DataFrame(getData('./data/gallery_03_03_23_tray2',df, detector_h=detector_h, detector_v=detector_v))

    df = pandas.read_csv('data/gallery_31_03_23/Plantines_31_03_23 - Sheet1.csv')
    data_31_03_23 = pandas.DataFrame(getData('data/gallery_31_03_23',df, detector_h=detector_h, detector_v=detector_v))

    combined_df = pandas.concat(
        [data_03_03_23_tray1,
        data_03_03_23_tray2,
        data_17_03_23,
        data_24_03_23,
        data_31_03_23], axis=0)

    combined_df.to_csv('data/dataset_4weeks_H1W1A1_H2W2A2.csv', index=False)