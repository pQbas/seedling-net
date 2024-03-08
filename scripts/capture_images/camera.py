import cv2

class camera():
    def __init__(self,id):
        
        self.cam = cv2.VideoCapture(id)
        
        return
    
    def get_image(self, scale_percentage=100):
        
        ret, frame = self.cam.read()
        # to resize the image from the 2nd camera
        #scale_percent = 100 # percent of original size
        width = int(frame.shape[1] * scale_percentage / 100)
        height = int(frame.shape[0] * scale_percentage / 100)
        dim = (width, height)
        
        # resize image
        img = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
        return img