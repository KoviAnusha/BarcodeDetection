import time
from skimage.transform import pyramid_gaussian
from skimage.feature import hog
from sklearn.externals import joblib
from imutils.object_detection import non_max_suppression
import cv2
import numpy as np 
model_path= "../detect/data/models/svm.model"
imgpath="../detect/data/dataset/barData/TestImages/17.jpg"
def sliding_window(image, window_size, step_size):
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
    
if __name__ == "__main__":
    im = cv2.imread(imgpath)
    min_wdw_sz = (200, 120)
    step_size = (10, 10)
    downscale = 1.25
    visualize_det = False
    clf = joblib.load(model_path)
    detections = []
    probs=[]
    scale = 0
    for im_scaled in pyramid_gaussian(im, downscale=downscale):
        cd = []
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
        	break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            fd= hog(im_window, orientations=9, pixels_per_cell=(8,8), cells_per_block=(3,3), visualize=False, block_norm="L2-Hys", multichannel=None)                    
            fd=np.reshape(fd,(1,len(fd)))
            pred = clf.predict(fd)
            if pred == 1:
                detections.append((x, y,
                    int(min_wdw_sz[0]*(downscale**scale)),
                    int(min_wdw_sz[1]*(downscale**scale))))
                probs=np.append(probs,clf.decision_function(fd))
            if visualize_det:
                clone = im_scaled.copy()
                for x1, y1, _, _  in cd:
                    cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                        im_window.shape[0]), (0, 0, 0), thickness=100)
                cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                    im_window.shape[0]), (255, 255, 255), thickness=100)
                cv2.imshow("Sliding Window in Progress", clone)
                cv2.waitKey(3)
        scale+=1
    probs=np.array(probs)   
    clone = im.copy()
    for (x_tl, y_tl, w, h) in detections:
        cv2.rectangle(im, (x_tl, y_tl), (x_tl+w, y_tl+h), (255,255,0), thickness=2)
    cv2.imshow("Raw Detections before NMS", im)
    cv2.waitKey(0)
    detections = np.array([[x, y, x + w, y + h] for (x, y, w, h) in detections])
    detections = non_max_suppression(detections, probs, overlapThresh=.001)
    print "detections", detections
    for (x_tl, y_tl, w, h) in detections:
        cv2.rectangle(clone, (x_tl, y_tl), (w,h), (0,255,0), thickness=2)
    cv2.imshow("Final Detections after applying NMS", clone)
    cv2.waitKey(0)
