#-------------------------------------------------
#  import libary
#-------------------------------------------------
import numpy as np
import cv2
import glob
# from matplotlib import pyplot as plt

def main():
    links_img = glob.glob('*.png')
    for link_img in links_img:
        img = cv2.imread(link_img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # img_gray = cv2.GaussianBlur(img_gray, (15,15), 10)
        img_thres = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY_INV)[1]

        #-------------------------------------------------
        #  Remove the palm
        #-------------------------------------------------
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))
        tophat = cv2.morphologyEx(img_thres, cv2.MORPH_TOPHAT, kernel)

        #-------------------------------------------------
        #  opening, remove noise
        #-------------------------------------------------
        kernel = np.ones((8,8),np.uint8)
        opening = cv2.morphologyEx(tophat, cv2.MORPH_OPEN, kernel)

        _, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #-------------------------------------------------
        #  delete excess area
        #-------------------------------------------------
        for i in range(len(contours)):
            if(cv2.contourArea(contours[i]) < 2000):
                x_min = min(contours[i][:,:,0])[0] - 1
                x_max = max(contours[i][:,:,0])[0] + 1
                y_min = min(contours[i][:,:,1])[0] - 1
                y_max = max(contours[i][:,:,1])[0] + 1
                opening[y_min:y_max,x_min:x_max] = 0
        #-------------------------------------------------
        #  counting 
        #-------------------------------------------------
        _, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #-------------------------------------------------
        #  save image
        #-------------------------------------------------
        cv2.putText(img, str(len(contours)),(100,100), cv2.FONT_HERSHEY_SIMPLEX, 4,(0,0,255),2,cv2.LINE_AA)
        cv2.imwrite(str(len(contours))+'_final.png',img)
        cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
