import cv2
import math
import numpy as np
import skimage
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
from skimage import transform
from scipy import stats, ndimage

###############################################################################################################################################
#BINARIZATION

def isScan(img):
    ih,iw = img.shape
    npixel = ih*iw
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    grey_pixels = np.sum(hist[15:241])
    ratio = grey_pixels/npixel
    if ratio < 0.1:
        return 1
    else:
        return 0

def photo_threshold(img):
    if img.size>2000*1000:
        img = cv2.GaussianBlur(img,(9,9),3)

    ih,iw = img.shape
    win_size = int(min(ih,iw)/60)
    if win_size%2==0:
        win_size = win_size+1
        
    th_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,win_size,10)
    return th_img

def morphology(img):
    img = photo_threshold(img)
    img = 255-img
    kernel = np.ones((3,3),np.uint8)
    kernel1 = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    dilation = cv2.dilate(closing,kernel1,iterations =1)
    dilation = cv2.erode(dilation,kernel,iterations = 1)
    hole_size = int(0.0001*img.shape[0]*img.shape[1])
    arr = dilation>0
    fin_img = remove_small_objects(arr, min_size = hole_size)
    fin_img = fin_img.astype(np.uint8)
    fin_img[fin_img==1] = 255
    fin_img = 255 - fin_img
    return fin_img

def scan_binariztion(img):
    th, output = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return output

def binarization(img):
    if isScan(img) == 1:
        return scan_binariztion(img)

    else:
        th_img = photo_threshold(img)
        morpho_img = morphology(th_img)
        return morpho_img

#####################################################################################################################################################
#SKEW-CORRECTION

def skewCorrection(img):
    edges = cv2.Canny(img,50,150)
    lines = cv2.HoughLines(edges,1,np.pi/180,50)
    theta=np.empty([0])
    for i in range(len(lines)):
        theta = np.append(theta,lines[i][0][1])

    theta = (theta/np.pi)*180
    theta = 90-theta
    theta = -np.sort(-theta)
    theta = [int(np.round(i)) for i in theta]
    deskewing_angle = stats.mode(theta).mode[0]

    deskew_img = ndimage.rotate(img,-deskewing_angle, mode = 'constant', cval = 255)
    deskew_img = 255 - deskew_img


    kernel = np.ones((3,3),np.uint8)
    kernel1 = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(deskew_img, cv2.MORPH_CLOSE, kernel)
    deskew_img = cv2.dilate(closing,kernel1,iterations = 1)
    deskew_img = cv2.erode(deskew_img,kernel,iterations = 2)

    deskew_img = 255 - deskew_img.astype(np.uint8)
    return deskew_img
    
    #if lines is not None:
    #    for i in range(0, len(lines)):
    #        rho = lines[i][0][0]
    #        theta1 = lines[i][0][1]
    #        a = math.cos(theta1)
    #        b = math.sin(theta1)
    #        x0 = a * rho
    #        y0 = b * rho
    #        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #        cv2.line(img, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

    #cv2.imshow('houghlines3.jpg',img)

########################################################################################################################################################
#SEGMENTATION
def isInside(rect1,rect2): #Check if rect2 is inside rect1
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    if(x1<x2 and y1<y2 and (x1+w1)>(x2+w2) and (y1+h1)>(y2+h2)):
        return True
    else:
        return False

def segmentation(img):
    #Creating border map
    neg_img = 255-img
    kernel1 = np.ones((3,3),np.uint8)
    erode = cv2.erode(neg_img,kernel1,iterations = 1)
    dest_xor = cv2.bitwise_xor(erode, neg_img, mask = None)

    #Creating contours
    contours,hierarchy = cv2.findContours(img, mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)
    img = img.copy()
    convex_hull = []
    bounding_rect = []
    centroid = []
    contours = sorted(contours, key=cv2.contourArea, reverse = True)
    for cnt in contours:
        convex_hull.append(cv2.convexHull(cnt))
        bounding_rect.append(cv2.boundingRect(cnt))
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        centroid.append([cx, cy])

    #Checking if one box is enclosed in another
    index = []
    count=0
    for i in range(1,len(bounding_rect)):
        for j in range(1,len(bounding_rect)):
            if i==j:
                continue

            if(isInside(bounding_rect[i], bounding_rect[j])):
                index.append(j)       
    

    convex_hull_new = []
    bounding_rect_new = []
    centroid_new = []

    for i in range(1,len(bounding_rect)):
        if i in index:
            continue
        convex_hull_new.append(convex_hull[i])
        bounding_rect_new.append(bounding_rect[i])
        centroid_new.append(centroid[i])

    #Cutting out each character and creation the array if images
    images = []
    for rect in bounding_rect_new:
        x,y,w,h = rect
        char = img[y:y+h, x:x+w]
        char = 255 - char
        pad = 20
        char = cv2.copyMakeBorder(char,pad,pad,pad,pad,cv2.BORDER_CONSTANT)
        char = 255 - char
        images.append(char)

    return convex_hull_new, centroid_new, bounding_rect_new, images 


def view_images(images):
    for i in range(0, len(images)):
        s = "char" + str(i) 
        cv2.imshow(s,images[i])
        plt.show()

def draw_bbox(img, bounding_rect_new):
    backtorgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    for rect in bounding_rect_new:
        x,y,w,h = rect
        cv2.rectangle(backtorgb,(x,y),(x+w,y+h),(0, 0, 255),2)
    cv2.imshow("Segmented image",backtorgb)
    plt.show()

################################################################################################################################################################
def crop_image(img, default_size=None):
    old_im = img
    img_data = np.asarray(old_im, dtype=np.uint8) # height, width
    nnz_inds = np.where(img_data!=255)
    print(img_data,nnz_inds)
    if len(nnz_inds[0]) == 0:
        if not default_size:
            return old_im
        else:
            assert len(default_size) == 2, default_size
            x_min,y_min,x_max,y_max = 0,0,default_size[0],default_size[1]
            old_im = old_im[y_min:y_max+1, x_min:x_max+1]
            return old_im
    y_min = np.min(nnz_inds[0])
    y_max = np.max(nnz_inds[0])
    x_min = np.min(nnz_inds[1])
    x_max = np.max(nnz_inds[1])
    old_im = old_im[y_min:y_max+1, x_min:x_max+1]
    return old_im

img = cv2.imread("abc.png", 0)
cv2.imshow("sf", img)
# ih, iw = img.shape
# #cv2.imshow("input image", img)
# #plt.show()
# binarized_img = binarization(img)
# #cv2.imshow("binarized image", binarized_img)
# deskew_img =  skewCorrection(binarized_img)
# #sh,sw = deskew_img.shape
# #cropped = deskew_img[int(sh/2 - ih/2):int(sh/2 + ih/2) , int(sw/2 - iw/2):int(sw/2 + iw/2)]
# #cv2.imshow("deskew img", cropped)
# #plt.show()
# convex_hull, centroid, bounding_rect, images = segmentation(deskew_img)
# draw_bbox(deskew_img, bounding_rect)
# view_images(images)
cv2.imshow("sad", crop_image(img))


cv2.waitKey(0)
cv2.destroyAllWindows()