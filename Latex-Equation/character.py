import cv2
import os
import copy
import numpy as np
from matplotlib import pyplot as plt
import skimage
from skimage.feature import canny
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line, rotate
import math
from scipy import ndimage
from matplotlib.path import Path

#create identifier profiles for each character
def create_identifier(img,centroid):
    #     print("maxi:",np.max(img))
#     img = ~img
#     plt.imshow(img,cmap='gray')
#     plt.show()
    #assuming that the character pixels in the paper means those pixels belonging to character
    k = 16 
    id_profile = np.zeros(2*k+6)
    mom_inertia = 0
    char_count = 0
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if img[i,j] != 0:
                char_count+=1
                mom_inertia+=(i-centroid[0])**2+(j-centroid[1])**2
    mom_inertia/=(char_count**2)
    id_profile[0] = mom_inertia
    img[img==255] = 1
    mu00 = central_moments(img,0,0)
    etas = np.zeros((4,4))
    for p in range(0,4):
        for q in range(0,4):
            if p+q<=3:
                gamma = 1 + (p+q)/2
                etas[p,q] = central_moments(img,p,q)/(mu00**gamma)

    hu2 = (etas[2,0] - etas[0,2])**2 + (4*etas[1,1]**2)
    hu3 = (etas[3,0] - 3*etas[1,2])**2 + (3*etas[2,1] - etas[0,3])**2
    hu4 = (etas[3,0] + etas[1,2])**2 + (etas[2,0] + etas[0,3])**2
    hu5 = (etas[3,0] - 3*etas[1,2])*(etas[3,0] + etas[1,2])*((etas[3,0] + etas[1,2])**2 - 3*((etas[2,1] + etas[0,3])**2)) + (3*(etas[2,1] - etas[0,3]))*(etas[2,1] + etas[0,3])*(3*(etas[3,0] + etas[1,2])**2 - (etas[2,1] + etas[0,3])**2)
    hu6 = (etas[2,0] - etas[0,2])*((etas[3,0]+etas[1,2])**2 - (etas[2,1]+ etas[0,3])**2) + 4*etas[1,1]*((etas[3,0] + etas[1,2])*(etas[2,1] + etas[0,3]))
    hu7 = (3*etas[2,1] - etas[0,3])*(etas[3,0] + etas[1,2])*((etas[3,0] + etas[1,2])**2 - 3*(etas[2,1]-etas[0,3])**2) - (etas[3,0] - 3*etas[1,2])*(etas[2,1] + etas[0,3])*(3*(etas[3,0] + etas[1,2])**2 - (etas[2,1] + etas[0,3])**2)
    id_profile[2*k] = hu2
    id_profile[2*k+1] = hu3
    id_profile[2*k+2] = hu4
    id_profile[2*k+3] = hu5
    id_profile[2*k+4] = hu6
    id_profile[2*k+5] = hu7
    id_profile[1:2*k] = get_circular_topology(img,centroid,k)
    return id_profile

#calculate central moments
def central_moments(img, p, q):
    img[img==255] = 1
    M00 = 0 
    M10 = 0
    M01 = 0
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            M00+=img[i,j]
            M10+=i*img[i,j]
            M01+=j*img[i,j]
    
    x_bar = np.floor(float(M10)/float(M00))
    y_bar = np.floor(float(M01)/float(M00))
    
    cen_mmt = 0
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            cen_mmt += ((i-x_bar)**p)*((j-y_bar)**q)*img[i,j]
            
    return cen_mmt

# shift origin for centroids
def create_new_centroids(centroids,bboxs,ims):
    centroids_new = []
    for i in range(0, len(ims)):
        centrs = [centroids[i][0],centroids[i][1]]
        centrs[0]-=bboxs[i][0]
        centrs[1]-=bboxs[i][1]
        centroids_new.append(centrs)
    return centroids_new


#get angular distance
def ang_dist(xx,yy):
    lis = [] 
    for x,y in zip(xx,yy):
        delta = np.abs(x-y)
        if delta > np.pi:
            delta = 2*np.pi-delta 
        lis.append(delta)
    return max(lis)

# get circular mask
def circular_mask(img,centroid,radius):
    lastx,lasty = -10,-10
    last = 0
    count = 0
    cut_angles = []
    img = img.astype(np.uint8)
    angle = np.linspace(0,2*np.pi,3600)
    circle_x = np.rint(centroid[0]+radius*np.cos(angle)).astype(np.int64)
    circle_y = np.rint(centroid[1]+radius*np.sin(angle)).astype(np.int64)  
    for x,y,theta in zip(circle_x,circle_y,angle):
        if (x>=0 and x<img.shape[0] and y>=0  and y<img.shape[1]) == False:
            continue
        if img[x,y] != last :
            last = img[x,y]
            if img[x,y] == 1 and abs(lastx-x)+abs(lasty-y)>15:
                count += 1
                lastx,lasty = x,y
                cut_angles.append(theta)
                print(x,y)
    if count == 0:
        return 0,0
    cut_angles = np.asarray(cut_angles)
    angular_distance = ang_dist(np.roll(cut_angles,1),cut_angles)
    cv2.circle(img,(int(centroid[1]),int(centroid[0])),int(radius),1,thickness=1)
    # plt.imshow(img,cmap="gray")
    # plt.show()
    #print(count)
    return count,angular_distance/(2*np.pi)


# using connected components for determining cuts
def get_new_cuts(img, centroid, radius):
    lastx,lasty = -10,-10
    last = 0
    count = 0
    img = img.astype(np.uint8)
    img1 = np.zeros(img.shape).astype(np.uint8)
    cv2.circle(img1,(int(centroid[1]),int(centroid[0])),int(radius),1,thickness=1)
    after_and = np.logical_and(img1, img).astype(np.uint8)
    # percentage of image on circle
    img_on_circle = np.sum(after_and)/np.sum(img1)
    str_ele = skimage.morphology.selem.disk(3)
#     str_ele = skimage.morphology.selem.square(2)
    #after_and = skimage.morphology.binary_dilation(after_and,selem=str_ele)
    # plt.imshow(after_and,cmap='gray')
    # plt.show()
    all_labels = skimage.measure.label(after_and)
    reg = skimage.measure.regionprops(all_labels)
#     print(len(reg))
    cv2.circle(img,(int(centroid[1]),int(centroid[0])),int(radius),1,thickness=1)
    count = len(reg)

    return count,img_on_circle


# get all_ids for circles 
def get_circular_topology(img,centroid,k):
    circular_topology = np.zeros(2*k)
    #finding maximum distance
    edge_indices = np.argwhere(img)
    d=np.sqrt(np.max(np.sum((edge_indices-centroid)**2,axis=1)))/(k+1)
    for i in range(1,k+1):
#         circular_topology[i-1],circular_topology[7+i] = circular_mask(img,centroid,i*d)
        circular_topology[i-1],circular_topology[7+i] = get_new_cuts(img,centroid,i*d)
        
    return circular_topology[:-1]

#find nearest neighbour
def find_nn(img,centroid):
    #     fe_vector = create_identifier(cv2.GaussianBlur(img.astype(np.uint8),(5,5),3),centroid)
    fe_vector = create_identifier(img,centroid)
    mind = -1
    mival = 1000 
    for i in range(0,len(imgs_list)):
        val = np.sum( np.abs(np.array(fe_vector)-np.array(id_profs_list[i]))   )
        if val < mival:
            mival = val
            mind = i
    plt.imshow(imgs_list[mind])
    plt.show()
    #print(id_profs_list[mind])
    #print(fe_vector)

# find nearest neighbour weighted identifier profile 
def find__nn_new(img,centroid,id_profs_list):
#     fe_vector = create_identifier(cv2.GaussianBlur(img.astype(np.uint8),(5,5),3),centroid)
    fe_vector = create_identifier(img,centroid)
    mind = -1
    mival = 1000 
    mask = np.ones(fe_vector.size)
    mask[1:3] = 1
    for i in range(0,len(id_profs_list)):
        val = np.sum( np.abs(np.array(fe_vector)-np.array(id_profs_list[i]))  *mask  )
        if val < mival:
            mival = val
            mind = i
    return mind