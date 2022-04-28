import os 
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import imutils
import json
import scipy.ndimage
import math
import string
import warnings

#set file path--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
pinholeimg = (
        "data/images/pinhole v2/Image_20220128120328217.bmp",
        "data/images/pinhole v2/Image_20220128120338885.bmp",
        "data/images/pinhole v2/Image_20220128120354435.bmp",
        "data/images/pinhole v2/Image_20220128120401761.bmp",
        "data/images/pinhole v2/Image_20220128120438410.bmp",
        "data/images/pinhole v2/Image_20220128120449320.bmp"
)

crackimg = (
            "data/images/crack v2/Image_20220128120524949.bmp",
            "data/images/crack v2/Image_20220128120703362.bmp",
            "data/images/crack v2/Image_20220128120732189.bmp",
            "data/images/crack v2/Image_20220128121231370.bmp",
            "data/images/crack v2/Image_20220128121318791.bmp",
            "data/images/crack v2/Image_20220128121338966.bmp",
            "data/images/crack v2/Image_20220128121453834.bmp",
)

knotimg = (
           "data/images/knot v2/Image_20220128121004555.bmp",
           "data/images/knot v2/Image_20220128121045551.bmp",
           "data/images/knot v2/Image_20220128121104393.bmp",
)

undersizeimg = (
           "data/images/Undersize v2/Image_20220128121539218.bmp",
           "data/images/Undersize v2/Image_20220128121601854.bmp",
           "data/images/Undersize v2/Image_20220128121621265.bmp",
           "data/images/Undersize v2/Image_20220128121634083.bmp",
)

crack_video = "data/videos/crack_simulated_video.avi"
deadknot_video = "data/videos/deadknot_simulated_video.avi"
smallknot_video = "data/videos/smallknot_simulated_video.avi"
pinhole_video = "data/videos/holes_simulated_video.avi"
good_video = "data/videos/good_simulated_video.avi"

#set file path-[END]-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

os.system("cls")

#function config-------------------------------------------------------------------------------------------------------------------------------

#config for some function
def orientated_non_max_suppression(mag, ang):
    ang_quant = np.round(ang / (np.pi/4)) % 4
    winE = np.array([[0, 0, 0],[1, 1, 1], [0, 0, 0]])
    winSE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    winS = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    winSW = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    magE = non_max_suppression(mag, winE)
    magSE = non_max_suppression(mag, winSE)
    magS = non_max_suppression(mag, winS)
    magSW = non_max_suppression(mag, winSW)

    mag[ang_quant == 0] = magE[ang_quant == 0]
    mag[ang_quant == 1] = magSE[ang_quant == 1]
    mag[ang_quant == 2] = magS[ang_quant == 2]
    mag[ang_quant == 3] = magSW[ang_quant == 3]
    return mag

#config for some function
def non_max_suppression(data, win):
    data_max = scipy.ndimage.filters.maximum_filter(data, footprint=win, mode='constant')
    data_max[data != data_max] = 0
    return data_max

#function config-[END]------------------------------------------------------------------------------------------------------------------------------


#pre-processing------------------------------------------------------------------------------------------------------------------

#gammacorrection needed!!!
def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
 
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
 
    return cv2.LUT(src, table)

#piece-wise transformation
def pixel_value(img, r1,s1,r2, s2):
         if (0 <=img and img<=r1):
             return (s1/r1)*img
         elif (r1 <= img and img <=r2):
             return 255 # 0 
         else:
             return ((255-s2)/(255-r2))* (img-r2)+s2

#adjust brightness and contrast
def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

#remove background (mostly conveyer belt) and return cropped image
def croppic(img):
    
    #preprocessing
    oriimg = img.copy()
    img = cv2.blur(img, (3,3))
    img = cv2.bilateralFilter(img, 1, 99, 99)
    img = apply_brightness_contrast(img,100,80)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    #threshing
    thresh = cv2.inRange(hsv, (45, 0, 60), (220, 255, 255))
    
    #find edge
    contours = cv2.findContours(thresh ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    contours = imutils.grab_contours(contours) 
    cv2.drawContours(img, contours, -1, (255,0 , 0), 2)
    
    #setup mask
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, -1, (255,255,255), -1) 
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (17,17), 1)
   
    #remove noise
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 696:
            mask[y:y + h, x:x + w] = 255
    
    #inverse it just in case i need it
    mask_inverse = 255 - mask
    
    #add it to the image (crop it with the white mask)
    out=cv2.addWeighted(mask, 1, oriimg, 1, 0)
    
    #return it
    return out

#pre-processing-[END]-----------------------------------------------------------------------------------------------------------------



#defect detection-------------------------------------------------------------------------------------------------------------------

#crack detection 
def detectCrack(img):
    
    #config
    with_nmsup = True 
    fudgefactor = 0.8 
    sigma = 21 
    kernel = 2*math.ceil(2*sigma)+1 
    
    #backup image for spare
    oriimg = img.copy()
    gray_image = img.copy()
    
    #pre-processing
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.blur(gray_image,(1,1))
    gray_image = cv2.bilateralFilter(gray_image, 7, 55,55)
    gray_image = apply_brightness_contrast(gray_image,-25,45)
    gray_image = cv2.blur(gray_image,(1,1))
    gray_image = gammaCorrection(gray_image,1.2) #ok
    gray_image = (np.log(gray_image+1)/(np.log(1+np.max(gray_image))))*255
    gray_image = np.array(gray_image,dtype=np.uint8)
    blur = cv2.GaussianBlur(gray_image, (51, 51), 21)

    # perform sobel edge detection
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=1)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=1)
    mag = np.hypot(sobelx, sobely)
    ang = np.arctan2(sobely, sobelx)

    # threshold
    threshold = 4 * fudgefactor * np.mean(mag)
    mag[mag < threshold] = 0
    
    #either get edges directly
    if with_nmsup is False:
        mag = cv2.normalize(mag, 0, 255, cv2.NORM_MINMAX)
        kernel = np.ones((5,5),np.uint8)
        result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
        orb = cv2.ORB_create(nfeatures=15)

        # Make featured Image
        keypoints, descriptors = orb.detectAndCompute(result, None)
        featuredImg = cv2.drawKeypoints(oriimg, keypoints,oriimg, color = (0,255,0))
       
        return featuredImg,keypoints

    #or apply a non-maximal suppression
    else:

        # non-maximal suppression
        mag = orientated_non_max_suppression(mag, ang)
        
        # create mask
        mag[mag > 0] = 255
        mag = mag.astype(np.uint8)


        kernel = np.ones((3,3),np.uint8)
        result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel,iterations = 3)
        
        orb = cv2.ORB_create(nfeatures=150)

        # Make featured Image
        keypoints, descriptors = orb.detectAndCompute(result, None)
        featuredImg = cv2.drawKeypoints(oriimg, keypoints,oriimg, color = (0,255,255))
       
        return featuredImg,keypoints

#knot detection    
def detechknot(img):
    
    #backup image for spare
    featureimg = img.copy()
    cropedimg = img.copy()

    #pre-processing
    img = cv2.blur(img, (7,7))
    img = apply_brightness_contrast(img,-10,20)
    img = gammaCorrection(img,0.9) 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blur = cv2.blur(hsv,(3,3))  

    #prepare mask for hsv...
    #and trash it
    lower_red = np.array([0,0,0])
    upper_red = np.array([99,255,100])
    mask = cv2.inRange(blur, lower_red, upper_red)
    _, thresh = cv2.threshold(mask, 170, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=1)

    #find edgy
    contours, _= cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #crop the deathknot out to reduce noise for further processing...
    #and put text on it if detected death knot
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 500:
            continue
        cropedimg = cv2.rectangle(cropedimg, (x, y), (x+w, y+h), (255, 255, 255), -1)
        cv2.rectangle(featureimg, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(featureimg, "Status: {}".format('Death Knot'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 0), 2)

    #return the feature img just in case...
    #return the croped img for further processing...
    #return the edgy also   
    return featureimg,cropedimg,contours

#detect smallknot
def detechsmallknot(img):
    
    #backup img
    featureimg = img.copy()
    cropedimg = img.copy()

    #periperi-processing
    img = cv2.GaussianBlur(img, (17,17), 0)  
    img = apply_brightness_contrast(img,-60,70)
    img = gammaCorrection(img,1.5) #ok
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blur = cv2.blur(hsv,(3,3))  
    
    #prepare mask and trash the image
    lower_red = np.array([0,0,0])
    upper_red = np.array([99,100,50])
    mask = cv2.inRange(blur, lower_red, upper_red)
    _, thresh = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=1)

    #find edgy
    contours, _= cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #produce a croped image and featuredimg pls forgive my typo
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 100:
            continue
        cropedimg = cv2.rectangle(cropedimg, (x, y), (x+w, y+h), (255, 255, 255), -1)
        cv2.rectangle(featureimg, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(featureimg, "Status: {}".format('Small Knot'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)

    #return featureimg,cropedimg,and edgy
    return featureimg,cropedimg,contours

#detech pinhole
def detectpinhole(img):
    
    #backup
    featuredimg = img.copy()
    
    #preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grau = cv2.medianBlur(gray, 3)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    gray = gammaCorrection(gray,1.75) #ok
    gray = apply_brightness_contrast(gray, -10, 20)
    ret, thresh = cv2.threshold(gray,127, 255, cv2.THRESH_BINARY)
    
    #kenel
    linek = np.zeros((11,11),dtype=np.uint8)
    linek[5,...,5]=1
    x=cv2.morphologyEx(gray, cv2.MORPH_OPEN, linek ,iterations=5)
    gray-=x
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE)
    counter=0

    #put desc in img
    for cnt in contours:
        counter += 1
        area = cv2.contourArea(cnt)
        (x, y, w, h) = cv2.boundingRect(cnt)
        if area<150:
            cv2.drawContours(featuredimg,[cnt],0,(255,0,0),2)
            cv2.rectangle(featuredimg, (x-5, y-5), (x+w+5, y+h+5), (0, 0, 255), 2)
            cv2.putText(featuredimg, str(counter), (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2)
        
    #return pic and edge
    return featuredimg,contours

#defect detection-[END]------------------------------------------------------------------------------------------------------------------

#draw defect--------------------------------------------------------------------------------------------------------------------------

#draw crack
def crackdraw(keypoint,img):    

    # draw keypoint
    featuredImg = cv2.drawKeypoints(img, keypoint,img, color = (255,0,255))
    return img
    
#draw deathknot
def deathknowdrawcontoure(contours,img):
    
    for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 500:
                continue
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, "Status: {}".format('Death Knot'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)    

    return img

#draw smallknot
def smallknotdrawcontoure(contours,img):
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 100:
            continue
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, "Status: {}".format('Small Knot'), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)
 
    return img   

#drawpinhole
def pinholedrawcontoure(contours,img):
    counter = 0
    for cnt in contours:
        counter += 1
        area = cv2.contourArea(cnt)
        (x, y, w, h) = cv2.boundingRect(cnt)
        if area<100:
            cv2.drawContours(img,[cnt],0,(255,0,0),2)
            cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), (0, 0, 255), 2)
            cv2.putText(img, str(counter), (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2)
            
    return img

#draw defect-[END]-------------------------------------------------------------------------------------------------------------------------



#video processing---------------------------------------------------------------------------------------------------------------------------------------

#put name in video
def putvideoname(name,img) :
    cv2.putText(img, "{}".format(name), (10, img.shape[1]+150), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)

#wood defect detection algo for wood video
def detect_defect_video(path):
    
    cap = cv2.VideoCapture(path)
    
    while(True):

            #some preprocess
            ret, frame = cap.read() 
            frame = cv2.resize(frame,(1590,940) )
            frame = frame[0:frame.shape[0]-150, 550:frame.shape[1]-450]
            framecopy = frame.copy()

            #backup
            croppig = croppic(frame)
            
            #detect defect
            knotimg,detectedknotcroped,deathknotcontour = detechknot(croppig)
            crackimg,keypoint = detectCrack(detectedknotcroped)
            pinimg,pincontore = detectpinhole(detectedknotcroped)
            smallknotimg,detectedsmallknot,smallknotcontour =detechsmallknot(detectedknotcroped)
            
            #draw defect
            deathknowdrawcontoure(deathknotcontour, framecopy)
            crackdraw(keypoint, framecopy)
            pinholedrawcontoure(pincontore, framecopy)
            smallknotdrawcontoure(smallknotcontour, framecopy)
           
            #show featured video
            putvideoname(path.split("/",-1)[2], framecopy)
            cv2.imshow('framecopy',framecopy)
            
            #show ori video
            putvideoname(path.split("/",-1)[2], frame)
            cv2.imshow('video',frame)

            #whatever
            if cv2.waitKey(1) & 0XFF ==ord('q'):
                    break

#video processing-[END]--------------------------------------------------------------------------------------------------------------------------------------

#image processing---------------------------------------------------------------------------------------------------------------------------------------
              
#undersize detection for picture            
def detectundersize(img):
    #bup   
    oriimg = img.copy()
    
    #pre
    img = cv2.blur(img, (9,9))
    img = cv2.bilateralFilter(img, 1, 99, 99 )
    img = apply_brightness_contrast(img,100,80)
    cv2.imshow('test1',img)
    
    #mask and trash
    lower = np.array([220, 220, 220])
    upper = np.array([255, 255, 255])
    thresh = cv2.inRange(img, lower, upper)
    thresh = 255 - thresh
    
    #morp
    dilated = cv2.dilate(thresh, None, iterations=3)
    
    #find edge
    contours, _= cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #draw feature
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) > 50000 and cv2.contourArea(contour) < 300000 :
            approx = cv2.contourArea(contour)
            print(approx)
            cv2.rectangle(oriimg, (x, y), (x+w, y+h), (255, 0, 0), 3)
            cv2.putText(oriimg, "Status: {}".format('Undersize'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 0), 2)
        
    #return
    return oriimg     

#wood detection algo for wood img
def detect_defect_img(path):
    
    #load all the image in path
    for p in path :
        print("loading " + p)
        #pre
        frame = cv2.imread(p)
        imgheight = int (frame.shape[1]/3)
        imgwidth = int(frame.shape[0]/3)
        frame = cv2.resize(frame,(imgheight,imgwidth) )
        croppig = croppic(frame)
        #show img
        cv2.imshow('knot',detectundersize(croppig))
        cv2.waitKey(0)

#image processing-[END]--------------------------------------------------------------------------------------------------------------------------------------

#MAIN---------------------------------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
os.system("cls")

print("Wood Defect Detection")
print("[[ -------START------- ]]")

detect_defect_video(crack_video)
detect_defect_video(deadknot_video)
detect_defect_video(smallknot_video)
detect_defect_video(pinhole_video)
detect_defect_video(good_video)

print("[[ ---PROGRAM ENDED--- ]]")

#MAIN-[END]-----------------------------------------------------------------------------------------------------------------------------------------