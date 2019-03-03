import cv2
import numpy as np
import psycopg2
from sklearn.externals import joblib

################################## Resize Images ##############################
def image_resize(image, width = 250, height = 125, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    h, w = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
################################## Detect Plate ###############################
def detect_plate(image):
    #print(image)
    image=cv2.imread(image)
    if image is None:
        return("Image doesn't exist");
    res_image = image_resize(image)
    img_gray = cv2.cvtColor(res_image,cv2.COLOR_RGB2GRAY)
    retval,binary_original=cv2.threshold(img_gray,128,255,cv2.THRESH_OTSU)
    noise_removal = cv2.bilateralFilter(binary_original,10,50,50)
    equal_histogram = cv2.equalizeHist(noise_removal)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(300,500))
    morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=1)
    sub_morp_image = cv2.subtract(equal_histogram,morph_image)
    ret,thresh_image = cv2.threshold(sub_morp_image,50,255,cv2.THRESH_OTSU)
    canny_image = cv2.Canny(thresh_image,150,255)
    canny_image = cv2.convertScaleAbs(canny_image)
    kernel = np.ones((4,4), np.uint8)
    dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
    _,contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    # Sort the contours based on area ,so that the number plate will be in top 10 contours
    screenCnt = None
    # loop over our contours
    for c in contours:
    	# approximate the contour
    	peri = cv2.arcLength(c, True)
    	approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Approximating with 6% error
    	# if our approximated contour has four points, then
    	# we can assume that we have found our screen
    	if len(approx) == 4:  # Select the contour with 4 corners
    		screenCnt = approx
    		break
    if  screenCnt is not None:
        final = cv2.drawContours(res_image, [screenCnt], -1, (255, 255, 255), 1)
    
    (x, y, w, h) = cv2.boundingRect(screenCnt)
    if final is not None:
        im = final[y:y+h,x:x+w]
        if h != 0 or w != 0:
            im = cv2.resize(im,(100,50))
            cv2.imwrite('C:/seg/img/segmented.png', im)
            return "C:/seg/img/segmented.png";
        
################################## Segment Detected PLate #####################
def segment(img):
    img = cv2.imread(img, 0)
    cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,img)
    image, contours, hier = cv2.findContours(img, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    d=0
    threshold_area = 20  #threshold area 
    l=[]
    for ctr in contours:
        area = cv2.contourArea(ctr)  
        if area > threshold_area:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)
            # Getting ROI
            roi = image[y:y+h, x:x+w]
            cv2.imwrite('C:/segmented/character_{}.png'.format(d), roi)
            l.append('C:/segmented/character_{}.png'.format(d))
            d+=1
    return l
################################## Classification #############################
#takes a list of segmented images and return a list of letters images   
def classify_noise(segmented_list):
    classnoise = joblib.load('classnoise.pkl')
    #print(classnoise)
    letters = []
    for i in range(len(segmented_list)):
        im = cv2.imread(segmented_list[i],0)
        #print(im.shape)
        if im is None:
            continue
        img = cv2.resize(im,(50,50))
        #print(img.shape)
        img = img.flatten().reshape(1,-1)
        #print(img.shape)
        pred = classnoise.predict(img)
        if pred == 0:
            letters.append(im)
            
    digit=0
    char=0
    classtext = joblib.load('classtext.pkl')
    for i in range(len(letters)):
        img= cv2.resize(letters[i],(50,50))
        img = img.flatten().reshape(1,-1)
        pred = classtext.predict(img)
        if pred == 0:
            cv2.imwrite("C:/Segmented/digit/"+str(digit)+'.jpg',letters[i])
            digit +=1
        else:
            cv2.imwrite("C:/Segmented/char/"+str(char)+'.jpg',letters[i])
            char +=1
    
    for i in range(len(letters)):
        p = "C:/Segmented/noise/"+str(i)+'.jpg'
        cv2.imwrite(p,letters[i])
        












###################### Get PlateNumber to Search in DB ########################
def get_pn(l):
    if len(l)==7:
        pn="'{}{}{}{} {} {} {}'".format(l[0],l[1],l[2],l[3],l[-3],l[-2],l[-1])
    elif len(l)==6:
        pn="'{}{}{} {} {} {}'".format(l[0],l[1],l[2],l[-3],l[-2],l[-1])
    elif len(l)==5:
        pn="'{}{} {} {} {}'".format(l[0],l[1],l[-3],l[-2],l[-1])
    elif len(l)==4:
        pn="'{} {} {} {}'".format(l[0],l[-3],l[-2],l[-1])
    else:
        pn="''"
    return pn

#################### Check if PlateNumber stored in sys. or Not################
def check_db(pn):
    
     connection = psycopg2.connect(user = "postgres",password = "123456789",
                                       host = "127.0.0.1",
                                       port = "5432",
                                       database = "plates")

     cursor = connection.cursor()
     postgreSQL_select_Query = "select * from plates where license_plate=N{}".format(pn)
        
     cursor.execute(postgreSQL_select_Query)
     plate_number = cursor.fetchall() 
     if plate_number ==[]:
         retval=False
     else:
         #plate_number=plate_number[0][0]
         retval=True
     cursor.close()
     connection.close()
     return retval