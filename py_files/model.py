import cv2
import numpy as np
import psycopg2
from sklearn.externals import joblib
import glob
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import Reshape, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
import cv2
import pandas as pd
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization
import h5py
import itertools
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from keras.layers.convolutional import *
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

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
########################## Classification #############################
#takes a list of segmented images and classify noise/char/digits
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
   ######################
     











        
############################### SVM Classification #####################################        
def recoginition():
    chars = glob.glob('C:/Segmented/char/*')
    digits = glob.glob('C:/Segmented/digit/*')
    char = []
    digit =[]
    classdigit = joblib.load('classdigit.pkl')
    classchar = joblib.load('classchar.pkl')
    encoder = LabelEncoder()
    encoder.classes_ = np.load('char_enc.npy')
    for i in range(len(chars)):
        image = cv2.imread(chars[i])
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(50,50))
        img = img.flatten().reshape(1,-1)
        pred = classchar.predict(img)
        c = encoder.inverse_transform(np.array([pred]))[0]
        char.append(c[0])
        
    for i in range(len(digits)):
        image = cv2.imread(digits[i])
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(50,50))
        img = img.flatten().reshape(1,-1)
        pred = classdigit.predict(img)
        digit.append(pred[0])
        
    if len(char)>=3:
        char=[char[0].upper(),char[1].upper(),char[2].upper()]
    else:
        char=[""]
    if len(digit)>=4:
        digit=[digit[0],digit[1],digit[2],digit[3]]
    else:
        digit=digit
        
    return char,digit











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
 ##########
 









img_width, img_height = 150, 150





from keras.layers.convolutional import *
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


# # Arabic Numbers


def model_loading():
    
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=(img_width, img_height,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(17))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    model.load_weights('models/arabic_letters_epoch_10.h5')
    return model




def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    

    return img_tensor





arabic = ['أ','ب','د','ع','ق','ه','ح','ك','ل','ن','ر','س','ط','و','ي','ص','م']
def arabic_letters(img_folder_path):
    i = -1
    arr = []
    for im in os.listdir(img_folder_path):
        i= i + 1
        new_image = load_image(img_folder_path+'/{0}.jpg'.format(i))
        model=model_loading()
        pred = model.predict(new_image)
        array = pred
        result= array[0]
        answer = np.argmax(result)
        for m in range(17):
            if answer == m:
                arr.append(arabic[m])
    return arr


def model_loading2():
    
    model2 = Sequential()
    model2.add(Convolution2D(32, (3, 3), input_shape=(img_width, img_height,3)))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))

    model2.add(Convolution2D(32, (3, 3)))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))

    model2.add(Convolution2D(64, (3, 3)))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))

    model2.add(Flatten())
    model2.add(Dense(64))
    model2.add(Activation('relu'))
    model2.add(Dropout(0.5))
    model2.add(Dense(9))
    model2.add(Activation('softmax'))
    model2.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    model2.load_weights('models/model_epoch_10.h5')
    return model2
    





def arabic_numbers(img_folder_path):
    i = -1
    arr = []
    for im in os.listdir(img_folder_path):
        i= i + 1
        new_image = load_image(img_folder_path+'/{0}.jpg'.format(i))
        model2=model_loading2()
        pred = model2.predict(new_image)
        array = pred
        result= array[0]
        answer = np.argmax(result)
        for m in range(9):
            if answer == m:
                arr.append(m+1)
    return arr



def model_loading3():
    
    model3 = Sequential()
    model3.add(Convolution2D(32, (3, 3), input_shape=(img_width, img_height,3)))
    model3.add(Activation('relu'))
    model3.add(MaxPooling2D(pool_size=(2, 2)))

    model3.add(Convolution2D(32, (3, 3)))
    model3.add(Activation('relu'))
    model3.add(MaxPooling2D(pool_size=(2, 2)))

    model3.add(Convolution2D(64, (3, 3)))
    model3.add(Activation('relu'))
    model3.add(MaxPooling2D(pool_size=(2, 2)))

    model3.add(Flatten())
    model3.add(Dense(64))
    model3.add(Activation('relu'))
    model3.add(Dropout(0.5))
    model3.add(Dense(1))
    model3.add(Activation('sigmoid'))
    model3.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    model3.load_weights('models/noise_basic_cnn_epochs_10.h5')
    return model3






def noise(img_folder_path):
    i = -1
    arr = []
    ind = []
    for im in os.listdir(img_folder_path):
        i= i + 1
        new_image = load_image(img_folder_path+'/{0}.jpg'.format(i)) 
        model3=model_loading3()
        m ='{0}.jpg'.format(i)
        pred = model3.predict(new_image)
        array = pred[0]
        if array > .5:
            arr.append(m)
            ind.append(i)
    #print("Noise :")
    return ind




#ch = 'c:/Segmented/char'
#di = 'c:/Segmented/digit'
#arabic_numbers(di)
#arabic_letters(no)

def final_fun(ch, di):
    digit_arr = noise(di)
    char_arr = noise(ch)
    i,l = 0,0
    number_pred = arabic_numbers(di)
    char_pred = arabic_letters(ch)
    if len(char_arr) > 0 :
        for i in range(len(char_arr)):
            char_pred.remove(char_pred[char_arr[i]])
    temp = []
    if len(digit_arr) > 0 :     
        for l in range(len(digit_arr)):
            temp.append(digit_arr[l])
        temp.reverse()
        for j in range(len(temp)):
            x = temp[j]
            del number_pred[x]
    char_pred.reverse()
    final_ = number_pred + char_pred
    return final_



def final(ch, di):
    digit_arr = noise(di)
    char_arr = noise(ch)
    i,l = 0,0
    number_pred = arabic_numbers(di)
    char_pred = arabic_letters(ch)
    if len(char_arr) > 0 :
        for i in range(len(char_arr)):
            char_pred.remove(char_pred[char_arr[i]])
    temp = []
    if len(digit_arr) > 0 :     
        for l in range(len(digit_arr)):
            temp.append(digit_arr[l])
        temp.reverse()
        for j in range(len(temp)):
            x = temp[j]
            del number_pred[x]
    char_pred.reverse()
    if (len(number_pred) > 4):
        number_pred = number_pred[0:4]
    if (len(char_pred) > 3):
        char_pred = char_pred[0:3]
    final_ = number_pred + char_pred
    return final_


