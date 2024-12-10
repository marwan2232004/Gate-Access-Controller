from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numba import jit
from skimage.measure import find_contours
from skimage.draw import rectangle
## testing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

##############################################--Global--##############################################

additionsDataBase = []
parasitismsDataBase = []
CharDataBase = []

additionsWidth = 60
additionsHeight = 60

charWidth = 60
charHeight = 60

Threshold = 175

#############################################--Classes--##############################################
class Character:
    def __init__(self, char, template='', width=charWidth, height=charHeight, img=None):
        self.char = char
        if img is None:
            self.template = cv2.imread(template, 0)
        else:
            self.template = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        self.col_sum = np.zeros(shape=(height, width))
        self.corr = 0
        self.resize_and_calculate(height, width)

    def resize_and_calculate(self, height, width):
        # Perform resizing of the template
        dim = (height, width)
        self.template = cv2.resize(self.template, dim, interpolation=cv2.INTER_AREA)

        # Perform calculations using char_calculations function
        self.corr, self.col_sum = char_calculations(self.template, height, width)

class NotCharacter:
    def __init__(self, char, template='', width=additionsWidth, height=additionsHeight, img=None):
        self.char = char
        if img is None:
            self.template = cv2.imread(template, 0)
        else:
            self.template = img
            
        self.col_sum = np.zeros(shape=(height, width))
        self.corr = 0
        self.resize_and_calculate(height, width)

    def resize_and_calculate(self, height, width):
        # Perform resizing of the template
        dim = (height, width)
        self.template = cv2.resize(self.template, dim, interpolation=cv2.INTER_AREA)

        # Perform calculations using char_calculations function
        self.corr, self.col_sum = char_calculations(self.template, height, width)

def char_calculations(A, height, width):
    A_mean = A.mean()
    col_A = 0
    corr_A = 0
    sum_list = np.zeros(shape=(height, width))
    img_row = 0
    while img_row < height:
        img_col = 0
        while img_col < width:
            col_A += (A[img_row, img_col] - A_mean) ** 2
            sum_list[img_row][img_col] = A[img_row, img_col]
            img_col += 1
        corr_A += col_A
        col_A = 0
        img_row += 1
    return corr_A, sum_list

###############################################--DB--#################################################

def buildCharDB():
    # Letters
    global CharDataBase
    CharDataBase = []

    Alf1 = Character("alf", 'dataSet/Char/alf_1.jpg')
    Alf2 = Character("alf", 'dataSet/Char/alf_2.jpg')
    Alf3 = Character("alf", 'dataSet/Char/alf_3.jpg')
    Alf4 = Character("alf", 'dataSet/Char/alf_4.jpg')
    Alf5 = Character("alf", 'dataSet/Char/alf_5.jpg')
    Alf6 = Character("alf", 'dataSet/Char/alf_6.jpg')
    Alf7 = Character("alf", 'dataSet/Char/alf_7.png')
    Alf8 = Character("alf", 'dataSet/Char/alf_8.jpg')
    Alf9 = Character("alf", 'dataSet/Char/alf_9.jpg')
    Alf10 = Character("alf", 'dataSet/Char/alf_10.jpg')
    Beh1 = Character("beh", 'dataSet/Char/beh_1.jpg')
    Beh2 = Character("beh", 'dataSet/Char/beh_2.jpg')
    Beh3 = Character("beh", 'dataSet/Char/beh_3.jpg')
    Beh4 = Character("beh", 'dataSet/Char/beh_4.jpg')
    Beh5 = Character("beh", 'dataSet/Char/beh_5.jpg')
    Dal1 = Character("dal", 'dataSet/Char/dal_1.jpg')
    Dal2 = Character("dal", 'dataSet/Char/dal_2.jpg')
    Dal3 = Character("dal", 'dataSet/Char/dal_3.jpg')
    Dal4 = Character("dal", 'dataSet/Char/dal_4.jpg')
    Dal5 = Character("dal", 'dataSet/Char/dal_5.jpg')
    Dal6 = Character("dal", 'dataSet/Char/dal_6.jpg')
    Ein1 = Character("ein", 'dataSet/Char/ein_1.png')
    Ein2 = Character("ein", 'dataSet/Char/ein_2.png')
    Ein3 = Character("ein", 'dataSet/Char/ein_3.png')
    Fih1 = Character("fih", 'dataSet/Char/fih_1.jpg')
    Fih2 = Character("fih", 'dataSet/Char/fih_2.png')
    Gem1 = Character("gem", 'dataSet/Char/gem_1.jpg')
    Gem2 = Character("gem", 'dataSet/Char/gem_2.jpg')
    Gem3 = Character("gem", 'dataSet/Char/gem_3.jpg')
    Gem4 = Character("gem", 'dataSet/Char/gem_4.jpg')
    Gem5 = Character("gem", 'dataSet/Char/gem_5.jpg')
    Heh1 = Character("heh", 'dataSet/Char/heh_1.jpg')
    Heh2 = Character("heh", 'dataSet/Char/heh_2.png')
    Heh3 = Character("heh", 'dataSet/Char/heh_3.png')
    Kaf1 = Character("kaf", 'dataSet/Char/kaf_1.jpg')
    Kaf2 = Character("kaf", 'dataSet/Char/kaf_2.jpg')
    Kaf3 = Character("kaf", 'dataSet/Char/kaf_3.jpg')
    Kaf4 = Character("kaf", 'dataSet/Char/kaf_4.jpg')
    Kaf5 = Character("kaf", 'dataSet/Char/kaf_5.jpg')
    Kaf6 = Character("kaf", 'dataSet/Char/kaf_6.jpg')
    Kaf7 = Character("kaf", 'dataSet/Char/kaf_7.png')
    Lam1 = Character("lam", 'dataSet/Char/lam_1.png')
    Lam2 = Character("lam", 'dataSet/Char/lam_2.png')
    Lam3 = Character("lam", 'dataSet/Char/lam_3.jpg')
    Mem1 = Character("mem", 'dataSet/Char/mem_1.jpg')
    Mem2 = Character("mem", 'dataSet/Char/mem_2.jpg')
    Mem3 = Character("mem", 'dataSet/Char/mem_3.jpg')
    Mem4 = Character("mem", 'dataSet/Char/mem_4.jpg')
    Mem5 = Character("mem", 'dataSet/Char/mem_5.jpg')
    Non1 = Character("non", 'dataSet/Char/non_1.png')
    Non2 = Character("non", 'dataSet/Char/non_2.png')
    Reh1 = Character("reh", 'dataSet/Char/reh_1.png')
    Reh2 = Character("reh", 'dataSet/Char/reh_2.jpg')
    Reh3 = Character("reh", 'dataSet/Char/reh_3.jpg')
    Reh4 = Character("reh", 'dataSet/Char/reh_4.jpg')
    Reh5 = Character("reh", 'dataSet/Char/reh_5.jpg')
    Sad1 = Character("sad", 'dataSet/Char/sad_1.jpg')
    Sad2 = Character("sad", 'dataSet/Char/sad_2.jpg')
    Sad3 = Character("sad", 'dataSet/Char/sad_3.jpg')
    Sad4 = Character("sad", 'dataSet/Char/sad_4.jpg')
    Sad5 = Character("sad", 'dataSet/Char/sad_5.jpg')
    Sad6 = Character("sad", 'dataSet/Char/sad_6.jpg')
    Sen1 = Character("sen", 'dataSet/Char/sen_1.jpg')
    Sen2 = Character("sen", 'dataSet/Char/sen_2.png')
    Tah1 = Character("tah", 'dataSet/Char/tah_1.jpg')
    Tah2 = Character("tah", 'dataSet/Char/tah_2.jpg')
    Tah3 = Character("tah", 'dataSet/Char/tah_3.jpg')
    Waw1 = Character("waw", 'dataSet/Char/waw_1.jpg')
    Waw2 = Character("waw", 'dataSet/Char/waw_2.jpg')
    Waw3 = Character("waw", 'dataSet/Char/waw_3.jpg')
    Waw4 = Character("waw", 'dataSet/Char/waw_4.jpg')
    Waw5 = Character("waw", 'dataSet/Char/waw_5.jpg')
    Waw6 = Character("waw", 'dataSet/Char/waw_6.jpg')
    Waw7 = Character("waw", 'dataSet/Char/waw_7.jpg')
    Waw8 = Character("waw", 'dataSet/Char/waw_8.jpg')
    Waw9 = Character("waw", 'dataSet/Char/waw_9.jpg')
    Yeh1 = Character("yeh", 'dataSet/Char/yeh_1.jpg')
    Yeh2 = Character("yeh", 'dataSet/Char/yeh_2.jpg')


    # Numbers
    One1 = Character("1", 'dataSet/Char/one_1.jpg')
    One2 = Character("1", 'dataSet/Char/one_2.jpg')
    One3 = Character("1", 'dataSet/Char/one_3.jpg')
    One4 = Character("1", 'dataSet/Char/one_4.jpg')
    One5 = Character("1", 'dataSet/Char/one_5.jpg')
    Two1 = Character("2", 'dataSet/Char/two_1.jpg')
    Two2 = Character("2", 'dataSet/Char/two_2.jpg')
    Two3 = Character("2", 'dataSet/Char/two_3.jpg')
    Two4 = Character("2", 'dataSet/Char/two_4.jpg')
    Two5 = Character("2", 'dataSet/Char/two_5.jpg')
    Three1 = Character("3", 'dataSet/Char/three_1.jpg')
    Three2 = Character("3", 'dataSet/Char/three_2.jpg')
    Three3 = Character("3", 'dataSet/Char/three_3.jpg')
    Three4 = Character("3", 'dataSet/Char/three_4.jpg')
    Three5 = Character("3", 'dataSet/Char/three_5.jpg')
    Four1 = Character("4", 'dataSet/Char/four_1.jpg')
    Four2 = Character("4", 'dataSet/Char/four_2.jpg')
    Four3 = Character("4", 'dataSet/Char/four_3.jpg')
    Four4 = Character("4", 'dataSet/Char/four_4.jpg')
    Four5 = Character("4", 'dataSet/Char/four_5.jpg')
    Five1 = Character("5", 'dataSet/Char/five_1.jpg')
    Five2 = Character("5", 'dataSet/Char/five_2.jpg')
    Five3 = Character("5", 'dataSet/Char/five_3.jpg')
    Five4 = Character("5", 'dataSet/Char/five_4.jpg')
    Five5 = Character("5", 'dataSet/Char/five_5.jpg')
    Six1 = Character("6", 'dataSet/Char/six_1.jpg')
    Six2 = Character("6", 'dataSet/Char/six_2.jpg')
    Six3 = Character("6", 'dataSet/Char/six_3.jpg')
    Six4 = Character("6", 'dataSet/Char/six_4.jpg')
    Seven1 = Character("7", 'dataSet/Char/seven_1.jpg')
    Seven2 = Character("7", 'dataSet/Char/seven_2.jpg')
    Seven3 = Character("7", 'dataSet/Char/seven_3.jpg')
    Seven4 = Character("7", 'dataSet/Char/seven_4.jpg')
    Seven5 = Character("7", 'dataSet/Char/seven_5.jpg')
    Eight1 = Character("8", 'dataSet/Char/eight_1.jpg')
    Eight2 = Character("8", 'dataSet/Char/eight_2.jpg')
    Eight3 = Character("8", 'dataSet/Char/eight_3.jpg')
    Eight4 = Character("8", 'dataSet/Char/eight_4.jpg')
    Nine1 = Character("9", 'dataSet/Char/nine_1.jpg')
    Nine2 = Character("9", 'dataSet/Char/nine_2.jpg')
    Nine3 = Character("9", 'dataSet/Char/nine_3.jpg')
    Nine4 = Character("9", 'dataSet/Char/nine_4.jpg')
    Nine5 = Character("9", 'dataSet/Char/nine_5.jpg')


	# Add to database
    # Append Alf instances
    CharDataBase.append(Alf1)
    CharDataBase.append(Alf2)
    CharDataBase.append(Alf3)
    CharDataBase.append(Alf4)
    CharDataBase.append(Alf5)
    CharDataBase.append(Alf6)
    CharDataBase.append(Alf7)
    CharDataBase.append(Alf8)
    CharDataBase.append(Alf9)
    CharDataBase.append(Alf10)
    
    # Append Beh instances
    CharDataBase.append(Beh1)
    CharDataBase.append(Beh2)
    CharDataBase.append(Beh3)
    CharDataBase.append(Beh4)
    CharDataBase.append(Beh5)

    # Append Dal instances
    CharDataBase.append(Dal1)
    CharDataBase.append(Dal2)
    CharDataBase.append(Dal3)
    CharDataBase.append(Dal4)
    CharDataBase.append(Dal5)
    CharDataBase.append(Dal6)

    # Append Ein instances
    CharDataBase.append(Ein1)
    CharDataBase.append(Ein2)
    CharDataBase.append(Ein3)

    # Append Fih instances
    CharDataBase.append(Fih1)
    CharDataBase.append(Fih2)

    # Append Gem instances
    CharDataBase.append(Gem1)
    CharDataBase.append(Gem2)
    CharDataBase.append(Gem3)
    CharDataBase.append(Gem4)
    CharDataBase.append(Gem5)

    # Append Heh instances
    CharDataBase.append(Heh1)
    CharDataBase.append(Heh2)
    CharDataBase.append(Heh3)

    # Append Kaf instances
    CharDataBase.append(Kaf1)
    CharDataBase.append(Kaf2)
    CharDataBase.append(Kaf3)
    CharDataBase.append(Kaf4)
    CharDataBase.append(Kaf5)
    CharDataBase.append(Kaf6)
    CharDataBase.append(Kaf7)

    # Append Lam instances
    CharDataBase.append(Lam1)
    CharDataBase.append(Lam2)
    CharDataBase.append(Lam3)

    # Append Mem instances
    CharDataBase.append(Mem1)
    CharDataBase.append(Mem2)
    CharDataBase.append(Mem3)
    CharDataBase.append(Mem4)
    CharDataBase.append(Mem5)

    # Append Non instances
    CharDataBase.append(Non1)
    CharDataBase.append(Non2)

    # Append Reh instances
    CharDataBase.append(Reh1)
    CharDataBase.append(Reh2)
    CharDataBase.append(Reh3)
    CharDataBase.append(Reh4)
    CharDataBase.append(Reh5)

    # Append Sad instances
    CharDataBase.append(Sad1)
    CharDataBase.append(Sad2)
    CharDataBase.append(Sad3)
    CharDataBase.append(Sad4)
    CharDataBase.append(Sad5)
    CharDataBase.append(Sad6)

    # Append Sen instances
    CharDataBase.append(Sen1)
    CharDataBase.append(Sen2)

    # Append Tah instances
    CharDataBase.append(Tah1)
    CharDataBase.append(Tah2)
    CharDataBase.append(Tah3)

    # Append Waw instances
    CharDataBase.append(Waw1)
    CharDataBase.append(Waw2)
    CharDataBase.append(Waw3)
    CharDataBase.append(Waw4)
    CharDataBase.append(Waw5)
    CharDataBase.append(Waw6)
    CharDataBase.append(Waw7)
    CharDataBase.append(Waw8)
    CharDataBase.append(Waw9)

    # Append Yeh instances
    CharDataBase.append(Yeh1)
    CharDataBase.append(Yeh2)
    
    # Append One instances
    CharDataBase.append(One1)
    CharDataBase.append(One2)
    CharDataBase.append(One3)
    CharDataBase.append(One4)
    CharDataBase.append(One5)
    
    # Append Two instances
    CharDataBase.append(Two1)
    CharDataBase.append(Two2)
    CharDataBase.append(Two3)
    CharDataBase.append(Two4)
    CharDataBase.append(Two5)
    
    # Append Three instances
    CharDataBase.append(Three1)
    CharDataBase.append(Three2)
    CharDataBase.append(Three3)
    CharDataBase.append(Three4)
    CharDataBase.append(Three5)
    
    # Append Four instances
    CharDataBase.append(Four1)
    CharDataBase.append(Four2)
    CharDataBase.append(Four3)
    CharDataBase.append(Four4)
    CharDataBase.append(Four5)
    
    # Append Five instances
    CharDataBase.append(Five1)
    CharDataBase.append(Five2)
    CharDataBase.append(Five3)
    CharDataBase.append(Five4)
    CharDataBase.append(Five5)
    
    # Append Six instances
    CharDataBase.append(Six1)
    CharDataBase.append(Six2)
    CharDataBase.append(Six3)
    CharDataBase.append(Six4)
    
    # Append Seven instances
    CharDataBase.append(Seven1)
    CharDataBase.append(Seven2)
    CharDataBase.append(Seven3)
    CharDataBase.append(Seven4)
    CharDataBase.append(Seven5)
    
    # Append Eight instances
    CharDataBase.append(Eight1)
    CharDataBase.append(Eight2)
    CharDataBase.append(Eight3)
    CharDataBase.append(Eight4)
    
    # Append Nine instances
    CharDataBase.append(Nine1)
    CharDataBase.append(Nine2)
    CharDataBase.append(Nine3)
    CharDataBase.append(Nine4)
    CharDataBase.append(Nine5)
    
def buildAdditionsDB():
	global additionsDataBase 
	additionsDataBase = []
	hamza = NotCharacter('hamza','dataSet/Additions/hamza.jpg')
	no2taB_1 = NotCharacter('no2taB','dataSet/Additions/no2taBeh_1.jpg')
	no2taB_2 = NotCharacter('no2taB','dataSet/Additions/no2taBeh_2.jpg')
	no2taB_3 = NotCharacter('no2taB','dataSet/Additions/no2taBeh_3.jpg')
	no2taN_1 = NotCharacter('no2taN','dataSet/Additions/no2taNoon_1.jpg')
	no2taN_2 = NotCharacter('no2taN','dataSet/Additions/no2taNoon_2.jpg')
	no2taG = NotCharacter('no2taG','dataSet/Additions/no2taGem_1.jpg')
	additionsDataBase.append(hamza)
	additionsDataBase.append(no2taB_1)
	additionsDataBase.append(no2taB_2)
	additionsDataBase.append(no2taB_3)
	additionsDataBase.append(no2taN_1)
	additionsDataBase.append(no2taN_2)
	additionsDataBase.append(no2taG)

def buildParasitismsDB():
    global parasitismsDataBase 
    parasitismsDataBase = []
    bar1 = NotCharacter('bar', 'dataSet/Additions/bar1.jpg')
    bar2 = NotCharacter('bar', 'dataSet/Additions/bar2.jpg')
    bar3 = NotCharacter('bar', 'dataSet/Additions/bar3.jpg')
    bar4 = NotCharacter('bar', 'dataSet/Additions/bar4.jpg')
    nesr1 = NotCharacter('nesr', 'dataSet/Additions/nesr1.jpg')
    nesr2 = NotCharacter('nesr', 'dataSet/Additions/nesr2.jpg')
    nesr3 = NotCharacter('nesr', 'dataSet/Additions/nesr3.jpg')
    parasitismsDataBase.append(bar1)
    parasitismsDataBase.append(bar2)
    parasitismsDataBase.append(bar3)
    parasitismsDataBase.append(bar4)
    parasitismsDataBase.append(nesr1)
    parasitismsDataBase.append(nesr2)
    parasitismsDataBase.append(nesr3)

#############################################--Utilts--###############################################

# Preprocessing steps for the image before plate detection
def pre_process_image(image_path):
    # Read the image from the specified path
    car = cv2.imread(image_path)
    
    # Resize the image to a specific size (1200x800 pixels)
    resized_image = cv2.resize(car, (1200, 800))
    
    # Convert the resized image to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter for noise removal while keeping the edges sharp
    noise_removal = cv2.bilateralFilter(gray, 11, 17, 17)
    
    return noise_removal

def calculate_area(image_array):
    height, width = image_array[0].shape
    return width * height

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0

	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True

	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1

	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))

	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

@jit(nopython = True)
def cal_corr(corr_A, corr_B, A_sum, B_sum):
    corr_both = np.multiply(A_sum, B_sum)
    corr_both = corr_both.sum()
    r = corr_both / math.sqrt(corr_A * corr_B)
    return r


def isAdditionLetter(imgI):
    letter = NotCharacter('unk', img=imgI)

    for l in additionsDataBase:
        temp1 = letter.template.astype(np.float32)
        temp2 = l.template.astype(np.float32)

        hist1 = cv2.calcHist([temp1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([temp2], [0], None, [256], [0, 256])

        r = cv2.compareHist(hist1, hist2, method=cv2.HISTCMP_CORREL)
        rCorr = cal_corr(letter.corr, l.corr, letter.col_sum, l.col_sum)

        if rCorr > 0.75 and r > 0.5:
            return True

    return False

 
def isBar(imgI):
    letter = NotCharacter('unk',img = imgI)
    for l in parasitismsDataBase:
        temp1 = letter.template.astype(np.float32)
        temp2 = l.template.astype(np.float32)
        hist1=0
        hist2=0
        hist1=cv2.calcHist([temp1],[0],None,[256],[0,256]) 
        hist2=cv2.calcHist([temp2],[0],None,[256],[0,256]) 
        r = cv2.compareHist(hist1, hist2, method = cv2.HISTCMP_CORREL)
        rCorr = cal_corr(letter.corr,l.corr,letter.col_sum,l.col_sum)
        if(rCorr>.85 and r > .85):
            return True
    return False

@jit(nopython = True)
def intersection(a,b):
    a = list(a)
    b = list(b)
    var = 16
    a[0]-=var
    a[1]-=var
    a[2]+=var
    a[3]+=var
    b[0]-=var
    b[1]-=var
    b[2]+=var
    b[3]+=var
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return False 
    return True


def platePreProcess(plate):
    dim = (1404, 446)
    plate = cv2.resize(plate, dim, interpolation=cv2.INTER_AREA)
    height, width = plate.shape[:2]
    start_x = (width - 50) // 2
    end_x = start_x + 50
    plate[:, start_x:end_x] = 255 
    start_x = width - 30
    end_x = width
    plate[:, start_x:end_x] = 255 
    start_x = 0
    end_x = 10
    plate[:, start_x:end_x] = 255 
    start_x = height - 20
    end_x = height
    plate[start_x:end_x, :] = 255 
    start_x = 0
    end_x = 5
    plate[start_x:end_x, :] = 255 
    
    return plate  # Return the modified image



#########################################--Core Functions--############################################
def plate_detection_using_contours(path):
    car = pre_process_image(path)
    found = False
    thresh = 180
    plate = []

    while found == False and thresh >= 0:
        ret, bin_img = cv2.threshold(car,thresh,255,cv2.THRESH_BINARY)
        open1 = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        close1 = cv2.morphologyEx(open1, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        close2 = cv2.morphologyEx(close1, cv2.MORPH_CLOSE, np.ones((12,12),np.uint8))
        open2 = cv2.morphologyEx(close2, cv2.MORPH_OPEN, np.ones((12,12),np.uint8))
        
        # Find contours in the edged image
        contours = find_contours(open2)
        bounding_boxes = []
        plate_suspects = []

        # Iterate through the detected contours
        for contour in contours:
            width = contour[:, 1].max() - contour[:, 1].min()
            height = contour[:, 0].max() - contour[:, 0].min()
            aspect = width / height
            if 2 < aspect < 6.5:
                bounding_boxes.append([contour[:, 1].min(), contour[:, 1].max(), contour[:, 0].min(), contour[:, 0].max()])
        img_with_boxes = np.zeros(shape=bin_img.shape)

        # Get the bounding rectangle of the plate contour
        for box in bounding_boxes:
            [Xmin, Xmax, Ymin, Ymax] = box
            if Xmin > 300 and Xmax < 900 and Ymin > 150 and Ymax < 650:
                rr, cc = rectangle(start = (Ymin,Xmin), end = (Ymax,Xmax), shape=bin_img.shape)
                plate_suspects.append([close1[rr, cc], np.rot90(np.fliplr(car[rr, cc]), k=1)])
                img_with_boxes[rr, cc] = 255 #set color white

        found_con_length = 3
        plate_suspects = sorted(plate_suspects, key=calculate_area)
        for plate_suspect in plate_suspects:
            contours = find_contours(plate_suspect[0])
            if found_con_length < len(contours):
                plate = plate_suspect[1]
                found_con_length = len(contours)
                found = True
            else:
                continue
        thresh -= 10

    if found:
        return plate
    else:
        return car



def PlateToLetters(plate):
    plate = platePreProcess(plate)
    blurPlate = cv2.blur(plate,(10,10))
    blurPlate = cv2.blur(blurPlate,(10,10))
    medianPlate= cv2.medianBlur(blurPlate,5)    
    _, thresholdPlate = cv2.threshold(medianPlate,Threshold,255,cv2.THRESH_BINARY)    
    # Create a custom kernel representing a 45-degree oval
    angle = 45  # Angle for the oval (in degrees)
    kernel_size = (25, 60)  # Size of the kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Rotate the rectangular kernel to create a 45-degree oval-shaped kernel
    center = (kernel_size[0] // 2, kernel_size[1] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_kernel = cv2.warpAffine(kernel, rotation_matrix, kernel_size)
    
    
    kernel1=cv2.getStructuringElement(cv2.MORPH_RECT,(20,30))
    kernel2=cv2.getStructuringElement(cv2.MORPH_RECT,(7,16))
    
    
    # Split the plate image into two halves
    half_width = plate.shape[1] // 2
    left_half = thresholdPlate[:, :half_width]  # Left half of the plate image
    right_half = thresholdPlate[:, half_width:]  # Right half of the plate image

    # Apply morphological operations to each half separately
    close_left = cv2.morphologyEx(left_half, cv2.MORPH_CLOSE, kernel2)
    open_left = cv2.morphologyEx(close_left, cv2.MORPH_OPEN, kernel1)

    close_right = cv2.morphologyEx(right_half, cv2.MORPH_CLOSE, kernel2)
    open_right = cv2.morphologyEx(close_right, cv2.MORPH_OPEN, rotated_kernel)

    # Concatenate the processed halves back together
    result_plate = np.hstack((open_left, open_right))
    contours, _ = cv2.findContours(result_plate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sort_contours(contours)[0]


    letters = []
    rects = []
 
    T = False
    for contour in contours:
        x,y,w,h = rect = cv2.boundingRect(contour)

        if(y>30 and x > 10 and w < 250 and y+h<750 and cv2.contourArea(contour) > 3000 and y < 575 and cv2.contourArea(contour) <100000):
            for index,r in enumerate(rects):
                if((x-20<r[0] and x+w+20 > r[0]+r[2]) or (x+20>r[0] and x+w-20 < r[0]+r[2])):
                    T = True
                    miniImg = np.copy(plate[y:y+h,x:x+h])
                    if miniImg is not None:
                        if(isAdditionLetter(miniImg)):
                            minY = min(y,r[1])                    
                            maxH = max(y+h,r[1]+r[3])-minY
                            minX = min(x,r[0])                    
                            maxW = max(x+w,r[0]+r[2])-minX
                            rects[index] = (minX,minY,maxW,maxH)
                            break
                if (intersection(rect,r)):
                    T = True
                    minY = min(y,r[1])                    
                    maxH = max(y+h,r[1]+r[3])-minY
                    minX = min(x,r[0])                    
                    maxW = max(x+w,r[0]+r[2])-minX
                    rects[index] = (minX,minY,maxW,maxH)
                    break
            if(T):
                T = False
                continue
            rects.append(rect)

    for rect in rects:
        imgX = None
        imgX = np.copy(plate[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]])
        if imgX is not None:
            letters.append(imgX)

            # cv2.imshow('Image', imgX)
            # cv2.waitKey(0)
            # if(not isBar(imgX)):
            #     letters.append(imgX)
    return letters

def extract_features(letters):
    letterFeatures = []

    for letter in letters:
        # Perform resizing of the template
        ret, letter = cv2.threshold(letter,140,255,cv2.THRESH_BINARY)
        dim = (charHeight, charWidth)
        letter = cv2.resize(letter, dim, interpolation=cv2.INTER_AREA)
        corr, col_sum = char_calculations(letter, charHeight, charWidth)
        flattened_col_sum = col_sum.flatten()
        letterFeatures.append(flattened_col_sum)

    return letterFeatures


##############################################--KNN--################################################
# Extract features and labels from your CharDataBase
features = []  # Add the features you want to use for similarity
labels = []    # Add the corresponding labels

# Initialize the KNN classifier
k = 3
knn = KNeighborsClassifier(n_neighbors=3,p=2,metric='euclidean')

def testKnn():
    # Split the data into training and testing sets
    train_input, test_input, train_output, test_output = train_test_split(features, labels, test_size=20, random_state=203)

    # Train the classifier
    knn.fit(train_input, train_output)
    # Predict using the trained classifier
    predictions = knn.predict(test_input)

    # Evaluate the accuracy
    accuracy = accuracy_score(test_output, predictions)
    print("Accuracy:", accuracy)

def trainKnn():
    # Train the classifier
    knn.fit(features, labels)


def predictKnn(letter):
    prediction = knn.predict(letter)
    return prediction
    

def extract_features(letters):
    letterFeatures = []

    for letter in letters:
        # Perform resizing of the template
        ret, letter = cv2.threshold(letter,Threshold,255,cv2.THRESH_BINARY)
        dim = (charHeight, charWidth)
        letter = cv2.resize(letter, dim, interpolation=cv2.INTER_AREA)
        corr, col_sum = char_calculations(letter, charHeight, charWidth)
        flattened_col_sum = col_sum.flatten()
        letterFeatures.append(flattened_col_sum)

    return letterFeatures



##############################################--Main--################################################
def main(path):
    
    buildCharDB()
    buildAdditionsDB()
    buildParasitismsDB()
    
    global Threshold
    Threshold=175

    for char_instance in CharDataBase:
        # Assuming col_sum is a 2D array, flatten it to 1D
        flattened_col_sum = char_instance.col_sum.flatten()
        
        # Concatenate or combine features as needed
        #combined_features = np.concatenate([flattened_col_sum])
        combined_features = flattened_col_sum
        
        # Append combined features and label to lists
        features.append(combined_features)
        labels.append(char_instance.char)

    plate = plate_detection_using_contours(path)
    
    cv2.imwrite('plate.jpg',plate)
    
    
    letters = PlateToLetters(plate)
    while(len(letters)<2 and Threshold>90):
        Threshold-=75
        letters = PlateToLetters(plate)
    

    #extract features from the resulting letters
    letterFeatures = extract_features(letters)

    #train the knn then predict the letters
    trainKnn()

    return predictKnn(letterFeatures)
