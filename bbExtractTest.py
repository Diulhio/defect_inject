import cv2
from commonFuncs import *

coefficient = 0.25
roiWidth = 900
roiHeight = 300

def clickRoi(event, x, y, flags, param):
    #print (event)
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        roiX = int((x/coefficient) - roiWidth/2 )
        roiY = int((y/coefficient) - roiHeight/2 )
        roiXEnd = int(roiX + roiWidth)
        roiYEnd = int(roiY + roiHeight)
        roiImg = img[ roiY:roiYEnd , roiX:roiXEnd]


        cv2.imshow("roi", roiImg)
        cv2.waitKey(0)


srcPath = "/home/diulhio/Codigos/pcbs/clean_ci/2/"
file_type = "*"

files = getFiles(srcPath + file_type)

cv2.namedWindow("original")
cv2.setMouseCallback("original", clickRoi)

for imgFile in files:
    img = cv2.imread(imgFile)
    imgResize = cv2.resize(img, None, fx=coefficient, fy=coefficient, interpolation=cv2.INTER_CUBIC)
    cv2.imshow("original", imgResize)
    k = cv2.waitKey(0)
    if k == ord('q'):
        exit()
