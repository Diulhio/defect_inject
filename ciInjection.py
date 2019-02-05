import cv2
from commonFuncs import *

imgCi = "./ci1.png"

def executeCiInjection(img, x, y, roiWidth, roiHeight):
    roiX = int(x - roiWidth / 2)
    if roiX < 0:
        roiX = 0
    roiY = int(y - roiHeight / 2)
    if roiY < 0:
        roiY = 0
    roiWid = int(roiX + roiWidth)
    roiHei = int(roiY + roiHeight)
    imgRoi = img[roiY:roiHei, roiX:roiWid]

    overImg = cv2.imread(imgCi, cv2.IMREAD_UNCHANGED)

    #cv2.imshow("overimg", overImg)
    #cv2.imshow("roi", imgRoi)
    #cv2.waitKey(0)

    newImg = smartOverlay(imgRoi, overImg, 0, 0 )

    #cv2.imshow("new", newImg)
    #cv2.waitKey(0)

    img[roiY:roiHei, roiX:roiWid] = newImg

    return img, 1
    # def smartOverlay(srcImg, overImg, x, y):