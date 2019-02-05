import numpy as np
import glob

class roi:
    def __init__(self, _x, _y, quadSize):
        self.centerX = _x
        self.centerY = _y
        self.x = int(_x - quadSize/2)
        self.y = int(_y - quadSize/2)
        self.width = int(_x + quadSize/2)
        self.height = int(_y + quadSize/2)

    def show(self):
        print("roi: ", self.x, self.y, self.width, self.height)

    def logFormat(self):
        return "(" + str(self.x) + ":" + str(self.width) + "," + str(self.y) + ":" + str(self.height) + ")"


def getFiles(path):
    return glob.glob(path, recursive=True)


def arrPoints(totalShape, squareSize, nElements):
    diff = (totalShape - squareSize) / (nElements - 1)
    centers = [squareSize / 2]
    for i in range(0, nElements - 1):
        centers.append(centers[i] + diff)

    centers[-1] = totalShape - (squareSize / 2)
    return [int(item) for item in centers]

def getGridArray(img, quadSize=256, nCols=31, nRows=21):
    centersX = arrPoints( img.shape[1], quadSize, nCols )
    centersY = arrPoints( img.shape[0], quadSize, nRows )

    rois = []
    for i in centersX:
        for j in centersY:
            rois.append(roi(i ,j, quadSize))

    return rois

def smartOverlay(srcImg, overImg, x, y):
    #print("srcImg", srcImg.shape)
    #print("overImg", overImg.shape)

    height, width, channels = srcImg.shape
    overHeight, overWidth, overChannels = overImg.shape
    newImg = srcImg.copy()

    if(overChannels != 4):
        print('The overImg must have 4 channels!')
        return None

    for i in range (0, overHeight-1):
        for j in range (0, overWidth-1):
            if( overImg[i][j][3] != 0 and i+x < newImg.shape[1] and j+y < newImg.shape[0]):
                #print(i+x, j+y,i,j)
                newImg[i+x][j+y][0] = overImg[i][j][0]
                newImg[i+x][j+y][1] = overImg[i][j][1]
                newImg[i+x][j+y][2] = overImg[i][j][2]
    return newImg