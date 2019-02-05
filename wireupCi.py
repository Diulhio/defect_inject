import cv2
import os
import numpy as np
import imutils
from random import *
from scipy.spatial import distance
from commonFuncs import *
import math

imgHighAngle = "./wireup_ci.png"
imgLowAngle = "./wireup_reto_ci.png"
img90Angle = "./wireup_90_ci.png"
img0Angle = "./wireup_0_ci.png"


def findTerminals(img, lowMean, highMean, devPad, minArea):
    mask = cv2.inRange(img, lowMean, highMean)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            media = np.mean(img[y, x])
            desvpad = np.sqrt(np.var(img[y, x]))
            if media > lowMean and media < highMean and desvpad < devPad:
                mask[y, x] = 255
            else:
                mask[y, x] = 0

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    thCnts = []

    for c in cnts:
        shapeArea = cv2.contourArea(c)
        if (shapeArea > minArea):
            thCnts.append(c)

    return mask, thCnts


def cntCentroid(cnt):
    M = cv2.moments(cnt)
    cX = cY = 0
    if (M["m00"] != 0):
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))

    return cX, cY


def detectGroupTerminals(img, cnts, deltaY, maxDistance, minDistance, minGroup):
    groups = []
    inGroup = []

    # print("Qnt contornos: ", len(cnts))
    for index, c in enumerate(cnts):  ## c -> contorno referencia
        tempGroup = [c]
        ## Centroide do contorno atual
        cX, cY = cntCentroid(c)

        if (index not in inGroup):
            for cnGroup in tempGroup:
                debugImg = img.copy()
                cnGroupX, cnGroupY = cntCentroid(cnGroup)

                cv2.drawContours(debugImg, [cnGroup], -1, (255, 0, 0), 2)

                for cn in range(0, len(cnts)):

                    if np.array_equal(cnts[cn],
                                      c) or cn in inGroup:  ## Ignora contorno referencia ou contornos que ja estejam em outros grupos
                        pass
                    else:
                        cnX, cnY = cntCentroid(cnts[cn])

                        cv2.drawContours(debugImg, [cnts[cn]], -1, (255, 255, 255), 2)
                        euDistance = distance.euclidean([cnX, cnY], [cnGroupX, cnGroupY])
                        if (np.abs(cnGroupY - cnY) < deltaY and euDistance < maxDistance and euDistance > minDistance):
                            tempGroup.append(cnts[cn])
                            inGroup.append(cn)
                            # print("Inserido no grupo! Distancia: ", euDistance)
                            cv2.drawContours(debugImg, [cnts[cn]], -1, (0, 0, 255), 2)
                        # break

                        # cv2.drawContours(debugImg, [cnGroup], -1, (255, 0, 0), 2)
                        '''
                        cv2.imshow("debug", debugImg)
                        cv2.waitKey(0)
                        '''

            if len(tempGroup) >= minGroup:
                # print(len(tempGroup))
                # input(".")
                inGroup.append(index)
                groups.append(tempGroup)

    return groups


def findTerminalsByThreshold(img, th, minArea):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = cv2.erode(thresh, kernel, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    thCnts = []
    for c in cnts:
        shapeArea = cv2.contourArea(c)
        if (shapeArea > minArea):
            thCnts.append(c)

    return thCnts

'''
Wireup de terminais de dois CIs diferentes
'''
def executeWireupCi(img, x, y, roiWidth, roiHeight):
    roiX = int(x - roiWidth / 2)
    if roiX < 0:
        roiX = 0
    roiY = int(y - roiHeight / 2)
    if roiY < 0:
        roiY = 0
    roiWid = int(roiX + roiWidth)
    roiHei = int(roiY + roiHeight)
    imgRoi = img[roiY:roiHei, roiX:roiWid]

    #cv2.imshow("roi", imgRoi)
    #cv2.waitKey(0)

    imgBlank = np.zeros([imgRoi.shape[0], imgRoi.shape[1], 1], dtype=np.uint8)

    ## Procura por grupos de terminais de CIs pelo seu contorno e posicao
    mask, cnts = findTerminals(imgRoi, 120, 220, 9.0, 300)
    groups = detectGroupTerminals(imgRoi, cnts, 5, 50, 30, 3)

    ## Se não achou grupos, procura novamente com outros parametros...
    if len(groups) <= 0:
        mask, cnts = findTerminals(imgRoi, 170, 255, 9.0, 100)
        groups = detectGroupTerminals(imgRoi, cnts, 5, 50, 30, 3)

    for group in groups:
        for cnt in group:
            cv2.drawContours(imgBlank, [cnt], -1, (255, 255, 255), -1)

    thGroups = []

    ## São necessarios dois grupos para o processamento, caso não tenha 2, procura novamente usando metodo ByThreshold
    if (len(groups) < 2):
        for th in range(0, 255):
            cnts = findTerminalsByThreshold(imgRoi, th, 100)
            tempGroup = detectGroupTerminals(imgRoi, cnts, 5, 50, 30, 3)
            # thGroups.append(tempGroup)

            # print(tempGroup)
            for gp in tempGroup:
                thGroups.append(gp)

        for group in thGroups:
            groups.append(group)
            for cnt in group:
                cv2.drawContours(imgBlank, [cnt], -1, (255, 255, 255), -1)

    kernel = np.ones((3, 3), np.uint8)
    imgBlank = cv2.erode(imgBlank, kernel, iterations=2)
    imgBlank = cv2.dilate(imgBlank, kernel, iterations=2)

    endcnts = cv2.findContours(imgBlank.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    endcnts = imutils.grab_contours(endcnts)

    finalGroups = detectGroupTerminals(imgRoi, endcnts, 5, 50, 30, 3)

    #print("Número de grupos:", len(finalGroups))
    if len(finalGroups) == 0:
        return 0

    if (len(finalGroups) >= 2):  # Se houverem 2 grupos, injeta falha
        ## Seleciona aleatoriamente dois terminais pro wireup
        opt1 = randint(0, len(finalGroups[0]) - 1)
        opt2 = randint(0, len(finalGroups[1]) - 1)
        pin1 = finalGroups[0][opt1]
        pin2 = finalGroups[1][opt2]

        pin1X, pin1Y = cntCentroid(pin1)
        pin2X, pin2Y = cntCentroid(pin2)

        ## Calcula angulo entre os terminais
        degreesPoints = math.degrees(math.atan2(pin2Y - pin1Y, pin2X - pin1X))

        overImg = None

        ## Conforme o angulo injeta um wireup diferente
        if degreesPoints < -60 and degreesPoints > -120:

            if degreesPoints < -80 and degreesPoints > -100:
                overImg = cv2.imread(img90Angle, cv2.IMREAD_UNCHANGED)
                newOverImg = cv2.resize(overImg, (int(overImg.shape[1]), int(np.abs(pin1Y - pin2Y))))
                imgRoi = smartOverlay(imgRoi, newOverImg, pin2Y, pin2X)
            else:
                overImg = cv2.imread(imgHighAngle, cv2.IMREAD_UNCHANGED)
                newOverImg = cv2.resize(overImg, (int(np.abs(pin1X - pin2X)), int(np.abs(pin1Y - pin2Y))))

                if degreesPoints >= -90:
                    newOverImg = cv2.flip(newOverImg, 0)
                    imgRoi = smartOverlay(imgRoi, newOverImg, pin2Y, pin1X)
                else:
                    imgRoi = smartOverlay(imgRoi, newOverImg, pin2Y, pin2X)
        else:
            overImg = cv2.imread(imgHighAngle, cv2.IMREAD_UNCHANGED)
            newOverImg = cv2.resize(overImg, (int(np.abs(pin1X - pin2X)), int(np.abs(pin1Y - pin2Y))))

            if degreesPoints > -60:
                newOverImg = cv2.flip(newOverImg, 0)
                imgRoi = smartOverlay(imgRoi, newOverImg, pin2Y, pin1X)
            else:
                imgRoi = smartOverlay(imgRoi, newOverImg, pin2Y, pin2X)

    img[roiY:roiHei, roiX:roiWid] = imgRoi

    return 1
    #cv2.imshow("ci final", imgRoi)
    #cv2.waitKey(0)

'''
Wireup de terminais de um mesmo CI
'''
def executeWireupTerminal(img, x, y, roiWidth, roiHeight):
    roiX = int(x - roiWidth / 2)
    if roiX < 0:
        roiX = 0
    roiY = int(y - roiHeight / 2)
    if roiY < 0:
        roiY = 0
    roiWid = int(roiX + roiWidth)
    roiHei = int(roiY + roiHeight)
    imgRoi = img[roiY:roiHei, roiX:roiWid]
    imgRoiOriginal = imgRoi.copy()
    # cv2.imshow("roi", imgRoi)
    # cv2.waitKey(0)

    imgBlank = np.zeros([imgRoi.shape[0], imgRoi.shape[1], 1], dtype=np.uint8)

    ## Procura por grupos de terminais de CIs pelo seu contorno e posicao
    mask, cnts = findTerminals(imgRoi, 120, 220, 9.0, 300)
    groups = detectGroupTerminals(imgRoi, cnts, 5, 50, 30, 3)

    ## Se não achou grupos, procura novamente com outros parametros...
    if len(groups) <= 0:
        mask, cnts = findTerminals(imgRoi, 170, 255, 9.0, 100)
        groups = detectGroupTerminals(imgRoi, cnts, 5, 50, 30, 3)

    for group in groups:
        for cnt in group:
            cv2.drawContours(imgBlank, [cnt], -1, (255, 255, 255), -1)

    thGroups = []

    ## São necessarios dois grupos para o processamento, caso não tenha 2, procura novamente usando metodo ByThreshold
    if (len(groups) < 1):
        for th in range(0, 255):
            cnts = findTerminalsByThreshold(imgRoi, th, 100)
            tempGroup = detectGroupTerminals(imgRoi, cnts, 5, 50, 30, 3)
            # thGroups.append(tempGroup)

            # print(tempGroup)
            for gp in tempGroup:
                thGroups.append(gp)

        for group in thGroups:
            groups.append(group)
            for cnt in group:
                cv2.drawContours(imgBlank, [cnt], -1, (255, 255, 255), -1)

    kernel = np.ones((3, 3), np.uint8)
    imgBlank = cv2.erode(imgBlank, kernel, iterations=2)
    imgBlank = cv2.dilate(imgBlank, kernel, iterations=2)

    endcnts = cv2.findContours(imgBlank.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    endcnts = imutils.grab_contours(endcnts)

    finalGroups = detectGroupTerminals(imgRoi, endcnts, 5, 50, 30, 3)

    #print("grupos:", len(finalGroups))
    if len(finalGroups) == 0:
        return 0

    if len(finalGroups) > 1:
        opt = randint(0, len(finalGroups) - 1)
        procGroup = finalGroups[opt]
    else:
        procGroup = finalGroups[0]

    opt1 = randint(0, len(procGroup) - 1)
    pin1 = procGroup[opt1]

    pin1X, pin1Y = cntCentroid(pin1)

    nearIdx = opt1
    nearDistance = 10000000000
    for idx, item in enumerate(procGroup):
        if idx != opt1:
            pin2X, pin2Y = cntCentroid(item)
            euDistance = distance.euclidean([pin1X, pin1Y], [pin2X, pin2Y])
            if euDistance < nearDistance:
                nearIdx = idx
                nearDistance = euDistance

    pin2 = procGroup[nearIdx]
    pin2X, pin2Y = cntCentroid(pin2)

    overImg = cv2.imread(img0Angle, cv2.IMREAD_UNCHANGED)
    newOverImg = cv2.resize(overImg, (int(np.abs(pin1X - pin2X)), int(overImg.shape[0])) )

    if pin1X > pin2X:
        imgRoi = smartOverlay(imgRoi, newOverImg, pin2Y, pin2X)
    else:
        imgRoi = smartOverlay(imgRoi, newOverImg, pin1Y, pin1X)


    imgTeste = imgRoiOriginal - imgRoi

    #cv2.imshow("11", imgTeste)
    #cv2.waitKey(0)

    img[roiY:roiHei, roiX:roiWid] = imgRoi

    #cv2.imshow("teste", imgRoi)
    #cv2.waitKey(0)

    return 1