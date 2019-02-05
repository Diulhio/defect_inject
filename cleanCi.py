import cv2
import numpy as np
import glob
import imutils
import math
import commonFuncs as cF

srcPath = "/home/diulhio/Codigos/pcbs/clean_ci/1/"
file_type = "*"  ## */* para buscar recursivamente


def cntCentroid(cnt):
    M = cv2.moments(cnt)
    cX = cY = 0
    if (M["m00"] != 0):
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))

    return cX, cY


def isClose(cnt1, cnt2, maxDistance, centroidDistance = 150):
    # print(cnt1)

    cn1X, cn1Y = cntCentroid(cnt1)
    cn2X, cn2Y = cntCentroid(cnt2)

    if math.hypot(cn1X - cn2X, cn1Y - cn2Y) > centroidDistance:
        return False

    minorDistance = 100000
    for cn1 in cnt1:
        #tt = img.copy()
        for cn2 in cnt2:
            distance = math.hypot(cn1[0][0] - cn2[0][0], cn1[0][1] - cn2[0][1])

            if distance < minorDistance:
                minorDistance = distance

            if distance < maxDistance:
                return True
    return False


def detectGroups(cnts, minGroup):
    groups = []
    inGroup = []
    global img

    for index, c in enumerate(cnts):
        tempGroup = [c]

        if index not in inGroup:
            for cnGroup in tempGroup:

                for idxCn, cn in enumerate(cnts):
                    if np.array_equal(cn, cnGroup) or idxCn in inGroup:
                        pass
                    elif isClose(cn, cnGroup, 20, 180):
                        #print("Grupo")
                        tempGroup.append(cn)
                        inGroup.append(idxCn)

        if len(tempGroup) >= minGroup:
            inGroup.append(index)
            groups.append(tempGroup)

    return groups

def detectGroupContours(img, threshold=140, itDilate=3, minCntArea=500, minNGroup=3):
    #img = cv2.imread(imgPath)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=itDilate)

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(contours)

    if len(cnts) == 0:
        return []

    newCnt = []
    for item in cnts:
        if cv2.contourArea(item) > minCntArea:
            newCnt.append(item)
    cnts = newCnt

    cntGroups = detectGroups(cnts, minNGroup)

    return cntGroups

def selectByBB(groups, img=None, minArea=15000, minRelation=0.35, maxRelation=0.7, minbbcntRelation=0.5):
    selection = []
    for group in groups:
        groupIntersection = groupBoudingBox(group)

        x, y, w, h = cv2.boundingRect(groupIntersection)
        bbArea = w*h
        relation = h/float(w)
        cntArea = cv2.contourArea(groupIntersection)
        bbcntRelation = cntArea/bbArea

        #print(bbArea, relation)
        #print(cntArea, cntArea/bbArea)
        #print("")

        if img != None:
            showImg = img.copy()
            cv2.rectangle(showImg, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.imshow("bbs", showImg)
            cv2.waitKey(0)

        if bbArea > minArea and relation > minRelation and relation < maxRelation and bbcntRelation > minbbcntRelation:
            selection.append( [group, [x, y, w, h] ] )

    return selection


def groupBoudingBox(group):
    newGroups = []
    for memberGroup in group:
        for pt in memberGroup:
            newGroups.append(pt)

    return np.array(newGroups)

def executeCleanCi(img, x, y, roiWidth, roiHeight):
    roiX = int(x-roiWidth/2)
    if roiX < 0:
        roiX = 0
    roiY = int(y-roiHeight/2)
    if roiY < 0:
        roiY = 0
    roiWid = int(roiX + roiWidth)
    roiHei = int(roiY + roiHeight)
    imgRoi = img[roiY:roiHei, roiX:roiWid]
    #print("----> ", roiX, roiY, roiWid, roiHei)
    #cv2.imshow("imgRoi", imgRoi)
    #cv2.waitKey(1)

    groups = detectGroupContours(imgRoi)
    cntSelection = selectByBB(groups)
    imgBlank = np.zeros([imgRoi.shape[0], imgRoi.shape[1]], dtype=np.uint8)
    imgBlankFinal = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)

    for sel in cntSelection:
        tmpImg = imgRoi[sel[1][1]:sel[1][1] + sel[1][3],
                 sel[1][0]:sel[1][0] + sel[1][2]]

        tmpImg = cv2.cvtColor(tmpImg, cv2.COLOR_BGR2GRAY)
        ret, tmpImg = cv2.threshold(tmpImg, 140, 255, cv2.THRESH_BINARY)

        imgBlank[sel[1][1]:sel[1][1] + sel[1][3],
        sel[1][0]:sel[1][0] + sel[1][2]] = tmpImg

        imgBlankFinal[roiY:roiHei, roiX:roiWid] = imgBlank

    kernel = np.ones((3, 3), np.uint8)
    imgBlank = cv2.dilate(imgBlankFinal, kernel, iterations=1)

    #cv2.imshow("imgBlank", imgBlank)
    #cv2.waitKey(0)

    final = cv2.inpaint(img, imgBlank, 11, cv2.INPAINT_TELEA)

    #showImg = cv2.resize(final,None,fx=0.25,fy=0.25, interpolation=cv2.INTER_CUBIC)
    #img[y:roiHeight, x:roiWidth] = final

    imgTest = img - final

    imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2GRAY)
    #print(imgTest.shape)

    ret = 0
    if cv2.countNonZero(imgTest) == 0:
        ret = 0
    else:
        ret = 1

    return final, ret



'''
files = cF.getFiles(srcPath + file_type)

for imgPath in files:
    img = cv2.imread(imgPath)
    groups = detectGroupContours(imgPath)
    print(len(groups))

    cntSelection = selectByBB(groups)

    imgBlank = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)

    showImg = img.copy()
    for sel in cntSelection:
        cv2.rectangle(showImg, (sel[1][0], sel[1][1]), (sel[1][0] + sel[1][2], sel[1][1] + sel[1][3]), (0, 255, 0), 2)
        tmpImg = img[sel[1][1]:sel[1][1]+sel[1][3],
                     sel[1][0]:sel[1][0]+sel[1][2]]

        tmpImg = cv2.cvtColor(tmpImg, cv2.COLOR_BGR2GRAY)
        ret, tmpImg = cv2.threshold(tmpImg, 140, 255, cv2.THRESH_BINARY)

        imgBlank[sel[1][1]:sel[1][1]+sel[1][3],
                 sel[1][0]:sel[1][0]+sel[1][2]] = tmpImg

    kernel = np.ones((3, 3), np.uint8)
    imgBlank = cv2.dilate(imgBlank, kernel, iterations=1)

    final = cv2.inpaint(img, imgBlank, 11, cv2.INPAINT_TELEA)

    cv2.imshow("blank", imgBlank)
    cv2.imshow("bounding", showImg)
    cv2.imshow("final", final)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
'''