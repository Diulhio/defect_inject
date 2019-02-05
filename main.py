# -*- coding: utf-8 -*-
import cv2
import glob
import time
from cleanCi import *
from wireupCi import *
from ciInjection import *
import random
import sys, os
import warnings
import datetime
warnings.simplefilter(action='ignore', category=FutureWarning)

MAX_RANDOM = 5

#imgPath = "/home/diulhio/Codigos/pcbs/dataset/referencia.png"

if len(sys.argv) < 2:
    print("usage: python3 main.py <file or path>")
    exit()

classes = ["ci", "clean_ci", "none", "wireup_ci", "wireup_term"]


fileList = 0
fileArg = sys.argv[1]
isDirectory = os.path.isdir(fileArg)
showOption = ""
classIndices = [[] for i in range(0,len(classes))]
if isDirectory:
    #fileList = [fileArg+x for x in os.listdir(fileArg)]
    fileList = glob.glob(fileArg + '*.png')
    fileList.extend(glob.glob(fileArg + '*.jpg'))
    fileList.extend(glob.glob(fileArg + '*.jpeg'))

    print(fileList)
    showOption = input("Mostrar resultados? (s ou n): ")
    #input_response = [random.randint(0, MAX_RANDOM) for i in range(0,len(classes))]
else:
    fileList = [fileArg]
    tempPath = fileArg.split('/')
    fileArg = '/'.join(tempPath[:-1])

    input_entries = ["nº ci: ", "nº clean ci: ", "nº wireup cis: ", "nº wireup terminals: "]
    input_response = np.zeros(len(input_entries))

    for idx, item in enumerate(input_entries):
        value = input(item)

        if value.isdigit() and int(value) > 0:
            input_response[idx] = value
        else:
            input_response[idx] = 0

classes_random = [[] for i in range(0,len(classes))]

## Cria diretorio de saida
if fileArg[-1] != '/':
    fileArg += '/'
fileOutput = fileArg + "image_output"
if not os.path.exists(fileOutput):
    os.makedirs(fileOutput)

print("Diretorio de saida: ", fileOutput)

fileOutputLog = open(fileOutput + "/" + datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".txt", "w")

# Leitura do SVM treinado
svmModel = cv2.ml.SVM_load("pcb_svm_model_SIFT.yml")

# Leitura do descritor
detect = cv2.xfeatures2d.SIFT_create()
extract = cv2.xfeatures2d.SIFT_create()

flann_params = dict(algorithm=1, trees=5)
matcher = cv2.FlannBasedMatcher(flann_params, {})

voc = np.load("sift_bow.dat")
extract_bow = cv2.BOWImgDescriptorExtractor(extract, matcher)
extract_bow.setVocabulary(voc)

for imgPath in fileList:
    input_response = [random.randint(0, MAX_RANDOM) for i in range(0, len(classes))]
    print(input_response)
    fileName = imgPath.split('/')[-1]
    fileExtension = fileName.split('.')[-1]
    fileName = fileName.split('.')[0]
    responseArr = [0, 0, 0, 0]

    print(imgPath)
    img = cv2.imread(imgPath)
    imgOri = img.copy()
    rois = getGridArray(img)

    _start = time.time()
    for index, item in enumerate(rois):
        imgRoi = img[item.y:item.height, item.x:item.width]
        #cv2.imshow("roi", imgRoi)
        #cv2.waitKey(0)

        descriptor = extract_bow.compute(imgRoi, detect.detect(imgRoi))
        #print(descriptor)

        if descriptor != None:
            descriptor = np.asarray(descriptor[0], np.float32)
            descriptor = [descriptor]
            descriptor = np.array(descriptor)

            testResponse = svmModel.predict(descriptor, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
            testResponse = int(testResponse[1][0][0])
            classIndices[testResponse].append(index)
        else:
            classIndices[2].append(index)

    resultsOutput = ""
    ################ ci ################
    classes_random[0] = random.sample(range(0, len(classIndices[0])), len(classIndices[0]) )
    if int(input_response[0]) > 0:
        for item in classes_random[0]:
            ret = 0
            img, ret = executeCiInjection(img, rois[classIndices[0][item]].centerX, rois[classIndices[0][item]].centerY, 256, 256)
            responseArr[0] += ret
            if ret != 0:
               resultsOutput += ";ci" + rois[classIndices[0][item]].logFormat()
            if responseArr[0] >= int(input_response[0]):
                break

    ################ clean ci ################
    classes_random[1] = random.sample(range(0, len(classIndices[1])), len(classIndices[1]) )
    if int(input_response[1]) > 0:
        for item in classes_random[1]:
            img, ret = executeCleanCi(img, rois[classIndices[1][item]].centerX, rois[classIndices[1][item]].centerY, 1200, 500)
            responseArr[1] += ret
            if ret != 0:
               resultsOutput += ";clean" + rois[classIndices[1][item]].logFormat()
            if responseArr[1] >= int(input_response[1]):
                break

    ################ wireup ci ################
    classes_random[2] = random.sample(range(0, len(classIndices[3])), len(classIndices[3]) )
    #print("Wire up ci: ", int(input_response[2]))
    if int(input_response[2]) > 0:
        for item in classes_random[2]:
            ret = executeWireupCi(img,rois[classIndices[3][item]].centerX, rois[classIndices[3][item]].centerY,256,256)
            responseArr[2] += ret
            if ret != 0:
               resultsOutput += ";wci " + rois[classIndices[3][item]].logFormat()
            if responseArr[2] >= int(input_response[2]):
                break


    ################ wireup term ################
    classes_random[3] = random.sample(range(0, len(classIndices[4])), len(classIndices[4]) )
    if int(input_response[3]) > 0:
        for item in classes_random[3]:
            ret = executeWireupTerminal(img,rois[classIndices[4][item]].centerX, rois[classIndices[4][item]].centerY,256,256)
            responseArr[3] += ret
            if ret != 0:
               resultsOutput += ";wterm" + rois[classIndices[4][item]].logFormat()
            if responseArr[3] >= int(input_response[3]):
                break

    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    imgOri = cv2.resize(imgOri, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

    if not isDirectory or showOption == "s":
        cv2.imshow("diff", (img-imgOri))
        cv2.imshow("new", img)
        cv2.imshow("original", imgOri)
        cv2.waitKey(0)

    responseArr = [str(item) for item in responseArr]
    finalFile = fileOutput + '/' + fileName + '_' + '_'.join(responseArr) + "." + fileExtension

    fileOutputLog.write(finalFile + resultsOutput + "\n")

    print(finalFile)
    cv2.imwrite(finalFile, img)

    print("Tempo: ", time.time() - _start)
    print("N classes: ", [len(item) for item in classIndices])
fileOutputLog.close()