import os
import json
import numpy as np

trainPath = '../dataset/BOP/train_pbr'
testPath = '../dataset/BOP/train_pbr'
splitTrainList = True
n_splitTrainList = 1000

date = 1010

dataListPath = os.path.join(trainPath, 'data_list_{}.txt'.format(date))
trainListPath = os.path.join(trainPath, 'train_list_{}.txt'.format(date))
trainListSplitPath = os.path.join(trainPath, 'train_list_split{}_{}.txt'.format(n_splitTrainList, date))
validListPath = os.path.join(testPath, 'valid_list_{}.txt'.format(date))
dataList = open(dataListPath, 'w')
trainList = open(trainListPath, 'w')
# trainListSplit = open(trainListSplitPath, 'w')
validList = open(validListPath, 'w')

files = os.listdir(trainPath)
for subFile in files:
    if os.path.isdir(os.path.join(trainPath, subFile)):
        labelPath = os.path.join(trainPath, subFile, 'scene_gt.json')
        with open(labelPath, 'r') as f:
            data = json.load(f)
            for i in data:
                sceneKey = str(i)
                scene = data[sceneKey]
                object_num = len(scene)
                folderName = subFile
                sceneId = "{:0>6d}".format(int(i))
                scenePath = "{} {} {}".format(folderName, sceneId, object_num)
                lineScene = scenePath + '\n'
                dataList.write(lineScene)
dataList.close()

with open(dataListPath, 'r') as f:
    data = f.readlines()
    # np.random.seed(666)
    # np.random.shuffle(data)
    number = len(data)
    trainPoint = int(number * 0.9)
    for i, image in enumerate(data):
        if i < trainPoint:
            trainList.write(image)
        else:
            validList.write(image)
trainList.close()
validList.close()
