import random
import math
import numpy as np


# split data into training data and testing data with split ratio
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


# separate data  by different class
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    # print(separated)
    # prior(separated)
    return separated


def prior(dataset, classNum):
    dataNum = 0
    prior = np.zeros((1, classNum))
    for i in range(classNum):
        #　print(len(dataset))
        dataNum += len(dataset[i + 1])
    # all data number
    for i in range(classNum):
        prior[0][i] = (float(len(dataset[i + 1]) / dataNum))
    # print(prior)
    return prior


def mean(numbers):
    return sum(numbers)/float(len(numbers))


def stdev(numbers):
    e = 1e-15
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)+e


# calculate the mean and standard variance of each attribute (column)
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]

    return summaries


# separate the data by class
# calculate the each class's mean and standard variance of each attribute (column)
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


# p(x|wi)
def calculateProbability(x, mean, stdev):
    # print(x)

    exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev, 2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


# calculate the p(x|wi) of each class
# classValue = which class
# classSummaries = mean and stdev of the attribute
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        # print(len(classSummaries))
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            # print(calculateProbability(x, mean, stdev))
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    # print(probabilities)

    return probabilities


def getAccuracy(testSet, predictions, confusion_matrix):
    correct = 0
    for x in range(len(testSet)):
        confusion_matrix[int(testSet[x][-1])-1][int(predictions[x])-1]+=1
        if testSet[x][-1] == predictions[x]:
            correct += 1
    print("confusion matrix")
    print(confusion_matrix)
    return (correct/float(len(testSet))) * 100.0 , confusion_matrix

def splitXandY(dataset, attrNum, len):

    splitX = np.zeros((len, attrNum))
    splitY = np.zeros((len, 1))

    for i in range(len):
        for j in range(attrNum):
            splitX[i][j] = dataset[i][j]
        splitY[i][0] = dataset[i][attrNum]
    return [splitX, splitY]

def zero_the_label(dataset):

    arr = np.zeros((len(dataset),len(dataset[0])))


    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i][j] = dataset[i][j]
        arr[i][-1] = 0
    return arr

