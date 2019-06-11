import numpy as np


def convertfunc (x):
    if x == "Iris-setosa":
        return float(1)
    if x == "Iris-versicolor":
        return float(2)
    if x == "Iris-virginica":
        return float(3)


# load data
def loadIris():
    names = ["sepal Length", "sepal Width", "petal Length", "petal Width", "class"]
    data = np.genfromtxt("iris_data/iris_dataset.txt", dtype=None, delimiter=',', names=names,
                         skip_header=0, encoding='ascii', converters={"class": convertfunc})
    return data

def loadGlass():
    names = ["ID","RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "class"]
    data = np.genfromtxt("glass_data/glass.data", dtype=None, delimiter=',', names=names,
                         skip_header=0, usecols = range(1,11), encoding='ascii')

    return np.array(data)


def loadWine():
    dataset = np.zeros((178, 14))
    with open("wine_data/wine.data") as file:
        lines = file.readlines()
        lineNum = 0
        for line in lines:
            data = line.split('\n')[0].split(',')
            for i in range(1,14):
                dataset[lineNum][i-1] = data[i]
            dataset[lineNum][13] = data[0]
            lineNum += 1

    # print(dataset)
    return dataset


def convert(x):
    if (x == 'g'):
        return float(1)
    if (x == 'b'):
        return float(2)


def loadIono():

    dataset = np.zeros((351, 33))
    with open("ionosphere/ionosphere.data") as file:
        lines = file.readlines()
        lineNum = 0
        for line in lines:
            data = line.split('\n')[0].split(',')

            for i in range(32):
                dataset[lineNum][i] = (float(data[i+2])+1)/2

            if data[34] == 'g':
                dataset[lineNum][32] = float(1)

            if data[34] == 'b':
                dataset[lineNum][32] = float(2)


            lineNum += 1
    return np.array(dataset)


def loadBreast():

    dataset = np.zeros((699, 10))
    with open("breast/breast-cancer-wisconsin.data") as file:
        lines = file.readlines()
        lineNum = 0
        for line in lines:
            data = line.split('\n')[0].split(',')
            for i in range(1, 11):
                if data[i]=='?':
                    data[i] = 0
                # print(data[i])
                dataset[lineNum][i - 1] = float(data[i])/10
            if dataset[lineNum][9] == 0.2:
                dataset[lineNum][9] = 1
            if dataset[lineNum][9] == 0.4:
                dataset[lineNum][9] = 2

            lineNum += 1
    # print(np.array(dataset))
    return np.array(dataset)



import numpy as np
from PIL import Image



def cutImg( filename,W,H,flip=False):
    from torchvision import transforms
    openImage = Image.open(filename)

    # print(np.array(openImage))
    cutImage = []
    dx = 40
    dy = 40
    n = 1

    x1 = 0
    y1 = 0
    x2 = 40
    y2 = 40

    while x2 <= H:
        while y2 <= W:
            temp = openImage.crop((y1, x1, y2, x2))
            np_temp = np.array(temp)# /256
            cutImage.append(np_temp.flatten())
            if flip == True:
                trans = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                ])
                fliped = trans(temp)
                np_filped = np.array(fliped)
                cutImage.append(np_filped.flatten())
            y1 = y1 + dy
            y2 = y1 + 40
        x1 = x1 + dx
        x2 = x1 + 40
        y1 = 0
        y2 = 40

    # print(np.array(cutImage).shape)
    return cutImage


def loadGender():
    female = cutImg("fP1.bmp",400, 400)
    male = cutImg("mP1.bmp",400, 400)
    alldata = np.vstack((female, male))
    allLabel = []
    for i in range(100):
        allLabel.append(1)
    for i in range(100):
        allLabel.append(2)

    return np.hstack((alldata, np.array(allLabel)[:,None]))

def loadFace():
    face = cutImg("facesP1.bmp", 640, 200, True)
    allLabel = []
    for i in range(80):
        allLabel.append(i%16)
        allLabel.append(i % 16)


    return np.hstack((np.array(face), np.array(allLabel)[:, None]))




