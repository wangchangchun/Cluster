'''

1. Eps = Density = 以資料點為圓心所設的半徑長度
2. Core Point（核心點）：以核心點為半徑所圍繞出來的範圍能包含超過我們指定的數量
3. Border Point（ 邊界點）：被某個核心點包含，但以他為中心卻沒辦法包含超過我們指定的數量。
4. Noise Point（雜訊點）：不屬於核心點，也不屬於邊界點，即為雜訊點。
5. 密度相連：如果兩個核心點互為邊界點的話，則可把兩個核心點合併在同一個群組中

1. 將所有的點做過一次搜尋，找出核心點、邊界點、雜訊點
2. 移除所有雜訊點
3. 設立一個「當前群集編號」的變數＝0
4. 　for 1 到 最後一個核心點 do
5. 　　假設 這個核心點並沒有被貼上群組編號 則
6. 　　　那就把「當前群集編號」的變數 + 1 　　
7. 　　　把「當前群集編號」給這個被抽出的核心點
8. 　　結束這個假設
9. 　　for 這個核心點在密度相連後所有可以包含的點 do
10.　　　假設 這個點還沒有被貼上任何群組編號 則
11.　　　　把這個點貼上「當前變數的編號」
12.　　　結束假設
13.　   結束for迴圈
14.　結束for迴圈
'''

import numpy as np
import loadData
import math
import general as g
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score




def separateByClass(dataset, label):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (label[i] not in separated):
            separated[label[i]] = []
        separated[label[i]].append(vector)
    # print(separated)
    return separated


class DBSCAN:
    def __init__(self, dataset, eps, minPts):
        self.dataset = dataset
        self.data = g.zero_the_label(self.dataset)
        for i in range(len(self.dataset)):
            self.data[i][-1] = 0
        self.R = eps
        self.density = minPts
        self.cluster_num = 0


    def find_core_point(self):
        for i in range(len(self.data)):
            neighbor = self.find_border_point(self.data[i])
            if len(neighbor) >= self.density and self.data[i][-1] == 0:
                # print("core ",self.data[i])
                self.cluster_num += 1
                self.data[i][-1] = self.cluster_num  # new cluster
                self.combine(i, neighbor)


    def find_border_point(self, point):
        neighbor = []
        for i in range(len(self.data)):
            dis = np.linalg.norm(point[0:-1] - self.data[i][0:-1])
            if dis < self.R:
                neighbor.append(i)

        return neighbor

    def combine(self, core, neighbor):
        for j in neighbor:
            if self.data[j][-1] == 0:
                self.data[j][-1] = self.data[core][-1]
                near_neighbor = self.find_border_point(self.data[j])
                if len(near_neighbor) >= self.density:
                    self.combine(j, near_neighbor)

                # self.core_points.append(self.not_noise_points[j])

    def plot(self):
        clusted, label = g.splitXandY(self.data, 13, len(self.data))
        separated = separateByClass(clusted, label.flatten())
        import matplotlib.pyplot as plt
        import itertools

        for key, data in separated.items():
            print(key, " : ", len(data))
        allKey = list(separated.keys())
        colors = itertools.cycle(["red", "blue", "green","yellow", "orange"])
        for i in range(len(separated.keys())):
            color_this = next(colors)
            for j in range(len(separated[allKey[i]])):
                plt.scatter(separated[allKey[i]][j][0], separated[allKey[i]][j][1], color=color_this, alpha=0.6)

        plt.show()



dataset = loadData.loadWine()
X,y = g.splitXandY(dataset,13,len(dataset))
dbscan = DBSCAN(dataset,eps=50, minPts=10)
dbscan.find_core_point()
print("eps=1, minPts=20")
dbscan.plot()
print("Adjusted Rand :", metrics.adjusted_rand_score(y.flatten(), dbscan.data[:,-1]))
print("Normalized Mutual Info:", normalized_mutual_info_score(y.flatten(), dbscan.data[:,-1]))


