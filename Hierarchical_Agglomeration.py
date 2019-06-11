import numpy as np
import loadData
import math
import general as g

'''
given a dataset (d1, d2, d3, ....dN) of size N
# compute the distance matrix
for i=1 to N:
   # as the distance matrix is symmetric about 
   # the primary diagonal so we compute only lower 
   # part of the primary diagonal 
   for j=1 to i:
      dis_mat[i][j] = distance[di, dj] 
each data point is a singleton cluster
repeat
   merge the two cluster having minimum distance
   update the distance matrix
untill only a single cluster remains

'''

def separateByClass(dataset, label):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (label[i] not in separated):
            separated[label[i]] = []
        separated[label[i]].append(vector)
    # print(separated)
    return separated


class Cluster:
    def __init__(self, dataset, type='single', cluster_num=2):
        self.dataset = dataset
        self.X, self.Y = g.splitXandY(dataset, 13, len(dataset))
        self.cluster_num  = cluster_num
        self.dis_mat = np.zeros((dataset.shape[0],dataset.shape[0]))
        self.label = []
        self.draw_label = []

        self.allCluster = []
        self.type = type
        # initialize label and distance matrix
        for i in range(dataset.shape[0]):
            self.label.append(i)
            self.draw_label.append(i)
            self.allCluster.append(i)
            for j in range(dataset.shape[0]):
                self.dis_mat[i][j] =  np.linalg.norm(self.X[i] - self.X[j], ord=1)
        self.link = []



    def calc_dis_matrix_single(self):
        distanceMat = np.zeros((dataset.shape[0],dataset.shape[0]))
        separated = separateByClass(self.X, self.label)

        for cluster_memKey_i, cluster_memData_i in separated.items():
            for cluster_memKey_j, cluster_memData_j in separated.items():
                if(cluster_memKey_i != cluster_memKey_j):
                    min_dis = 1e6
                    for i in range(len(cluster_memData_i)):
                        for j in range(len(cluster_memData_j)):
                            dis = np.linalg.norm(cluster_memData_i[i] - cluster_memData_j[j])
                            if (dis < min_dis and dis != 0):
                                min_dis = dis


                    distanceMat[cluster_memKey_i][cluster_memKey_j] = min_dis

        return  distanceMat

    def calc_dis_matrix_complete(self):
        distanceMat = np.zeros((dataset.shape[0],dataset.shape[0]))
        separated = separateByClass(self.X, self.label)

        for cluster_memKey_i, cluster_memData_i in separated.items():
            for cluster_memKey_j, cluster_memData_j in separated.items():
                if(cluster_memKey_i != cluster_memKey_j):
                    max_dis = -1
                    for i in range(len(cluster_memData_i)):
                        for j in range(len(cluster_memData_j)):
                            dis = np.linalg.norm(cluster_memData_i[i] - cluster_memData_j[j])
                            if (dis > max_dis and dis >= 0):
                                max_dis = dis
                    distanceMat[cluster_memKey_i][cluster_memKey_j] = max_dis

        return  distanceMat


    def update_dis_mat(self):
        distanceMat = self.dis_mat.copy()
        while len(self.allCluster) > 1 :
            min = 1e6
            merge_1 = 0
            merge_2 = 0
            for i in range(dataset.shape[0]):
                for j in range(i):
                    if distanceMat[i][j] < min and distanceMat[i][j] != 0 :
                        min = distanceMat[i][j]
                        merge_1 = i
                        merge_2 = j

            separated = separateByClass(self.X, self.draw_label)

            num = len(separated[self.draw_label[merge_1]])+len(separated[self.draw_label[merge_2]])
            self.link.append([self.draw_label[merge_1], self.draw_label[merge_2], min, num])
            for i in range(len(self.label)):
                if self.label[i] == merge_2:
                    self.draw_label[i] = len(self.X) + len(self.link) - 1
                if self.label[i] == merge_1:
                    self.label[i] = merge_2
                    self.draw_label[i] = len(self.X) + len(self.link) - 1


            self.allCluster = np.unique(self.label)
            # print(len(self.allCluster))
            if self.type == 'single':
                distanceMat = self.calc_dis_matrix_single()
            if self.type == 'complete':
                distanceMat = self.calc_dis_matrix_complete()
            if len(self.allCluster) == self.cluster_num :
                self.plot()
                self.final_ans = self.label.copy()


    def plot(self):
        separated = separateByClass(self.X, self.label)
        import matplotlib.pyplot as plt
        import itertools

        for key, data in separated.items():
            print(key," : ",len(data))
        # print(list(separated.keys()))
        allKey = list(separated.keys())
        colors = itertools.cycle(["red", "blue", "green"])
        for i in range(self.cluster_num):
            color_this = next(colors)
            for j in range(len(separated[allKey[i]])):
                plt.scatter(separated[allKey[i]][j][0], separated[allKey[i]][j][1], color=color_this, alpha=0.6)
        plt.show()


dataset = loadData.loadWine()
cluster = Cluster(dataset,type='single', cluster_num=3)
cluster.update_dis_mat()

from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score

print("Adjusted Rand :", metrics.adjusted_rand_score(cluster.Y.flatten(), cluster.final_ans))
print("Normalized Mutual Info:", normalized_mutual_info_score(cluster.Y.flatten(), cluster.final_ans))

plt.figure(figsize=(10, 7))
plt.title("Dendograms")
dend = shc.dendrogram(np.array(cluster.link),color_threshold=0.8)
plt.show()