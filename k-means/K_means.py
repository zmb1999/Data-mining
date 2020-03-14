import numpy as np
from collections import defaultdict
from random import uniform
from sklearn import datasets
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import metrics

np.set_printoptions(threshold = 1e6)

# 读取文件中的数据
def get_score(X, y_true, y_pred):
    Micro_F1 = metrics.f1_score(y_true,y_pred,labels=[1,2,3],average='micro')
    Macro_F1 = metrics.f1_score(y_true,y_pred,labels=[1,2,3],average='macro')
    s_score = metrics.silhouette_score(X,y_pred)
    return Macro_F1,Micro_F1,s_score

def get_data(filename):
    data_set = []
    target = []
    with open(filename, "r") as f:
        for line in f:
            data_set.append(list(map(float, line.split(',')))[1:])
            target.append(list(map(float, line.split(',')))[0])
    return data_set,target

def point_avg(points):
    dimensions = len(points[0])

    new_center = []

    for dimension in range(dimensions):
        dim_sum = 0
        for p in points:
            dim_sum += p[dimension]
        # average of each dimension
        new_center.append(dim_sum / float(len(points)))

    return new_center

def update_centers(data_set, assignments):
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)

    for points in new_means.values():
        centers.append(point_avg(points))

    return centers

def distance(a, b):
    dimensions = len(a)

    _sum = 0
    for dimension in range(dimensions):
        difference_seq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_seq

    return sqrt(_sum)

def assign_points(data_points, centers):
    assignments = []
    for point in data_points:
        shortest = float('Inf')
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index + 1)

    return assignments

def generate_k(data_set, k):
    centers = []
    dimensions = len(data_set[0])
    min_max = defaultdict(int)

    for point in data_set:
        for i in range(dimensions):
            val = point[i]
            min_key = 'min_%d' % i
            max_key = 'max_%d' % i
            if min_key not in min_max or val < min_max[min_key]:
                min_max[min_key] = val
            if max_key not in min_max or val > min_max[max_key]:
                min_max[max_key] = val

    for _k in range(k):
        rand_point = []
        for i in range(dimensions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]

            rand_point.append(uniform(min_val, max_val))

        centers.append(rand_point)
    return centers


def k_means(dataset, k):
    # k_points = generate_k(dataset, k)
    # print(k_points)
    k_points = [[14.21,4.04,2.44,18.9,111,2.85,2.65,.3,1.25,5.24,.87,3.33,1080],
                [13.86,1.51,2.67,25,86,2.95,2.86,.21,1.87,3.38,1.36,3.16,410],
                [12.85,3.27,2.58,22,106,1.65,.6,.6,.96,5.58,.87,2.11,570]]
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    times = 0
    while assignments != old_assignments:
        times += 1
        # print('times is :', times)
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
        if times > 100:
            break
    return (assignments, dataset)

filename = "data\wine.data"
data_set,y_true = get_data(filename)

k = 3

y_pred, cluster = k_means(data_set, k)
print("最终划分结果：",y_pred)

Macro_F1,Micro_F1,s_score = (get_score(data_set,y_true,y_pred))
print('kmeans的Macro_F1,Micro_F1,；轮廓系数分别为:',Macro_F1, Micro_F1, s_score)
# for i in range(len(data_set)):
#     print("点："+str(data_set[i])+"  属于簇："+str(assignments[i]))
