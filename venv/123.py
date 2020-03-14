# 生成数据模块
from sklearn.datasets import make_blobs
# k-means模块
from sklearn.cluster import KMeans
# 评估指标——轮廓系数,前者为所有点的平均轮廓系数，后者返回每个点的轮廓系数
from sklearn.metrics import silhouette_score, silhouette_samples

import numpy as np
import matplotlib.pyplot as plt
# 生成数据
x_true, y_true = make_blobs(n_samples= 600
                            , n_features= 2, centers= 4, random_state= 1)

# 绘制出所生成的数据
plt.figure(figsize= (6, 6))
plt.scatter(x_true[:, 0], x_true[:, 1], c= y_true, s= 10)
plt.title("Origin data")
plt.show()
# 根据不同的n_centers进行聚类
n_clusters = [x for x in range(3, 6)]

for i in range(len(n_clusters)):
    # 实例化k-means分类器
    clf = KMeans(n_clusters=n_clusters[i])
    y_predict = clf.fit_predict(x_true)

    # 绘制分类结果
    plt.figure(figsize=(6, 6))
    plt.scatter(x_true[:, 0], x_true[:, 1], c=y_predict, s=10)
    plt.title("n_clusters= {}".format(n_clusters[i]))

    ex = 0.5
    step = 0.01
    xx, yy = np.meshgrid(np.arange(x_true[:, 0].min() - ex, x_true[:, 0].max() + ex, step),
                         np.arange(x_true[:, 1].min() - ex, x_true[:, 1].max() + ex, step))

    zz = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    zz.shape = xx.shape

    plt.contourf(xx, yy, zz, alpha=0.1)

    plt.show()

    # 打印平均轮廓系数
    s = silhouette_score(x_true, y_predict)
    print("When cluster= {}\nThe silhouette_score= {}".format(n_clusters[i], s))

    # 利用silhouette_samples计算轮廓系数为正的点的个数
    n_s_bigger_than_zero = (silhouette_samples(x_true, y_predict) > 0).sum()
    print("{}/{}\n".format(n_s_bigger_than_zero, x_true.shape[0]))

