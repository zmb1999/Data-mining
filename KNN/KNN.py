from __future__ import print_function
from sklearn.model_selection import StratifiedKFold
import numpy as np

def classify0(inX, dataSet, labels, k):

    # 1. 距离计算
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    # print('distances.argsort()=', sortedDistIndicies)

    # 2. 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    maxClassCount = max(classCount, key=classCount.get)
    return maxClassCount

# ----------------------------------------------------------------------------------------
def file2matrix(filename):

    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    # 计算每种属性的最大值、最小值、范围
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 极差
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    # 设置测试数据的的一个比例（训练数据集比例=1-hoRatio）
    hoRatio = 0.1  # 测试范围,一部分测试一部分作为样本
    # 从文件中加载数据
    datingDataMat, datingLabels = file2matrix('./data/datingTestSet2.txt')
    #数据可视化
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
    plt.show()
    # 归一化数据
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # print(normMat,"\n***************")
    # m 表示数据的行数，即矩阵的第一维
    m = normMat.shape[0]
    # 设置测试的样本数量， numTestVecs:m表示训练样本的数量
    numTestVecs = int(m * hoRatio)
    print('numTestVecs=', numTestVecs)
    spl = int(1 / hoRatio)
    skf = StratifiedKFold(n_splits=spl, random_state=None, shuffle=True)
    sum = 0
    for train_index, test_index in skf.split(normMat,datingLabels):
        errorCount = 0.0
        train_X, train_y = np.array(normMat)[train_index], np.array(datingLabels)[train_index]
        test_X, test_y = np.array(normMat)[test_index], np.array(datingLabels)[test_index]
        for i in range(numTestVecs - 1):
            # 对数据测试
            classifierResult = classify0(test_X[i, :], train_X, train_y, 3)
            if (classifierResult != test_y[i]): errorCount += 1.0
        sum += errorCount
        print("本次验证错误率为： ",(errorCount / float(numTestVecs)),"错误个数为： " ,errorCount)
    print("*****************************************")
    print("平均错误率为： ",((sum / spl) / float(numTestVecs)),"平均错误个数为： " ,(sum / spl))

def classifyPerson():
    resultList = ['不喜欢的人', '魅力一般的人', '极具魅力的人']
    datingDataMat, datingLabels = file2matrix('./data/datingTestSet2.txt')
    ffMiles = float(input("每年获得的飞行常客里程数："))
    percentTats = float(input("玩游戏所耗时间百分比："))
    iceCream = float(input("每周消费的冰淇淋公升数："))
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels, 3)
    print("这个人是", resultList[classifierResult - 1])

if __name__ == '__main__':
    datingClassTest()
    classifyPerson()
