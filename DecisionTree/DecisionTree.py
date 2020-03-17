from __future__ import print_function

import operator
from math import log
from collections import Counter
import decisionTreePlot as dtPlot

def creatDataSet():
    '''
    |不浮出水面可以生存          | 是否有脚蹼     |属于鱼类
    |1-是----------------------|-是------------|是
    |2-是---------------------  |-是------------|是
    |3-是---------------------  |-否------------|否
    |4-否---------------------  |-是------------|否
    |5-否---------------------  |-是------------|否
    '''
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def calcShannonEnt(dataSet):
    # 统计标签出现的次数
    label_count = Counter(data[-1] for data in dataSet)
    # 计算概率
    probs = [float(p[1]) / len(dataSet) for p in label_count.items()]
    # 计算香农熵
    shannonEnt = sum([-p * log(p, 2) for p in probs])
    return shannonEnt

def splitDataSet(dataSet, index, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[index] == value:
            reduceFeatVec = featVec[:index]
            reduceFeatVec.extend(featVec[index + 1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    # Label标签信息熵
    baseEntropy = calcShannonEnt(dataSet)
    # 最优的信息增益值, 和最优的Featurn编号
    bestInfoGain, bestFeature = 0.0, -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和。
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # gain[信息增益]: 划分数据集前后的信息变化， 获取信息熵最大的值
        # 信息增益是熵的减少或者是数据无序度的减少。最后，比较所有特征中的信息增益，返回最好特征划分的索引值。
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    print("sortedClassCount:" + sortedClassCount)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 第一个停止条件：所有的类标签(Label)完全相同，则直接返回该类标签停止划分。
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最优的列，得到最优列对应的label含义
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    # 删除已经使用特征标签
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的标签label
        subLabels = labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        # print('myTree', value, myTree)
    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

def fishTest():
    myData, labels = creatDataSet()
    import copy
    myTree = createTree(myData, copy.deepcopy(labels))
    print(myTree)
    print(classify(myTree, labels, [1, 0]))
    #可视化创建好的决策树
    dtPlot.createPlot(myTree)

if __name__ == '__main__':
    fishTest()