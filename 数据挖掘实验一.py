import numpy as np
import math
from sklearn import preprocessing
import pandas
from scipy.io import arff

def getdata(data):
    instance = data.shape[0]
    attribute = data.shape[1] - 1
    Defective = data[:, attribute]
    data = np.delete(data, attribute, axis=1)
    return data,instance,attribute,Defective

def getDefectiveNum(Defective):
    DefectiveNum = []
    for i,v in enumerate(Defective):
        if v == b'Y':
            DefectiveNum.append(0)
        else:
            DefectiveNum.append(1)
    DefectiveNum = [float(i) for i in DefectiveNum]
    return DefectiveNum

def getminposition(mylist):
    minnum = min(mylist)
    position = mylist.index(minnum)
    return position

def delete(list1,list2):
    deltalist = []
    for i,v in enumerate(list1):
        deltalist.append(abs(list1[i]-list2[i]))
    return deltalist

def add(list1,list2):
    sumlist = []
    for i,v in enumerate(list1):
        sumlist.append(abs(list1[i]+list2[i]))
    return sumlist

def findAllIndexInList(aim, List):
    pos = 0
    index = []
    for each in List:
        if each == aim:
            index.append(pos)
        pos += 1
    return index

def CreateNewListByIndex(Index, List):
    newList = []
    List = list(List)
    Index = list(Index)
    for each in Index:
        newList.append(List[each])
    return newList

def Pi(aim, List):
    length = len(list(List))
    aimcount = (list(List)).count(aim)
    pi = (float)(aimcount/length)
    return pi

def entropy(data):
    data1 = np.unique(data)
    resultEn = 0
    for each in data1:
        pi = Pi(each, data)
        resultEn -= pi * math.log(pi, 2)
    return resultEn

def conditionalEntropy(dataX, dataY):
    YElementsKinds = list(np.unique(dataY))
    resultConEn = 0
    for uniqueYEle in YElementsKinds:
        YIndex = findAllIndexInList(uniqueYEle, dataY)
        dataX_Y = CreateNewListByIndex(YIndex, dataX)
        HX_uniqueYEle = entropy(dataX_Y)
        pi = Pi(uniqueYEle, dataY)
        resultConEn += pi * HX_uniqueYEle
    return resultConEn

def transpose(M):
    return [list(row) for row in zip(*M)]

filename = 'CM1'

dataset, mate = arff.loadarff(filename+'.arff')
#print(dataset)
df = pandas.DataFrame(dataset)
#print(df)
#dataset_ls = list(dataset)
originData = np.array(df)
#print(originData.shape)
data, instance, attribute, Defective = getdata(originData)
#print(data, "\n",instance,"\n", attribute,"\n", Defective)
#print(Tdata)
DefectiveNum = getDefectiveNum(Defective)

#标准化
data = data.astype('float64')
scaled = preprocessing.scale(data)
#print(scaled)

#计算k
k = np.sum(Defective == b'N')/np.sum(Defective == b'Y')
k = int(k)

#print(k)
#标准化距离
ed = np.zeros((instance, instance))
alldata = instance*instance
for i in range(0, instance):
    for j in range(0, instance):
        ed[i][j] = np.linalg.norm(scaled[i,:] - scaled[j,:])
#print(ed)

#临近样本
rank = []
for i in range(0, instance):
    rank.append(pandas.Series(ed[i, :]).rank(method='min').tolist())
#print(rank)

nearest = []

for index,i in enumerate(rank):
    n = []
    num = 0
    while 1:
        position = getminposition(i)
        if Defective[position] == b'Y' or position == index:
            i[position] = max(i)
        else:
            n.append(position)
            i[position] = max(i)
            num += 1
        if num == k:
            break
    nearest.append(n)
#print(nearest)

#特征差值
delta = []
for i,v in enumerate(data):
        d = []
        for j,w in enumerate(nearest[i]):
            d.append(delete(data[i], data[w]))
        delta.append(d)
#print(delta)

#特征权重
W = np.zeros(attribute)
#print(W)
for i,v in enumerate(delta):
    if Defective[i] == b'Y':
        for j in v:
            W = add(W,pandas.Series(j).rank(method='min').tolist())
print(W)

#特征排序列表
WRank = pandas.Series(W).rank(method='min').tolist()
fRank = []
flag = 0
while 1:
    for i, v in enumerate(WRank):
        if v == max(WRank):
            fRank.append(i)
            WRank[i] = -1
            flag += 1
        if flag == attribute:
            break
    if flag == attribute:
        break

print(filename+"特征排序列表:",fRank)

M = []
cnt = 0
Tdata = transpose(data)
for i in range(attribute):
    X = []
    for j in range(attribute):
        k = 4
        dx = pandas.cut(Tdata[i], k, labels=range(k))
        dy = pandas.cut(Tdata[j], k, labels=range(k))
        HX = entropy(dx)
        HY = entropy(dy)
        HXY = conditionalEntropy(dx,dy)
        IG = HX - HXY
        SU = 2 * IG / (HX + HY)
        # if SU > 1 or SU < 0:
        #     print("error")
        if SU > 0.5:
            cnt += 1
        #print(SU)
        X.append(SU)
    M.append(X)
M = np.array(M)
print("M的大小：",M.shape)
print("矩阵M：\n",M)
print("SU大于0.5的有：",cnt)