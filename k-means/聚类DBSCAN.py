
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn import metrics

def get_data(filename):
    data_set = []
    target = []
    with open(filename, "r") as f:
        for line in f:
            data_set.append(list(map(float, line.split(',')))[1:])
            target.append(list(map(float, line.split(',')))[0])
    return data_set,target

def index_divide_by_class(y_pred):
    temp1 = []
    temp2 = []
    temp3 = []
    # 类别可能的情况
    class_options = [i for i in set(y_pred) if i != -1]
    for i in range(len(y_pred)):
        if y_pred[i] == class_options[0]:
            temp1.append(i)
        elif y_pred[i] == class_options[1]:
            temp2.append(i)
        elif y_pred[i] == class_options[2]:
            temp3.append(i)
        else:
            continue
    return [temp1,temp2,temp3]

def cal_class_estimate_params(original_class, predict_class):
    # TP\FP\FN
    class_estimate_params = np.zeros((3,3))
    for i in range(3):
        original = original_class[i]
        intersection_id = -1
        intersection_num = 0
        for j in range(3):
            predict = predict_class[j]
            temp_num = len(np.intersect1d(original, predict))
            if temp_num > intersection_num:
                intersection_num = temp_num
                intersection_id = j
        # print('class'+str(i)+' --> '+'predict'+str(intersection_id))
        predict = predict_class[intersection_id]
        class_estimate_params[i][0] = intersection_num
        class_estimate_params[i][1] = len(np.setxor1d(predict, original))
        class_estimate_params[i][2] = len(np.setxor1d(original, predict))
    return class_estimate_params


def cal_macro_and_micro(estimate_params):
    f1s = []
    for i in range(3):
        TP = estimate_params[i][0]
        FP = estimate_params[i][1]
        FN = estimate_params[i][2]
        f1s.append((2 * TP) / (2 * TP + FP + FN))
    macro_f1 = np.average(f1s)

    temp_sum = estimate_params.sum(axis=0)
    TP = temp_sum[0]
    FP = temp_sum[1]
    FN = temp_sum[2]
    micro_f1 = (2 * TP) / (2 * TP + FP + FN)
    return macro_f1, micro_f1


x_data, y_data = get_data("data\wine.data")

standard_scaler = StandardScaler()
x_data = standard_scaler.fit_transform(x_data)

eps = 2.16
min_samples = 5
model_dbscan = DBSCAN(eps=eps, min_samples=min_samples)
model_dbscan.fit(x_data)
y_pred = model_dbscan.labels_
print(y_pred)
class1 = np.arange(0,59)
class2 = np.arange(59,130)
class3 = np.arange(130,178)

score_silhouette = silhouette_score(x_data, y_pred, metric='euclidean')

predict_class = index_divide_by_class(y_pred)
original_class = [class1, class2, class3]
class_estimate_params = cal_class_estimate_params(original_class, predict_class)
# print(class_estimate_params)

print('DBCSAN的Macro_F1,Micro_F1,；轮廓系数分别为:',cal_macro_and_micro(class_estimate_params),score_silhouette)
#
print(predict_class[0], len(predict_class[0]))
print(predict_class[1], len(predict_class[1]))
print(predict_class[2], len(predict_class[2]))
#
