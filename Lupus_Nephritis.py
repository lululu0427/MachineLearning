import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import time
print("請將執行檔以及病患資料放置於桌面")
filename = input("請輸入檔案名稱：")
#Lupus_Nephritis
filepath = "C:\\Users\\user\\Desktop\\" + filename+ ".csv"
dataset = pd.read_csv(filepath)
num_columns = dataset.shape[1]

# 檢查欄位數量是否符合預期
if len(dataset.columns) == num_columns:
    # 檢查每個欄位的資料型態是否符合預期
    if dataset.dtypes[0] == object and dataset.dtypes[1] == float:
        print('資料載入成功')
    else:
        print('資料格式不符合預期')
else:
    print('資料欄位數量不符合預期')
# 載入已經標記好的資料集，並將其分為特徵和標籤
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
#使用SimpleImputer填補缺失值
imp = SimpleImputer(strategy='constant',fill_value= 99999)
X = imp.fit_transform(X)
# 特徵篩選
selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X, y)
# 取得被選中的特徵名稱
selected_features = dataset.columns[1:-1][selector.get_support()]
# 資料集分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=0)
# 對特徵進行特徵縮放
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# 使用邏輯回歸建立模型
classifier = LogisticRegression(random_state=0,max_iter=10000)
classifier.fit(X_train, y_train)
#選擇重要特徵
lr= LogisticRegression(penalty='l1',solver='liblinear',C=0.1)
sfm=SelectFromModel(lr,threshold=0.1)
X_train_selected = sfm.fit_transform(X_train,y_train)
selected_feature = dataset.columns[1:-1][sfm.get_support(indices=True)]
print("最有意義之特徵:", ", ".join(selected_feature))
# 評估訓練集模型的準確性
y_tpred = classifier.predict(X_train)
tcm = confusion_matrix(y_train, y_tpred)
tacc = accuracy_score(y_train, y_tpred)
tauc = roc_auc_score(y_train, y_tpred)
tp, fn, fp, tn = tcm.ravel()
trecall = tp / (tp + fn)
tf1_score = f1_score(y_train, y_tpred)
# 評估測試集模型的準確性
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
p, n = np.bincount(y_test)
tp, fn, fp, tn = cm.ravel()
recall = tp / (tp + fn)
nf1_score = f1_score(y_test, y_pred)
#K-fold Cross-Validation
scores = cross_val_score(classifier, X_new,y, cv=5)

moduledata = input("是否顯示模型運行之數據(y/n)")
if moduledata == 'y':
    print("Selected features:", list(selected_features))
    print("\nK-fold Cross-Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("\n以下為訓練集之數據:")
    print("訓練集之混淆矩陣:\n",tcm)
    print("訓練集模型準確率為：",tacc)
    print("訓練集AUC 值為：", tauc)
    print("訓練集模型recall值為:", trecall)
    print("訓練集模型F1-score為:", tf1_score)
    #tariniing set數據視覺化呈現
    title = 'Model Performance on Training Set'
    data = {'Accuracy': tacc, 'AUC': tauc, 'Recall':trecall, 'F1-score': tf1_score}
    # 設定圖表大小和字體大小
    plt.figure(figsize=(10,6))
    plt.rcParams.update({'font.size': 14})
    # 將數據轉換為list
    names = list(data.keys())
    values = list(data.values())
    # 繪製柱狀圖
    plt.bar(names, values)
    # 設定標題和標籤
    plt.title(title)
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    # 顯示圖表
    plt.show()
    print("\n以下為測試集之數據:")
    print("測試集之混淆矩陣:\n",cm)
    print("測試集模型準確率為：",acc)
    print("測試集AUC 值為：", auc)
    print("測試集模型recall值為:", recall)
    print("測試集模型F1-score為:", nf1_score)
    #test set數據視覺化呈現
    title = 'Model Performance on Test Set'
    data = {'Accuracy': acc, 'AUC': auc, 'Recall':recall, 'F1-score': nf1_score}
    # 設定圖表大小和字體大小
    plt.figure(figsize=(10,6))
    plt.rcParams.update({'font.size': 14})
    # 將數據轉換為list
    names = list(data.keys())
    values = list(data.values())
    # 繪製柱狀圖
    plt.bar(names, values)
    # 設定標題和標籤
    plt.title(title)
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    # 顯示圖表
    plt.show()
input("\n輸入enter以結束程式")
