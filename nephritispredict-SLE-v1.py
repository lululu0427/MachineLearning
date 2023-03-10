#載入必要的Python套件
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
#載入資料集
data = pd.read_csv('sle_data.csv')
#印出資料集的前五行，檢查是否載入成功
print(data.head())
#特徵為'proteinuria', 'nephritis', 'serum_creatinine'，目標變量為SLE severity
X = data[['proteinuria', 'nephritis', 'serum_creatinine']]
y = data['severity']
#將資料集分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#訓練模型
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
#使用測試集來評估模型的表現
y_pred = rfc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cm)

##new_patient = [[500, 4, 5]]
##severity_prediction = rfc.predict(new_patient)
##print('Severity prediction for new patient:', severity_prediction[0])