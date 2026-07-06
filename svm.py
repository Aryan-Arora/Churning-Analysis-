import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



df=pd.read_csv("Churn_Modelling.csv")
df=df.drop(columns=["RowNumber","CustomerId","Surname"])

X=df.drop(columns=["Exited"])
Y=df["Exited"]

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

le=LabelEncoder()
X_train["Gender"]=le.fit_transform(X_train["Gender"])
X_test["Gender"]=le.transform((X_test["Gender"]))

X_train=pd.get_dummies(X_train, columns=["Geography"],drop_first=True)
X_test=pd.get_dummies(X_test, columns=["Geography"],drop_first=True)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

svm_model=SVC(kernel="rbf", random_state=42)
svm_model.fit(X_train,Y_train)
Y_pred_svm=svm_model.predict(X_test)
print(Y_pred_svm)
print(Y_test)
print(accuracy_score(Y_test, Y_pred_svm))
