from sklearn.datasets import fetch_openml

import numpy as np

from sklearn.model_selection import train_test_split , GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report , ConfusionMatrixDisplay , confusion_matrix , roc_curve , roc_auc_score

from sklearn.decomposition import PCA

from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd



mnist = fetch_openml('mnist_784', version=1)

x = mnist.data

y =mnist.target.astype(np.int8) 

mask = (0) | (1) | (2)

x_filtered = x[mask]

y_filtered = y[mask]



x_train , x_test , y_train , y_test = train_test_split(

    x_filtered , y_filtered , test_size=0.2 , random_state=42
)

pca = PCA(n_components=20)

pca = pca.fit(x_train)

x_train_pca = pca.transform(x_train)


x_test_pca = pca.transform(x_test)


parm_model = {'C' : np.logspace(-4 , 4 , 20) , 'penalty':['l2']}

model = GridSearchCV(LogisticRegression(solver='lbfgs' , max_iter=2000) , parm_model , cv=3)

model.fit(x_train_pca , y_train)

y_pred = model.predict(x_test_pca)




conf_matr = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matr, display_labels=model.classes_)

disp.plot(cmap='Blues')





plt.figure(figsize=(6,4))
 
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2]) 

for i in range(3):

    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], model.predict_proba(x_test_pca)[:, i])

    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc_score(y_test_binarized[:, i], model.predict_proba(x_test_pca)[:, i]):.2f})")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()





report = classification_report(y_test, y_pred, output_dict=True)  

df_report = pd.DataFrame(report).transpose()  

plt.figure(figsize=(10, 5))

sns.heatmap(df_report.iloc[:-1, :].T, annot=True,  fmt=".2f", cmap="Blues",)

plt.title("Classification Report Heatmap")

plt.xlabel("Metrics")

plt.ylabel("Classes")

plt.show()