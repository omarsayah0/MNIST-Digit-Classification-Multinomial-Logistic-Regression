# MNIST Digit Classification (Logistic Regression + PCA)

## About
In this repository, I implemented a **Multinomial Logistic Regression** model to classify digits **(0, 1, 2)** from the **MNIST dataset**.  
The project integrates **Principal Component Analysis (PCA)** for dimensionality reduction, **GridSearchCV** for hyperparameter tuning, and evaluates performance using **Confusion Matrix**, **ROC Curves**, and a **Classification Report Heatmap**.  
The dataset is limited to digits 0–2 to ensure faster execution.

---

## Files
- `multinomial-logistic-regression.py` → Python script implementing the classification model.
- `mnist_784` → Dataset automatically fetched via `fetch_openml()`.

---

## Steps Included

### 1️⃣ Data Loading and Filtering :

- Loaded the MNIST dataset using `fetch_openml('mnist_784')`
- Converted labels to integers using:
  
  ```python
    y = mnist.target.astype(np.int8)
- Filtered dataset to include only digits 0, 1, and 2 for efficiency.

### 2️⃣ Dimensionality Reduction using PCA

- Applied PCA to reduce dimensionality to 20 components:
  
  ```python
    pca = PCA(n_components=20)
  x_train_pca = pca.fit_transform(x_train)
  x_test_pca = pca.transform(x_test)

### 3️⃣ Model Training and Optimization

- Used GridSearchCV to find the best regularization parameter C:
  
    ```python
        params = {'C': np.logspace(-4, 4, 20), 'penalty': ['l2']}
        model = GridSearchCV(LogisticRegression(solver='lbfgs', max_iter=2000), params, cv=3)
        model.fit(x_train_pca, y_train)


### How to Run

1- Install Dependencies:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

2-Run :
```bash
python multinomial-logistic-regression.py
```

3- Model Evaluation

- Confusion : Matrix Displays correct vs. incorrect classifications
  
   ```python
     conf_matr = confusion_matrix(y_test, y_pred)
     ConfusionMatrixDisplay(confusion_matrix=conf_matr, display_labels=model.classes_).plot(cmap='Blues')
  
<p align="center">
  <img src="https://github.com/user-attachments/assets/a7eef9ac-9eab-4171-9499-8d571cb7fff6" alt="download" width="515" height="432">
</p>

- ROC Curves (One-vs-Rest) : Plots the ROC curves for each digit class
  
  ```python
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], model.predict_proba(x_test_pca)[:, i])

<p align="center">
  <img width="536" height="393" alt="download (1)" src="https://github.com/user-attachments/assets/b82f2cf4-cf0e-412d-89e5-b758cde519d0" />
</p>

- Classification Report Heatmap : Shows precision, recall, and F1-score per class
  
    ```python
    sns.heatmap(df_report.iloc[:-1, :].T, annot=True, fmt=".2f", cmap="Blues")

<p align="center">
  <img width="786" height="470" alt="download (2)" src="https://github.com/user-attachments/assets/ef43a062-cbd6-44c2-ab23-ede936338dcc" />
</p>


  ## Author
  
  Omar Alethamat

  LinkedIn : https://www.linkedin.com/in/omar-alethamat-8a4757314/

  ## License

  This project is licensed under the MIT License — feel free to use, modify, and share with attribution.


