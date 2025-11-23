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
<img width="1150" height="964" alt="image" src="https://github.com/user-attachments/assets/537b8ca4-7083-477a-a390-c555dad49eac" />
</p>

- ROC Curves (One-vs-Rest) : Plots the ROC curves for each digit class
  
  ```python
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], model.predict_proba(x_test_pca)[:, i])

<p align="center">
<img width="1193" height="870" alt="image" src="https://github.com/user-attachments/assets/47cf4efb-538c-45ab-aef4-93c2efbe6432" />
</p>

- Classification Report Heatmap : Shows precision, recall, and F1-score per class
  
    ```python
    sns.heatmap(df_report.iloc[:-1, :].T, annot=True, fmt=".2f", cmap="Blues")

<p align="center">
<img width="1758" height="1045" alt="image" src="https://github.com/user-attachments/assets/70cd1605-1257-41d3-9ce1-0c8c0bbcceac" />
</p>


  ## Author
  
  Omar Alethamat

  LinkedIn : https://www.linkedin.com/in/omar-alethamat-8a4757314/

  ## License

  This project is licensed under the MIT License — feel free to use, modify, and share with attribution.


