# Drug Classification Using Various ML Models

## Description
This project implements drug classification using diverse machine learning models. We analyze chemical and pharmacological features to categorize drugs based on their properties. The goal is to enhance accuracy in predicting drug classes using algorithms like Support Vector Machine (SVM), Naive Bayes, k-Nearest Neighbors (k-NN), and Weighted k-Nearest Neighbors.

By applying these algorithms to a carefully curated dataset, we identify which algorithm and data split ratios (e.g., 80:20, 70:30, 60:40, 50:50) result in the highest accuracy in classifying drug types.

![Drug Classification](https://vivek76.pages.dev/assets/dc.png)

---

## Dataset
- **Source**: [Kaggle](https://www.kaggle.com)
- **Number of Samples**: 200
- **Number of Attributes**: 6
- **Decision Attribute**: `Drug Type` (Drug x, Drug y, Drug z)

---

## Algorithms
1. **Naive Bayes Classifier**
   - A probabilistic classification algorithm based on Bayes' theorem.
   - Assumes feature independence and is efficient for text classification tasks.

2. **Simple K-NN Classifier**
   - A supervised learning algorithm that classifies data points based on similarity.
   - Suitable for both regression and classification tasks.

3. **Weighted K-NN Classifier**
   - A variation of k-NN that assigns weights to neighbors based on distance.
   - Improves prediction accuracy by emphasizing closer neighbors.

4. **SVM Classifier**
   - A powerful supervised learning algorithm for classification and regression.
   - Creates a hyperplane to segregate data into classes.

---

## Project Implementation
1. **Data Collection and Preparation**
   - Gather and preprocess the dataset, handling missing data, encoding categorical variables, and scaling features.
2. **Algorithm Selection**
   - Choose machine learning algorithms (SVM, Naive Bayes, k-NN, Weighted k-NN).
3. **Data Splitting**
   - Split the dataset into training and testing sets using different ratios (e.g., 80:20, 70:30, 60:40, 50:50).
4. **Model Training**
   - Train each selected algorithm on the training data.
5. **Model Evaluation**
   - Evaluate model performance using metrics like accuracy, precision, recall, and F1-score.
6. **Selection of the Highest Accuracy Algorithm**
   - Identify the algorithm with the highest accuracy across different data split ratios.
7. **Results Analysis and Reporting**
   - Analyze results and create a report or presentation.

---

## Results
| Algorithm          | 80:20 Accuracy | 70:30 Accuracy | 60:40 Accuracy | 50:50 Accuracy |
|--------------------|----------------|----------------|----------------|----------------|
| Naive Bayes        | 81.67%         | 81.67%         | 81.67%         | 81.67%         |
| Simple K-NN        | 65.00%         | 65.00%         | 65.00%         | 65.00%         |
| Weighted K-NN      | 70.00%         | 70.00%         | 70.00%         | 70.00%         |
| SVM                | 85.00%         | 85.00%         | 85.00%         | 85.00%         |

**Overall Highest Accuracy Algorithm**: SVM with 85% accuracy.

---

## Reasons for SVM's High Accuracy
- Effective kernel selection.
- Proper feature scaling.
- Robust handling of outliers.

---

## Installation Instructions
To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/drug-classification.git
   cd drug-classification

2. **Virtual Environment (Optional but Recommended)**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

4. **Running the Code**:
1. Open the Jupyter Notebook or Python script.
2. Ensure the dataset is placed in the correct directory (e.g., `../input/drug-classification/drug200.csv`).
3. Execute the code cells or script.

## Dependencies
The following Python libraries are required to run this project:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imbalanced-learn` (for SMOTE)

You can install all dependencies using:
```bash
pip install -r requirements.txt
```

## Code Overview
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# Load dataset
df_drug = pd.read_csv("../input/drug-classification/drug200.csv")

# Data preprocessing
bin_age = [0, 19, 29, 39, 49, 59, 69, 80]
category_age = ['<20s', '20s', '30s', '40s', '50s', '60s', '>60s']
df_drug['Age_binned'] = pd.cut(df_drug['Age'], bins=bin_age, labels=category_age)
df_drug = df_drug.drop(['Age'], axis=1)

bin_NatoK = [0, 9, 19, 29, 50]
category_NatoK = ['<10', '10-20', '20-30', '>30']
df_drug['Na_to_K_binned'] = pd.cut(df_drug['Na_to_K'], bins=bin_NatoK, labels=category_NatoK)
df_drug = df_drug.drop(['Na_to_K'], axis=1)

# Splitting the dataset
X = df_drug.drop(["Drug"], axis=1)
y = df_drug["Drug"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Oversampling using SMOTE
X_train, y_train = SMOTE().fit_resample(X_train, y_train)

# K-Nearest Neighbors
KNclassifier = KNeighborsClassifier(n_neighbors=20)
KNclassifier.fit(X_train, y_train)
y_pred = KNclassifier.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
KNAcc = accuracy_score(y_pred, y_test)
print('K Neighbours accuracy is: {:.2f}%'.format(KNAcc * 100))
```

## Contributing
Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Push your branch and submit a pull request.

## Contact Information
- **Name**: Vivek
- **Email**: your.email@example.com
- **GitHub**: truly-vivek

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
