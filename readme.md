# Heart Disease Prediction 🫀
A machine learning project that predicts whether a patient has heart disease based on clinical attributes from the Cleveland Heart Disease dataset.

## Project Overview
This project applies supervised machine learning techniques to predict heart disease presence (binary classification: `condition` = 0 or 1).  
It includes:

- Data loading and cleaning  
- Exploratory Data Analysis (EDA)  
- Feature scaling  
- Model training (KNN by default)  
- Evaluation using accuracy, confusion matrix, and classification report  
- Saving the trained model for future use  

The project is implemented entirely in Python using `pandas`, `numpy`, and `scikit-learn`.

---

## Dataset
The dataset is located in:

```
data/heart_cleveland_upload.csv
```

### **Target Column**
- `condition` — 0 = no disease, 1 = disease present

### **Features**
- age  
- sex  
- cp  
- trestbps  
- chol  
- fbs  
- restecg  
- thalach  
- exang  
- oldpeak  
- slope  
- ca  
- thal  

Dataset shape: **297 rows × 14 columns**  
No missing values.

---

## Methods Used

### **Preprocessing**
- Dropping missing values (none present)
- Train/test split (80/20)
- Feature scaling via `StandardScaler`

### **Model**
- **K-Nearest Neighbors (KNN)** classifier  
- Hyperparameters can be tuned (e.g., number of neighbors)

### **Evaluation**
- Accuracy score  
- Confusion matrix  
- Classification report (precision, recall, F1)

---

## Example Output

```
Accuracy: 0.86
Confusion Matrix:
 [[23  4]
  [ 3 29]]

Classification Report:
               precision    recall  f1-score   support
           0       0.88      0.85      0.86        27
           1       0.88      0.91      0.89        32
```

---

## 🧪 Running the Project

### **1. Install requirements**
```bash
pip install pandas numpy scikit-learn joblib
```

### **2. Run the script (PyCharm or terminal)**
```bash
python lookUp.py
```

### **3. The trained model will be saved as**
```
heart_model.pkl
```

---

## Project Structure
```
HeartDiseasePrediction/
│
├── data/
│   └── heart_cleveland_upload.csv
│
├── lookUp.py
├── README.md
└── .venv/
```

---

## Future Improvements
- Add Logistic Regression, Random Forest, and SVM models  
- Hyperparameter tuning using GridSearchCV  
- Advanced EDA visualizations (heatmap, pairplot, histograms)  
- Deployment as a Flask API  
- Interactive UI with Streamlit  
- Model comparison dashboard  

---

## Technologies Used
- Python 3.10+
- pandas
- numpy
- scikit-learn
- joblib
- PyCharm IDE

---

