# Heart disease prediction

Predicts whether a patient has heart disease from 13 clinical features (age, chest pain type, cholesterol, max heart rate, etc.). Compares Logistic Regression and Random Forest, with median imputation for the 6 missing values.

**Data:** UCI Heart Disease, Cleveland subset (303 patients), included in `data/`.
Source: https://archive.ics.uci.edu/dataset/45/heart+disease

**Run**
```
pip install -r requirements.txt
python main.py
```
Output: cross-validation and test accuracy, report, top features (saved example in `run_output.txt`).
