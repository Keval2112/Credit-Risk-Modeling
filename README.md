# 🧠 Credit Risk Modeling - Streamlit App

A machine learning project to **predict the probability of a borrower defaulting on a loan** using classification models like Logistic Regression and LightGBM. This project includes full preprocessing, feature engineering, model building, evaluation, and deployment using Streamlit.

---

## 🎯 Project Objective

To build a **credit risk prediction model** that determines whether a loan should be approved or not, based on the applicant’s financial and personal information.

---

## 🛠️ Tools & Technologies

- **Python 3**
- **Pandas, NumPy, Scikit-learn**
- **LightGBM, Logistic Regression**
- **Streamlit** (for app deployment)
- **Joblib** (for model saving/loading)

---

## 📁 Dataset Used

The dataset contains information like:

| Column Name         | Description                            |
|---------------------|----------------------------------------|
| Gender              | Male/Female                            |
| Married             | Marital status                         |
| Dependents          | Number of dependents                   |
| Education           | Graduate / Not Graduate                |
| Self_Employed       | Employment type                        |
| ApplicantIncome     | Applicant's monthly income             |
| CoapplicantIncome   | Coapplicant's monthly income           |
| LoanAmount          | Loan amount requested                  |
| Loan_Amount_Term    | Duration of the loan (in months)       |
| Credit_History      | 1.0 = Good, 0.0 = Bad/Unknown history  |
| Property_Area       | Urban / Semiurban / Rural              |
| Loan_Status         | Target variable (Y/N for loan approval)|

---

## 🔍 Feature Engineering

- ✅ Created `TotalIncome = ApplicantIncome + CoapplicantIncome`
- ✅ Created `Loan_to_Income = LoanAmount / TotalIncome`
- ✅ Handled missing values
- ✅ Label encoded categorical columns

---

## 📊 Models Used

- ✅ **Logistic Regression** (baseline)
- ✅ **LightGBM Classifier** (final model used in Streamlit app)

---

## 📈 Model Evaluation Metrics

- Accuracy
- Precision, Recall, F1 Score
- ROC-AUC Curve
- Confusion Matrix

---

## 🚀 Streamlit App

The Streamlit app allows users to:

- Input features like income, dependents, loan amount, etc.
- Get a prediction on whether the loan will be approved or not
- View prediction confidence score

### ✅ Run the App Locally

```bash
# Clone the repo
git clone https://github.com/Keval2112/credit-risk-modeling.git
cd credit-risk-modeling

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
