
# Delinquency Prediction & Risk-Based Collections Automation

## 📌 Project Title:
**Intelligent Delinquency Prediction and Ethical Collections Strategy using Explainable AI**

## 🧠 Overview:
This project uses machine learning and fairness-aware modeling to **predict delinquent customers** in a financial dataset and drive **automated, ethical collections decisions**. It integrates:
- **EDA on Database**
- **Imputing missing values** (only income was missing so imputation was done one by one according to the employment status)
- **Data preprocessing**
- **Imbalanced classification**
- **Explainability (SHAP)**
- **Fairness-aware metrics (Fairlearn)**
- **Automation design for agentic AI systems**

## 🔍 Problem Statement:
Changes in Database:
- Using EDA on Database
- Cleaning database using imputing techniques and filling missing values  
Loan/service-based companies face challenges in:
- Accurately identifying high-risk customers
- Reducing false negatives (missed delinquents)
- Maintaining fairness and transparency in decisions  
This system addresses those by **predicting delinquency risks** and **recommending actions with built-in guardrails**.

## ⚙️ Tools & Technologies:
- **Python**: scikit-learn, pandas, numpy, imbalanced-learn
- **SHAP**: Explainable AI insights
- **Fairlearn**: Bias mitigation & fairness metrics
- **Excel**: Data ingestion and stakeholder reporting
- **Matplotlib / Seaborn**: Visualizations

## 📊 Key Features:
- ✅ **Delinquent-only classification optimization**  
- ✅ **Recall-focused model tuning** (Class 1 sensitivity)
- ✅ **SHAP-based explainability** for feature impact
- ✅ **Fairness diagnostics** on sensitive features
- ✅ **Automated PDF & PPT generation** for stakeholders
- ✅ **Structured business report with ethical AI insights**

## 🧠 How it Works:
1. Preprocess customer data from Excel
2. Handle missing values, encode categorical variables
3. Balance classes (SMOTE, class weights)
4. Train & test classifiers (RandomForest, Logistic Regression, etc.)
5. Analyze recall, fairness, SHAP explanations
6. Auto-generate insights for business decision-making
7. Design next steps for agentic AI actions (e.g., reminders, escalations)

## 📈 Outcome Highlights:
- Identified **79 delinquent customers** with ~87% accuracy
- Built actionable **risk reports** for collections team
- Ensured **no biased predictions** toward age, gender, employment
- Prepared **PPT, PDF, and documentation** for leadership review

## 🧭 Next Steps (Agentic System Planning):
- Implement **trigger-based reminders & outreach automation**
- Set thresholds for **human oversight**
- Build **self-learning loop** from outcomes
- Integrate **audit logs** and **transparency layers**

## ⚖️ Ethical Considerations:
- Built under **Responsible AI guidelines**
- Ensures **fairness**, **transparency**, and **accountability**
- Protects user data and flags bias risks
