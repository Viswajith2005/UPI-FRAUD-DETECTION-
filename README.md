# 🔐 UPI Fraud Detection Web App

A powerful and interactive **Streamlit web application** that detects potential fraud in UPI (Unified Payments Interface) transactions using machine learning. This project combines real-world datasets, feature engineering, and a production-ready frontend to showcase fraud classification in action.

> 🚀 Built as part of a capstone project to demonstrate end-to-end ML deployment.

---

## 📸 App Preview

Coming soon: Live deployment link

![App Screenshot Placeholder](<img width="1912" height="903" alt="image" src="https://github.com/user-attachments/assets/9ee2f99f-2fbc-477d-b3de-21ce2315b463" />
)

---

## 🎯 Key Features

✅ **Real-time UPI fraud detection**  
✅ **Upload and analyze batch transactions**  
✅ **Built with CatBoost – highly accurate classifier**  
✅ **Streamlit-based user interface for demo and testing**  
✅ **Insightful visualizations and EDA**  
✅ **Modular code for easy extension and improvement**

---

## 🚀 Live Demo

🔗 [Live Web App]([https://your-streamlit-link.streamlit.app](https://upi-fraud-detector.streamlit.app/)) *(Replace after deployment)*  


---

## 🧠 Machine Learning Overview

- **Model:** CatBoost Classifier  
- **Metrics Tracked:** Accuracy, Precision, Recall, F1-Score  
- **Training Method:** Cleaned UPI transaction data  
- **Preprocessing:** Handling duplicates, missing values, feature scaling  
- **Final Output:** Binary fraud prediction (`Fraud` / `Not Fraud`)

---

## 🛠️ Tech Stack

| Area         | Tools Used                             |
|--------------|----------------------------------------|
| Frontend     | Streamlit                              |
| Backend      | Python                                 |
| ML/Modeling  | CatBoost, Scikit-learn, Pandas, NumPy   |
| Visualization| Seaborn, Matplotlib                    |
| Deployment   | Streamlit Community Cloud              |

---

## 📂 File Structure
📦 UPI-Fraud-Detection/
├── streamlit_app.py # Main Streamlit web app
├── requirements.txt # Python dependencies
├── UPI Fraud Detection Final.pkl # Trained ML model
├── sample.csv # Sample transactions for testing
├── upi_fraud_dataset.csv # Main dataset
├── UPI Latest data.csv # Updated dataset
├── notebooks/
│ ├── UPI_Fraud_detection.ipynb
│ ├── Data analysis and training
│ └── Model evaluation
└── catboost_info/ # Model training logs

📄 Published Paper: [UPI FRAUD DETECTION]([https://link-to-your-paper.com](https://ieeexplore.ieee.org/abstract/document/11052942))  
Presented at [IEEE Conference], 2025

---

## 📦 Installation Guide

Clone the repository and run the app locally:

```bash
# Step 1: Clone the repo
git clone https://github.com/yourusername/upi-fraud-detection.git
cd upi-fraud-detection

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the Streamlit app
streamlit run streamlit_app.py



📈 Sample Use Case
Upload a CSV file of UPI transactions

App predicts which transactions are fraudulent

View predictions and model confidence

Use the app in fraud monitoring or as an ML showcase



