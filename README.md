# Credit Card Fraud Detection

A machine learning and deep learning based system to detect fraudulent credit card transactions from highly imbalanced financial data, built with **Python**, **Django**, and **MySQL**.

> 🎓 Final Year Major Project (B.Tech CSE – AI & ML) | JNTUH | 2026

---

##  Overview

Credit card fraud is a growing threat as digital and online payments continue to rise globally. One of the biggest challenges in building a fraud detection system is that fraudulent transactions make up a very small fraction of total transactions (highly **imbalanced data**), which causes traditional classifiers to be biased toward predicting legitimate transactions.

This project implements a **hybrid sampling-based fraud detection framework** that combines **oversampling** of fraudulent (minority) transactions and **undersampling** of legitimate (majority) transactions to create balanced datasets. Multiple machine learning and deep learning models are then trained and evaluated to accurately classify transactions as **fraudulent** or **legitimate**.

The dataset used contains **284,807 transactions** from European cardholders, of which only **~0.172%** are fraudulent — making this a real-world imbalanced classification problem.

---

##  Key Features

-  Upload and process real-world credit card transaction datasets
-  Hybrid sampling (oversampling + undersampling) to handle class imbalance (34:66 and 10:90 distributions)
-  Multiple ML algorithms implemented: **Naïve Bayes, K-Nearest Neighbors, Logistic Regression, Random Forest, XGBoost**
-  Deep learning model using a feedforward neural network (ReLU, Sigmoid, Dropout layers)
-  Model evaluation using Accuracy, Sensitivity (Recall), Specificity, Precision, F1-Score, Balanced Classification Rate, and Matthews Correlation Coefficient (MCC)
-  Interactive web interface (Django) to upload datasets, train models, and detect fraud on test data
-  Visual analytics — bar graphs comparing total, normal, and fraudulent transactions
-  MySQL backend for data storage and management

---

##  System Architecture

```
Customer Data (Data Warehouse)
        │
        ▼
   Fraud Rule Set
        │
        ▼
    Rules Engine
        │
        ▼
  Filter & Priority
        │
        ▼
   ML Algorithms  ───►  Fraud / Legitimate Classification
```

---

##  Tech Stack

| Category         | Technology                          |
|-------------------|--------------------------------------|
| Language          | Python 3                            |
| Web Framework     | Django (MVT Architecture)           |
| Frontend          | HTML, CSS, JavaScript               |
| Database          | MySQL                               |
| ML Libraries      | Scikit-learn, XGBoost               |
| DL Libraries      | TensorFlow / PyTorch                |
| Data Processing   | Pandas, NumPy                       |
| Visualization     | Matplotlib                          |
| Development Tools | Jupyter Notebook, Anaconda, PyCharm |

---

##  Machine Learning Models Used

| Model                | Purpose                                      |
|-----------------------|-----------------------------------------------|
| Naïve Bayes           | Baseline probabilistic classifier            |
| K-Nearest Neighbors   | Distance-based classification                |
| Logistic Regression   | Linear baseline classifier                   |
| Random Forest         | Ensemble learning, handles complex patterns  |
| XGBoost               | Gradient boosting for high accuracy          |
| Deep Neural Network   | Captures complex non-linear fraud patterns   |

**Class imbalance handling:** SMOTE (Synthetic Minority Over-sampling Technique) and hybrid under/oversampling.

---

##  Project Structure

```
credit-card-fraud-detection/
│
├── dataset/                  # Credit card transaction dataset (creditcard.csv)
├── models/                   # Trained ML/DL model files
├── static/                   # CSS, JS, images
├── templates/                # HTML templates (Django MVT)
├── app/                      # Django app (views, models, urls)
├── notebooks/                # Jupyter notebooks for model training & experiments
├── manage.py                 # Django project entry point
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

##  Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure MySQL database**
   - Create a database and update credentials in `settings.py`

5. **Run migrations**
   ```bash
   python manage.py migrate
   ```

6. **Start the development server**
   ```bash
   python manage.py runserver
   ```

7. **Open in browser**
   ```
   http://127.0.0.1:8000/
   ```

---

##  Usage

1. Launch the application and upload the credit card transaction dataset.
2. Generate train/test split.
3. Run the desired algorithm (Random Forest, XGBoost, etc.) to train the model.
4. Upload test data to detect fraudulent transactions.
5. View the fraud detection results and transaction distribution graph.

---

##  Results

| Model                | Accuracy   |
|------------------------|-----------|
| Naïve Bayes            | 97.92%    |
| K-Nearest Neighbors    | 97.69%    |
| Logistic Regression    | 54.86%    |
| Random Forest          | 99.78%    |

> K-Nearest Neighbors and Random Forest outperformed Naïve Bayes and Logistic Regression on the hybrid-sampled dataset.

---

##  Testing

The system was validated using:
- Unit Testing
- Integration Testing
- Functional Testing
- System Testing
- White Box & Black Box Testing
- User Acceptance Testing

All test cases passed successfully with no critical defects.

---

##  Future Enhancements

- Real-time transaction monitoring and fraud alerts
- Integration of advanced deep learning architectures (LSTM/Autoencoders)
- REST API for third-party integration
- Cloud deployment for scalability
- Enhanced explainability (SHAP/LIME) for fraud predictions

---

##  Contributors

- Mohammed Zakir Ahmed
- Chenchu Bharath
- Boga Mounika
- Pachimatla Sai Harshitha

**Guided by:** K. Pravalika, Assistant Professor
**Department of CSE (AI & ML), Sree Chaitanya Institute of Technological Sciences, Karimnagar**
*(Affiliated to JNTUH)*

---

##  References

1. Vesta Corporation, "Credit Card Fraud Detection Using Machine Learning: A Survey," IJCA, vol. 182, no. 10, 2018.
2. A. Dal Pozzolo et al., "Calibrating Probability with Undersampling for Unbalanced Classification," IEEE SSCI, 2015.
3. I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016.
4. L. Breiman, "Random Forests," Machine Learning, vol. 45, no. 1, 2001.
5. D. P. Kingma, J. Ba, "Adam: A Method for Stochastic Optimization," 2015.
6. N. V. Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique," JAIR, 2002.

---

## License

This project is developed for academic purposes as part of the B.Tech Major Project requirement at JNTUH.
