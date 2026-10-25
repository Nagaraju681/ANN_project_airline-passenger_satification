# ✈️ Airline Passenger Satisfaction Prediction using ANN

## 📌 Project Overview

This project predicts whether an airline passenger is **Satisfied** or **Dissatisfied** using an Artificial Neural Network (ANN).

The model uses passenger information, flight details, and service ratings to predict satisfaction. A Streamlit web application allows users to enter passenger details and view a prediction.

## 🎯 Objectives

- Predict airline passenger satisfaction using ANN.
- Prepare and preprocess passenger data.
- Apply deep learning for binary classification.
- Build an interactive web application using Streamlit.

## 🛠️ Technologies Used

- **Python** — Programming language
- **TensorFlow / Keras** — ANN model development
- **Pandas & NumPy** — Data processing
- **Scikit-learn** — Data preprocessing
- **Streamlit** — Web application
- **Joblib** — Loading the saved preprocessing object
- **Git & GitHub** — Version control and project hosting

## 🧠 Model Details

The project uses an Artificial Neural Network to predict passenger satisfaction.

The workflow includes:

1. Load and explore the dataset.
2. Clean the data and handle missing values.
3. Separate input features and target variable.
4. Preprocess numerical and categorical features.
5. Train the ANN model.
6. Evaluate model performance.
7. Save the trained model and preprocessor.
8. Use the saved files in the Streamlit application.

## 📂 Project Structure

```text
ANN_project/
│
├── app.py
├── airline_ann_model1.keras
├── airline_preprocessor1.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Nagaraju681/ANN_project_airline-passenger_satification.git
```

### 2. Open the project folder

```bash
cd ANN_project_airline-passenger_satification
```

### 3. Create a virtual environment (optional)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application using:

```bash
python -m streamlit run app.py
```

The application will open in your browser. If it does not open automatically, use the local URL displayed in your terminal.

## 📊 Prediction Output

The application predicts the passenger's satisfaction category:

- **Satisfied**
- **Dissatisfied**

The prediction depends on the passenger information and service ratings provided as input.

## 💡 Key Learnings

- Data preprocessing for machine learning.
- ANN architecture and binary classification.
- Saving and loading trained deep learning models.
- Using a preprocessing pipeline during prediction.
- Building an interactive ML application with Streamlit.
- Managing project files with Git and GitHub.

## 👨‍💻 Author

**Nagaraju**

GitHub: [Nagaraju681](https://github.com/Nagaraju681)

## 📌 Future Improvements

- Deploy the application online.
- Add model evaluation metrics and visualizations.
- Improve prediction performance through hyperparameter tuning.
- Enhance the user interface and user experience.

---

*This project was developed for learning and demonstrating an end-to-end deep learning workflow.*
