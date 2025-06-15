# Churn-Prediction
README.md
markdown

Collapse

Wrap

Copy
# Customer Churn Prediction Dashboard

A Dash-based web app for predicting customer churn using a survival analysis model. Users can upload a CSV or manually input customer features to visualize churn probability, survival/hazard functions, and feature importance.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Sample Visualization](#sample-visualization)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project provides an interactive dashboard for telecom customer churn prediction, built with Python's Dash and Bootstrap. It uses a pre-trained `scikit-survival` model to predict churn probabilities and visualize results. Users can input data via CSV or manual forms, with ngrok enabling public access during development.

---

## Features

- **Data Input**: Upload CSV or manually select features (e.g., tenure, contract, services).
- **Visualizations**:
  - Cumulative Hazard and Survival Probability graphs.
  - Gauge chart for churn probability.
  - Bar chart for predicted churn probabilities.
  - Pie chart for churn risk (High, Medium, Low).
  - Stacked bar chart for churn by tenure and contract.
  - Feature importance chart.
- **Tables**: All customers, risk segments, and contract stats.
- **Responsive UI**: Built with Dash and Bootstrap.

---

## Sample Visualization

Below is a sample pie chart showing the distribution of churn risk (High, Medium, Low):

![image](https://github.com/user-attachments/assets/e2a40c22-0ab4-4d49-baf2-b44d3faa0786)


*Note*: This is a placeholder image. Replace with an actual screenshot from your dashboard.

---

## Dataset

Expected CSV columns:

| Column            | Description                     | Type/Values                              |
|-------------------|---------------------------------|------------------------------------------|
| `customerID`      | Unique ID                      | String                                   |
| `gender`          | Gender                         | Male/Female                              |
| `SeniorCitizen`   | Senior citizen                 | 0/1                                      |
| `Partner`         | Has partner                    | Yes/No or 0/1                            |
| `Dependents`      | Has dependents                 | Yes/No or 0/1                            |
| `tenure`          | Months with company            | Integer                                  |
| `PhoneService`    | Phone service                  | Yes/No or 0/1                            |
| `InternetService` | Internet type                  | DSL/Fiber optic/No                       |
| `Contract`        | Contract type                  | Month-to-month/One year/Two year         |
| `PaymentMethod`   | Payment method                 | Electronic check/Mailed check/etc.        |
| `MonthlyCharges`  | Monthly bill                   | Float                                    |
| `TotalCharges`    | Total charges                  | Float                                    |

---

## Installation

1. **Clone Repository**:
   ```bash
   git clone https://github.com/your-username/customer-churn-prediction.git
   cd customer-churn-prediction
Set Up Virtual Environment:
bash

Collapse

Wrap

Run

Copy
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
Install Dependencies:
bash

Collapse

Wrap

Run

Copy
pip install -r requirements.txt
Configure ngrok:
Get an auth token from ngrok.com.
Update churn_dashboard.py:
python

Collapse

Wrap

Run

Copy
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
Add Model:
Place churn_model.joblib in the project root.
Usage
Run the Dashboard:
bash

Collapse

Wrap

Run

Copy
python churn_dashboard.py
Access via the ngrok URL (e.g., https://9ea0-35-231-45-36.ngrok-free.app/).
Interact:
Upload a CSV or enter features (e.g., tenure, contract).
Click "Predict" to view visualizations and tables.
Dependencies
requirements.txt:

text

Collapse

Wrap

Copy
dash==3.0.4
dash-bootstrap-components==2.0.3
pyngrok==7.2.11
pandas==2.2.2
numpy==2.0.2
joblib==1.5.1
plotly==5.24.1
scikit-survival==0.24.1
Install:

bash

Collapse

Wrap

Run

Copy
pip install dash dash-bootstrap-components pyngrok pandas numpy joblib plotly scikit-survival
Project Structure
text

Collapse

Wrap

Copy
customer-churn-prediction/
├── churn_dashboard.py       # Dash app script
├── churn_model.joblib       # Pre-trained model
├── requirements.txt         # Dependencies
├── README.md                # Documentation
└── .gitignore               # Git ignore
Model Details
Model: scikit-survival model (e.g., CoxPHSurvivalAnalysis) in churn_model.joblib.
Features (10):
Binary: SeniorCitizen, Partner, Dependents, OnlineSecurity, TechSupport.
Encoded: gender (Male=1), InternetService (Fiber optic=1), Contract (Month-to-month=1), PaymentMethod (Electronic check=1).
Continuous: tenure.
Outputs: Survival/hazard functions, churn probability (hardcoded 0.47), lifetime value ($1480.0).
Risk Levels: High (≥0.7), Medium (0.3–0.7), Low (<0.3).
Note: Replace hardcoded values with actual predictions for production.

Contributing
Fork the repository.
Create a branch: git checkout -b feature/your-feature.
Commit: git commit -m "Add feature".
Push: git push origin feature/your-feature.
Open a pull request.
License
MIT License

text

Collapse

Wrap

Copy
---

### Implementation Instructions

1. **Create `README.md`**:
   - Copy the Markdown content into `README.md` in your project root.
   - Save using a text editor or IDE.

2. **Customize**:
   - Replace `https://github.com/your-username/customer-churn-prediction` with your GitHub repository URL.
   - Update the ngrok token placeholder with a security note.
   - Replace the placeholder image link (`screenshots/churn_pie_chart.png`) with an actual screenshot:
     - Run the dashboard, take a screenshot of the pie chart (e.g., from the `churn-pie` graph).
     
     - Update the link: `(https://github.com/user-attachments/assets/e2a40c22-0ab4-4d49-baf2-b44d3faa0786)`.
   - Add a `LICENSE` file and link it in the [license](#license) section.

3. **Create `requirements.txt`**:
   ```bash
   echo -e "dash==3.0.4\ndash-bootstrap-components==2.0.3\npyngrok==7.2.11\npandas==2.2.2\nnumpy==2.0.2\njoblib==1.5.1\nplotly==5.24.1\nscikit-survival==0.24.1" > requirements.txt
Set Up .gitignore:
text

Collapse

Wrap

Copy
venv/
__pycache__/
*.pyc
churn_model.joblib
*.csv
.env
Push to GitHub:
bash

Collapse

Wrap

Run

Copy
git init
git add README.md churn_dashboard.py requirements.txt .gitignore
git commit -m "Add dashboard, README, and dependencies"
git remote add origin https://github.com/your-username/customer-churn-prediction.git
git push -u origin main
Sample Visualization Note
The pie chart is included as a sample because it’s a key visualization from the dashboard (showing churn risk distribution) and visually appealing for a README.
To create the actual image:
Run the dashboard (python churn_dashboard.py).
Input sample data or upload a CSV.
Take a screenshot of the pie chart (ID: churn-pie).
Save as screenshots/churn_pie_chart.png in your repository.
Push the image to GitHub: git add screenshots/churn_pie_chart.png.
