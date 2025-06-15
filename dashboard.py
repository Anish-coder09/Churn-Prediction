import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import base64
import io
import joblib
import plotly.graph_objs as go
from pyngrok import ngrok
import threading

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout
app.layout = dbc.Container([
    html.H1("Customer Churn Prediction Dashboard", className="mt-4 mb-4 text-center"),
    
    # Input section
    dbc.Row([
        dbc.Col([
            html.H4("📥 Upload Customer Data or Enter Details"),
            dcc.Upload(
                id='upload-data',
                children=html.Button('Upload CSV', className='btn btn-primary mb-3'),
                multiple=False
            ),
            html.Label("Select Features:"),
            dcc.Checklist(
                id='features-checklist',
                options=[
                    {'label': 'Senior Citizen', 'value': 'SeniorCitizen'},
                    {'label': 'Partner', 'value': 'Partner'},
                    {'label': 'Dependents', 'value': 'Dependents'},
                    {'label': 'Paperless Billing', 'value': 'PaperlessBilling'},
                    {'label': 'Phone Service', 'value': 'PhoneService'},
                    {'label': 'Multiple Lines', 'value': 'MultipleLines'},
                    {'label': 'Online Security', 'value': 'OnlineSecurity'},
                    {'label': 'Online Backup', 'value': 'OnlineBackup'},
                    {'label': 'Device Protection', 'value': 'DeviceProtection'},
                    {'label': 'Tech Support', 'value': 'TechSupport'},
                    {'label': 'Streaming TV', 'value': 'StreamingTV'},
                    {'label': 'Streaming Movies', 'value': 'StreamingMovies'},
                ],
                value=[],
                className="mb-3"
            ),
            html.Label("Gender:"),
            dcc.Dropdown(
                id='gender-dropdown',
                options=[
                    {'label': 'Male', 'value': 'Male'},
                    {'label': 'Female', 'value': 'Female'}
                ],
                value='Male',
                className="mb-3"
            ),
            html.Label("Internet Service:"),
            dcc.Dropdown(
                id='internet-dropdown',
                options=[
                    {'label': 'DSL', 'value': 'DSL'},
                    {'label': 'Fiber optic', 'value': 'Fiber optic'},
                    {'label': 'No', 'value': 'No'}
                ],
                value='DSL',
                className="mb-3"
            ),
            html.Label("Contract Type:"),
            dcc.Dropdown(
                id='contract-dropdown',
                options=[
                    {'label': 'Month-to-month', 'value': 'Month-to-month'},
                    {'label': 'One year', 'value': 'One year'},
                    {'label': 'Two year', 'value': 'Two year'}
                ],
                value='Month-to-month',
                className="mb-3"
            ),
            html.Label("Payment Method:"),
            dcc.Dropdown(
                id='payment-dropdown',
                options=[
                    {'label': 'Electronic check', 'value': 'Electronic check'},
                    {'label': 'Mailed check', 'value': 'Mailed check'},
                    {'label': 'Bank transfer (automatic)', 'value': 'Bank transfer (automatic)'},
                    {'label': 'Credit card (automatic)', 'value': 'Credit card (automatic)'}
                ],
                value='Electronic check',
                className="mb-3"
            ),
            html.Label("Tenure (Months):"),
            dcc.Input(id='tenure-input', type='number', value=12, className="form-control mb-3"),
            html.Label("Monthly Charges ($):"),
            dcc.Input(id='monthly-charges-input', type='number', value=50.0, className="form-control mb-3"),
            html.Button('Predict', id='predict-button', className='btn btn-success mb-3'),
        ], width=4),
        dbc.Col([
            html.H4("🔍 Prediction Results"),
            html.Div(id='prediction-output', className="mt-3"),
            dcc.Graph(id='hazard-graph', className="mt-3"),
            dcc.Graph(id='gauge-graph', className="mt-3"),
            dcc.Graph(id='survival-graph', className="mt-3"),
            dcc.Graph(id='churn-graph', className="mt-3"),
            dcc.Graph(id='churn-pie', className="mt-3"),
            dcc.Graph(id='tenure-contract-bar', className="mt-3"),
        ], width=8)
    ]),
    
    html.Hr(),
    html.H4("📊 Customer Data"),
    html.H5("All Customers", className="text-primary p-2"),
    html.Div(id='all-customers'),
    html.H5("📊 Contract Statistics", className="text-info p-2"),
    html.Div(id='contract-stats'),
    html.H5("🚨 High Risk Customers (A)", className="text-white bg-danger p-2"),
    html.Div(id='high-risk'),
    html.H5("⚠️ Medium Risk Customers (B)", className="text-dark bg-warning p-2"),
    html.Div(id='medium-risk'),
    html.H5("✅ Low Risk Customers (C)", className="text-white bg-success p-2"),
    html.Div(id='low-risk'),
    html.Hr(),
    html.H4("🔑 Feature Importance"),
    dcc.Graph(id='feature-importance', className="mt-3"),
], fluid=True)

# Data Parsing and Prediction Function
def parse_contents(contents, filename, features, gender, internet, contract, payment, tenure, monthly_charges):
    # If no file is uploaded, create a single-row DataFrame with manual inputs
    if contents is None:
        df = pd.DataFrame({
            'SeniorCitizen': [1 if 'SeniorCitizen' in features else 0],
            'Partner': [1 if 'Partner' in features else 0],
            'Dependents': [1 if 'Dependents' in features else 0],
            'PaperlessBilling': [1 if 'PaperlessBilling' in features else 0],
            'PhoneService': [1 if 'PhoneService' in features else 0],
            'MultipleLines': [1 if 'MultipleLines' in features else 0],
            'OnlineSecurity': [1 if 'OnlineSecurity' in features else 0],
            'OnlineBackup': [1 if 'OnlineBackup' in features else 0],
            'DeviceProtection': [1 if 'DeviceProtection' in features else 0],
            'TechSupport': [1 if 'TechSupport' in features else 0],
            'StreamingTV': [1 if 'StreamingTV' in features else 0],
            'StreamingMovies': [1 if 'StreamingMovies' in features else 0],
            'gender': [gender],
            'InternetService': [internet],
            'Contract': [contract],
            'PaymentMethod': [payment],
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [tenure * monthly_charges],  # Simplified calculation
        })
    else:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            # Ensure required columns exist
            required_columns = ['SeniorCitizen', 'Partner', 'Dependents', 'PaperlessBilling', 'PhoneService',
                               'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                               'StreamingTV', 'StreamingMovies', 'gender', 'InternetService', 'Contract',
                               'PaymentMethod', 'tenure', 'MonthlyCharges', 'TotalCharges']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0  # Default value if column is missing
        except Exception as e:
            return html.Div([f"Error reading file: {str(e)}"]), {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

    try:
        # Note: The model loading assumes 'churn_model.joblib' exists and is a scikit-survival model
        model = joblib.load("churn_model.joblib")
        df["TotalCharges"] = df["TotalCharges"].replace({",": ""}, regex=True)
        df[["tenure", "MonthlyCharges", "TotalCharges"]] = df[["tenure", "MonthlyCharges", "TotalCharges"]].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=["tenure", "MonthlyCharges", "TotalCharges"])

        # Preprocess categorical columns to convert 'Yes'/'No' to 1/0
        binary_columns = ['SeniorCitizen', 'Partner', 'Dependents', 'PaperlessBilling', 'PhoneService',
                          'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                          'StreamingTV', 'StreamingMovies']
        for col in binary_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0).astype(int)

        # Create feature vector (10 features as per training)
        feature_vector = np.zeros((len(df), 10))
        feature_vector[:, 0] = df['SeniorCitizen'].astype(int)
        feature_vector[:, 1] = df['Partner'].astype(int)
        feature_vector[:, 2] = df['Dependents'].astype(int)
        feature_vector[:, 3] = (df['gender'] == 'Male').astype(int)
        feature_vector[:, 4] = (df['InternetService'] == 'Fiber optic').astype(int)
        feature_vector[:, 5] = (df['Contract'] == 'Month-to-month').astype(int)
        feature_vector[:, 6] = (df['PaymentMethod'] == 'Electronic check').astype(int)
        feature_vector[:, 7] = df['tenure'].astype(float)
        feature_vector[:, 8] = df['OnlineSecurity'].astype(int)
        feature_vector[:, 9] = df['TechSupport'].astype(int)

        # Predict survival function
        survival_function = model.predict_survival_function(feature_vector)
        hazard_function = model.predict_cumulative_hazard_function(feature_vector)

        # Use the first row for visualization
        survival_func = survival_function[0]
        hazard_func = hazard_function[0]

        # Dummy churn probability (for gauge and bar chart)
        churn_prob = 0.47  # Static as per the original code
        expected_lifetime_value = 1480.0  # Static as per original code
        df["Churn Probability"] = churn_prob

        def classify_risk(prob):
            if prob >= 0.7:
                return 'High (A)'
            elif prob >= 0.3:
                return 'Medium (B)'
            else:
                return 'Low (C)'

        df["Churn Risk"] = df["Churn Probability"].apply(classify_risk)

        # Create tenure bins for new chart
        df['Tenure_Bin'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72, np.inf],
                                  labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72', '72+'])
        tenure_contract_stats = df.groupby(['Tenure_Bin', 'Contract'])['Churn Probability'].mean().unstack().fillna(0)

    except Exception as e:
        return html.Div([f"Model Prediction Error: {str(e)}"]), {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

    # Prediction Output
    prediction_output = f"Churn Probability is {churn_prob} and Expected Life Time Value is ${expected_lifetime_value}"

    # Hazard Graph
    hazard_fig = {
        "data": [{
            "x": survival_func.x,
            "y": hazard_func.y,
            "type": "line",
            "name": "Cumulative Hazard",
            "line": {"color": "red"}
        }],
        "layout": {
            "title": "Cumulative Hazard Over Time",
            "xaxis": {"title": "Tenure"},
            "yaxis": {"title": "Cumulative Hazard"}
        }
    }

    # Gauge Graph
    gauge_fig = {
        "data": [{
            "type": "indicator",
            "mode": "gauge+number",
            "value": churn_prob,
            "title": {"text": "Churn Probability"},
            "gauge": {
                "axis": {"range": [0, 1]},
                "bar": {"color": "black"},
                "steps": [
                    {"range": [0, 0.3], "color": "green"},
                    {"range": [0.3, 0.7], "color": "yellow"},
                    {"range": [0.7, 1], "color": "red"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": churn_prob
                }
            }
        }],
        "layout": {}
    }

    # Survival Graph
    survival_fig = {
        "data": [{
            "x": survival_func.x,
            "y": survival_func.y,
            "type": "line",
            "name": "Survival Probability",
            "line": {"color": "red"}
        }],
        "layout": {
            "title": "Survival Probability Over Time",
            "xaxis": {"title": "Tenure"},
            "yaxis": {"title": "Survival Probability"}
        }
    }

    # Bar Graph
    bar_fig = {
        "data": [{
            "x": df.index,
            "y": df["Churn Probability"],
            "type": "bar",
            "name": "Churn Probability",
            "marker": {"color": df["Churn Probability"], "colorscale": "Bluered"}
        }],
        "layout": {"title": "Predicted Churn Probability", "xaxis": {"title": "Customer Index"}, "yaxis": {"title": "Probability"}}
    }

    # Pie Chart
    risk_counts = df["Churn Risk"].value_counts()
    pie_fig = {
        "data": [{
            "values": risk_counts.values,
            "labels": risk_counts.index,
            "type": "pie",
            "hole": 0.5,
            "marker": {"colors": ["#dc3545", "#ffc107", "#28a745"]}
        }],
        "layout": {"title": "Churn Risk Distribution"}
    }

    # Stacked Bar Chart
    tenure_contract_bar = {
        "data": [
            {
                "x": tenure_contract_stats.index,
                "y": tenure_contract_stats[contract],
                "type": "bar",
                "name": contract,
                "marker": {"color": color}
            } for contract, color in zip(tenure_contract_stats.columns, ['#dc3545', '#ffc107', '#28a745'])
        ],
        "layout": {
            "title": "Churn Probability by Tenure and Contract Type",
            "xaxis": {"title": "Tenure (Months)"},
            "yaxis": {"title": "Average Churn Probability"},
            "barmode": "stack"
        }
    }

    # Feature Importance (dummy values)
    feature_importance = {
        "Contract_One year": 0.1,
        "SeniorCitizen": 0.15,
        "InternetService_No": 0.2,
        "TotalCharges": 0.25,
        "Contract_Two year": 0.3,
        "tenure": 0.35,
        "InternetService_Fiber optic": 0.4,
        "PaymentMethod_Electronic check": 0.47
    }
    importance_fig = {
        "data": [{
            "x": list(feature_importance.values()),
            "y": list(feature_importance.keys()),
            "type": "bar",
            "orientation": "h",
            "marker": {"color": "blue"}
        }],
        "layout": {
            "title": "Feature Importance",
            "xaxis": {"title": "Importance"},
            "yaxis": {"title": "Feature"}
        }
    }

    # Tables
    def generate_table(dataframe):
        return dash_table.DataTable(
            data=dataframe.to_dict('records'),
            columns=[{"name": i, "id": i} for i in dataframe.columns],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}
        )

    all_table = generate_table(df)
    high_table = generate_table(df[df["Churn Risk"] == "High (A)"])
    medium_table = generate_table(df[df["Churn Risk"] == "Medium (B)"])
    low_table = generate_table(df[df["Churn Risk"] == "Low (C)"])

    # Additional Table: Churn Statistics by Contract Type
    contract_stats = df.groupby('Contract').agg({
        'Churn Probability': 'mean',
        'tenure': 'mean',
        'MonthlyCharges': 'mean'
    }).reset_index()
    contract_stats_table = generate_table(contract_stats)

    return (prediction_output, hazard_fig, gauge_fig, survival_fig, bar_fig, pie_fig,
            all_table, contract_stats_table, high_table, medium_table, low_table, importance_fig, tenure_contract_bar)

# Dash Callbacks
@app.callback(
    [Output('prediction-output', 'children'),
     Output('hazard-graph', 'figure'),
     Output('gauge-graph', 'figure'),
     Output('survival-graph', 'figure'),
     Output('churn-graph', 'figure'),
     Output('churn-pie', 'figure'),
     Output('all-customers', 'children'),
     Output('contract-stats', 'children'),
     Output('high-risk', 'children'),
     Output('medium-risk', 'children'),
     Output('low-risk', 'children'),
     Output('feature-importance', 'figure'),
     Output('tenure-contract-bar', 'figure')],
    [Input('predict-button', 'n_clicks'),
     Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
     State('features-checklist', 'value'),
     State('gender-dropdown', 'value'),
     State('internet-dropdown', 'value'),
     State('contract-dropdown', 'value'),
     State('payment-dropdown', 'value'),
     State('tenure-input', 'value'),
     State('monthly-charges-input', 'value')]
)
def update_output(n_clicks, contents, filename, features, gender, internet, contract, payment, tenure, monthly_charges):
    if n_clicks is None and contents is None:
        return (dash.no_update,) * 13

    result = parse_contents(contents, filename, features, gender, internet, contract, payment, tenure, monthly_charges)
    if isinstance(result, tuple):
        return (result[0], result[1], result[2], result[3], result[4], result[5],
                result[6], result[7], result[8], result[9], result[10],
                result[11], result[12])
    else:
        return (result,) + (dash.no_update,) * 12

# Start the Dash app using ngrok
def run_dash():
    app.run_server(host='0.0.0.0', port=8050)

if __name__ == '__main__':
    ngrok.kill()
    # Replace with your actual ngrok auth token
    ngrok.set_auth_token("2yUEW9viYUv3E3okQYHoP9N7pLi_3JduMpjNdk7FBYyUw69L1")
    public_url = ngrok.connect(8050)
    print(f"Open your app here: {public_url}")
    thread = threading.Thread(target=run_dash)
    thread.start()