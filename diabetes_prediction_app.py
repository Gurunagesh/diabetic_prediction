import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime
import uuid
import os
import psycopg2_binary as psycopg2




general_health_goals_markdown_content = """
###### General Health Goals:
*   **Maintain a Balanced Diet:** Focus on whole grains, lean proteins, fruits, and vegetables. Limit processed foods, sugary drinks, and unhealthy fats.
*   **Regular Physical Activity:** Aim for at least 150 minutes of moderate-intensity aerobic activity per week, along with muscle-strengthening activities on 2 or more days.
*   **Achieve/Maintain Healthy Weight:** Even a modest weight loss can significantly reduce diabetes risk.
*   **Monitor Blood Sugar Levels:** If advised by your doctor, regularly check your glucose levels.
*   **Manage Stress:** Practice relaxation techniques like meditation or yoga.
*   **Regular Check-ups:** Consult your healthcare provider for regular screenings and personalized advice.
"""


# --- Configuration and Model Loading ---
st.set_page_config(page_title="Empowering Your Health: Diabetes Risk Predictor", layout="wide")

# Load artifacts
try:
    with open('robust_scaler.pkl', 'rb') as file:
        loaded_robust_scaler = pickle.load(file)
    with open('standard_scaler.pkl', 'rb') as file:
        loaded_standard_scaler = pickle.load(file)
    with open('gbc_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    with open('imputation_medians.pkl', 'rb') as file:
        imputation_medians = pickle.load(file)
    with open('final_feature_columns.pkl', 'rb') as file:
        final_feature_columns = pickle.load(file)

    
    # Get the directory of the current script
    #base_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the relative path to diabetes.csv
    #csv_path = os.path.join(base_dir, "diabetes.csv")

    # Load the CSV file
    original_df = pd.read_csv("diabetes.csv")

except FileNotFoundError as e:
    st.error(f"Error loading model artifacts: {e}. Make sure all .pkl files and diabetes.csv are in the correct directory.")
    st.stop() # Stop the app if essential files are missing

# Define Insulin capping upper bound explicitly for consistency
INSULIN_UPPER_BOUND = 270.0 # From previous analysis in the notebook

# --- Database Connection Function ---
DB_FILE = 'app_data.db'

def get_db_connection():
    db_url = os.getenv('DATABASE_URL')

    if db_url:
        # Assuming PostgreSQL for external database
        # import psycopg2 # Make sure psycopg2 is imported if this path is taken
        try:
            conn = psycopg2.connect(db_url)
            # st.success("Connected to PostgreSQL database!") # For debugging
            return conn
        except Exception as e:
            st.error(f"Error connecting to external database: {e}")
            st.stop()
    else:
        # Default to SQLite
        try:
            conn = sqlite3.connect(DB_FILE)
            # st.info("Connected to local SQLite database.") # For debugging
            return conn
        except sqlite3.Error as e:
            st.error(f"Error connecting to local SQLite database: {e}")
            st.stop()

# --- Feature Engineering Functions ---
def get_new_bmi_category(bmi_value):
    if bmi_value < 18.5:
        return "Underweight"
    elif 18.5 <= bmi_value <= 24.9:
        return "Normal"
    elif 24.9 < bmi_value <= 29.9:
        return "Overweight"
    elif 29.9 < bmi_value <= 34.9:
        return "Obesity 1"
    elif 34.9 < bmi_value <= 39.9:
        return "Obesity 2"
    else:
        return "Obesity 3"

def get_new_insulin_score(insulin_value):
    if 16 <= insulin_value <= 166:
        return "Normal"
    else:
        return "Abnormal"

def get_new_glucose_category(glucose_value):
    if glucose_value <= 70:
        return "Low"
    elif 70 < glucose_value <= 99:
        return "Normal"
    elif 99 < glucose_value <= 126:
        return "Overweight"
    else:
        return "Secret"

# --- Preprocessing Function (consolidated logic) ---
def preprocess_input(user_input_data_dict, imputation_medians, final_feature_columns, loaded_robust_scaler, loaded_standard_scaler, insulin_upper_bound):
    input_df = pd.DataFrame([user_input_data_dict])

    # 1. Replace 0s with imputed medians
    # Features where 0s were replaced by medians in the notebook
    features_to_impute_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in features_to_impute_zeros:
        if input_df[col].iloc[0] == 0:
            input_df[col] = imputation_medians[col] # Using loaded medians

    # 2. Outlier capping for Insulin
    if input_df['Insulin'].iloc[0] > insulin_upper_bound:
        input_df['Insulin'] = insulin_upper_bound

    # 3. Feature Engineering
    input_df['NewBMI'] = input_df['BMI'].apply(get_new_bmi_category)
    input_df['NewInsulinScore'] = input_df['Insulin'].apply(get_new_insulin_score)
    input_df['NewGlucose'] = input_df['Glucose'].apply(get_new_glucose_category)

    # 4. One-hot encoding for new categorical features
    # Initialize all possible OHE columns to False
    for col in final_feature_columns:
        if col.startswith('NewBMI_') or col.startswith('NewInsulinScore_') or col.startswith('NewGlucose_'):
            input_df[col] = False

    # Set True based on derived categories, ensuring column exists in final_feature_columns
    if input_df['NewBMI'].iloc[0] == 'Obesity 1' and 'NewBMI_Obesity 1' in final_feature_columns: input_df['NewBMI_Obesity 1'] = True
    elif input_df['NewBMI'].iloc[0] == 'Obesity 2' and 'NewBMI_Obesity 2' in final_feature_columns: input_df['NewBMI_Obesity 2'] = True
    elif input_df['NewBMI'].iloc[0] == 'Obesity 3' and 'NewBMI_Obesity 3' in final_feature_columns: input_df['NewBMI_Obesity 3'] = True
    elif input_df['NewBMI'].iloc[0] == 'Overweight' and 'NewBMI_Overweight' in final_feature_columns: input_df['NewBMI_Overweight'] = True
    elif input_df['NewBMI'].iloc[0] == 'Underweight' and 'NewBMI_Underweight' in final_feature_columns: input_df['NewBMI_Underweight'] = True

    if input_df['NewInsulinScore'].iloc[0] == 'Normal' and 'NewInsulinScore_Normal' in final_feature_columns: input_df['NewInsulinScore_Normal'] = True

    if input_df['NewGlucose'].iloc[0] == 'Low' and 'NewGlucose_Low' in final_feature_columns: input_df['NewGlucose_Low'] = True
    elif input_df['NewGlucose'].iloc[0] == 'Normal' and 'NewGlucose_Normal' in final_feature_columns: input_df['NewGlucose_Normal'] = True
    elif input_df['NewGlucose'].iloc[0] == 'Overweight' and 'NewGlucose_Overweight' in final_feature_columns: input_df['NewGlucose_Overweight'] = True
    elif input_df['NewGlucose'].iloc[0] == 'Secret' and 'NewGlucose_Secret' in final_feature_columns: input_df['NewGlucose_Secret'] = True

    # Drop the temporary categorical columns that were used to create OHE features
    input_df = input_df.drop(columns=['NewBMI', 'NewInsulinScore', 'NewGlucose'])

    # Reorder input_df columns to match final_feature_columns (essential step)
    input_df = input_df[final_feature_columns]

    # Separate numerical and categorical parts for scaling
    numerical_cols_for_robust_scaler = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    # Filter final_feature_columns to get only categorical OHE columns
    categorical_ohe_cols = [col for col in final_feature_columns if col not in numerical_cols_for_robust_scaler]

    input_numerical = input_df[numerical_cols_for_robust_scaler]
    input_categorical_ohe = input_df[categorical_ohe_cols]

    # Apply RobustScaler to numerical features
    input_numerical_scaled_robust = loaded_robust_scaler.transform(input_numerical)
    input_numerical_scaled_robust_df = pd.DataFrame(input_numerical_scaled_robust, columns=numerical_cols_for_robust_scaler, index=input_df.index)

    # Concatenate scaled numerical with one-hot encoded categorical features
    processed_input = pd.concat([input_numerical_scaled_robust_df, input_categorical_ohe], axis=1)

    # Apply StandardScaler to the entire processed input
    final_input_scaled = loaded_standard_scaler.transform(processed_input)

    return final_input_scaled

# --- Health Score Calculation Function ---
def calculate_health_score(user_input_data, imputation_medians, insulin_upper_bound):
    # Create a DataFrame from the user input for consistent processing
    score_df = pd.DataFrame([user_input_data])

    # Apply similar 0-value imputation as in preprocess_input for consistency
    features_to_impute_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in features_to_impute_zeros:
        if score_df[col].iloc[0] == 0:
            score_df[col] = imputation_medians[col]

    # Apply Insulin capping
    if score_df['Insulin'].iloc[0] > insulin_upper_bound:
        score_df['Insulin'] = insulin_upper_bound

    # Initialize risk score
    risk_score = 0

    # Assign scores based on feature values (example logic, can be refined)
    # Glucose
    if score_df['Glucose'].iloc[0] >= 126: # Diabetic range
        risk_score += 30
    elif score_df['Glucose'].iloc[0] >= 100: # Pre-diabetic range
        risk_score += 15

    # BMI
    if score_df['BMI'].iloc[0] >= 30: # Obese
        risk_score += 25
    elif score_df['BMI'].iloc[0] >= 25: # Overweight
        risk_score += 10

    # BloodPressure
    if score_df['BloodPressure'].iloc[0] >= 140: # Hypertensive
        risk_score += 20
    elif score_df['BloodPressure'].iloc[0] >= 120: # Elevated
        risk_score += 10

    # Age
    if score_df['Age'].iloc[0] >= 45:
        risk_score += 10
    if score_df['Age'].iloc[0] >= 60:
        risk_score += 10 # Additional risk for older age

    # Pregnancies
    if score_df['Pregnancies'].iloc[0] > 0:
        risk_score += score_df['Pregnancies'].iloc[0] * 1 # Example: 1 point per pregnancy

    # DiabetesPedigreeFunction
    risk_score += score_df['DiabetesPedigreeFunction'].iloc[0] * 20 # Scale DPF contribution

    # Insulin (abnormal levels)
    insulin_val = score_df['Insulin'].iloc[0]
    if insulin_val < 16 or insulin_val > 166: # Abnormal range
        risk_score += 15

    # Normalize the score to 0-100 range
    # Max possible score (approx): 30 (Glucose) + 25 (BMI) + 20 (BP) + 20 (Age) + 17 (Preg) + 2.5*20 (DPF) + 15 (Insulin) = 177
    # Let's set a conceptual max score a bit higher or adjust scaling factor
    max_conceptual_score = 200 # Adjust based on desired sensitivity
    normalized_score = (risk_score / max_conceptual_score) * 100

    # Ensure score stays within 0-100
    normalized_score = max(0, min(100, normalized_score))

    return round(normalized_score, 2)

# --- Usage Logging Function ---
def log_usage_session(session_id):
    conn = None
    try:
        conn = get_db_connection()
        if conn is None: # Handle case where connection failed
            st.error("Failed to connect to the database for usage logging.")
            return
        cursor = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute("INSERT INTO usage_logs (timestamp, session_id) VALUES (?, ?)",
                       (timestamp, session_id))
        conn.commit()
        # st.success(f"Session {session_id} logged.") # For debugging, can be removed
    except Exception as e:
        st.error(f"Error logging session: {e}")
    finally:
        if conn:
            conn.close()

# --- Admin/Metrics Functions ---
def fetch_feedback_logs():
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            st.error("Failed to connect to the database for fetching feedback.")
            return pd.DataFrame()
        df = pd.read_sql_query("SELECT * FROM feedback_logs ORDER BY timestamp DESC", conn)
        return df
    except Exception as e:
        st.error(f"Error fetching feedback logs: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def fetch_usage_logs():
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            st.error("Failed to connect to the database for fetching usage logs.")
            return pd.DataFrame()
        df = pd.read_sql_query("SELECT * FROM usage_logs ORDER BY timestamp DESC", conn)
        return df
    except Exception as e:
        st.error(f"Error fetching usage logs: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()


# --- Interpretation Function ---
def generate_interpretation(user_data, prediction_label, prediction_proba, threshold):
    interpretations = []

    # General advice
    if prediction_label == 'Diabetic':
        interpretations.append(f"Based on a probability of {prediction_proba[0][1]:.2f} (above threshold of {threshold:.2f}), the model predicts a **higher risk of diabetes** for this profile. It's crucial to consult a healthcare professional for diagnosis and management.")
    else:
        interpretations.append(f"Based on a probability of {prediction_proba[0][1]:.2f} (below threshold of {threshold:.2f}), the model predicts a **lower risk of diabetes** for this profile. Maintaining a healthy lifestyle is key for prevention.")

    interpretations.append("\n--- Key Factors --- ")

    # Glucose Interpretation
    glucose_val = user_data['Glucose']
    if glucose_val >= 126: # Diabetic range
        interpretations.append(f"- **Glucose ({glucose_val} mg/dL)**: This level is in the **diabetic range**. High blood glucose is a primary indicator of diabetes. Immediate medical evaluation is highly recommended.")
    elif glucose_val >= 100: # Pre-diabetic range
        interpretations.append(f"- **Glucose ({glucose_val} mg/dL)**: This level is in the **pre-diabetic range**. Lifestyle changes, such as diet and exercise, can help prevent progression to type 2 diabetes. Consult your doctor.")
    elif glucose_val < 70 and glucose_val != 0:
        interpretations.append(f"- **Glucose ({glucose_val} mg/dL)**: This level is lower than typical normal ranges. While often not directly indicative of diabetes, persistent low glucose should be discussed with a doctor.")
    else:
        interpretations.append(f"- **Glucose ({glucose_val} mg/dL)**: This level is within the generally **normal range**. Healthy eating and regular physical activity help maintain healthy blood sugar.")

    # BMI Interpretation
    bmi_val = user_data['BMI']
    if bmi_val >= 30: # Obese
        interpretations.append(f"- **BMI ({bmi_val:.1f})**: This indicates **obesity**. Obesity is a significant risk factor for type 2 diabetes. Weight management through diet and exercise is strongly advised.")
    elif bmi_val >= 25: # Overweight
        interpretations.append(f"- **BMI ({bmi_val:.1f})**: This indicates **overweight**. Being overweight increases diabetes risk. Aim for a healthy weight through balanced nutrition and physical activity.")
    elif bmi_val < 18.5 and bmi_val != 0: # Underweight
        interpretations.append(f"- **BMI ({bmi_val:.1f})**: This indicates **underweight**. While less common for diabetes risk, maintaining a healthy weight is important for overall health.")
    else:
        interpretations.append(f"- **BMI ({bmi_val:.1f})**: This is in the **healthy weight range**. Continue with a balanced diet and regular exercise.")

    # BloodPressure Interpretation
    bp_val = user_data['BloodPressure']
    if bp_val >= 140: # Hypertensive
        interpretations.append(f"- **Blood Pressure ({bp_val} mm Hg)**: This is in the **hypertensive range**. High blood pressure often co-occurs with diabetes and is a risk factor for cardiovascular disease. Medical consultation is recommended.")
    elif bp_val >= 120: # Elevated/Pre-hypertension
        interpretations.append(f"- **Blood Pressure ({bp_val} mm Hg)**: This is in the **elevated/pre-hypertension range**. Monitoring and lifestyle adjustments are important to prevent hypertension.")
    else:
        interpretations.append(f"- **Blood Pressure ({bp_val} mm Hg)**: This is within the **normal range**. Maintain a heart-healthy lifestyle.")

    # Age Interpretation
    age_val = user_data['Age']
    if age_val >= 45:
        interpretations.append(f"- **Age ({age_val} years)**: Diabetes risk generally increases with age, especially after 45. Regular screenings are advisable.")
    else:
        interpretations.append(f"- **Age ({age_val} years)**: While younger, it's still important to be aware of diabetes risk factors and maintain a healthy lifestyle.")

    # Pregnancies Interpretation
    pregnancies_val = user_data['Pregnancies']
    if pregnancies_val > 0:
        interpretations.append(f"- **Pregnancies ({pregnancies_val})**: A history of pregnancies, particularly multiple pregnancies or gestational diabetes, can be associated with an increased risk of developing type 2 diabetes later in life.")

    # DiabetesPedigreeFunction Interpretation
    dpf_val = user_data['DiabetesPedigreeFunction']
    if dpf_val > 0.5:
        interpretations.append(f"- **Diabetes Pedigree Function ({dpf_val:.3f})**: This score reflects the genetic predisposition to diabetes based on family history. A higher value indicates a stronger family history of diabetes, increasing your personal risk.")
    else:
        interpretations.append(f"- **Diabetes Pedigree Function ({dpf_val:.3f})**: This score indicates a relatively lower genetic predisposition to diabetes from family history, but other risk factors remain important.")

    # Insulin Interpretation
    insulin_val = user_data['Insulin']
    if insulin_val < 16 or insulin_val > 166: # Based on the 'Normal' range defined in NewInsulinScore
        interpretations.append(f"- **Insulin ({insulin_val:.1f} mu U/ml)**: This value falls outside the typical healthy range for fasting insulin. Abnormal insulin levels can indicate insulin resistance or other pancreatic issues. Further medical investigation is advised.")
    else:
        interpretations.append(f"- **Insulin ({insulin_val:.1f} mu U/ml)**: This value is within the typical healthy range for fasting insulin. Consistent healthy habits support pancreatic function.")

    interpretations.append("\n***Disclaimer:*** *These interpretations are based on statistical patterns from the training data and are not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any health concerns or before making any decisions related to your health or treatment.*")

    return "\n".join(interpretations)

# --- Session Management ---
# Ensure session_id is generated and logged only once per session
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

if 'session_logged' not in st.session_state:
    log_usage_session(st.session_state['session_id'])
    st.session_state['session_logged'] = True

# --- Streamlit UI ---
st.title("Empowering Your Health: Diabetes Risk Predictor")
st.markdown("--- ")

# Add a GIF for visual appeal at the top
st.image("8260cb45-a911-40fb-92bc-458b36125e36.webp", caption='Welcome to Diabetes Prediction!', width=300)

# --- About the App Section ---
with st.expander("✨ About this Diabetes Prediction App ✨"):
    st.markdown("""
    Welcome to your personal Diabetes Risk Predictor! This interactive application is designed to help you understand and manage your potential diabetes risk based on key health metrics. \n\nBuilt with advanced machine learning, our app goes beyond a simple prediction, offering personalized insights, educational resources, and actionable steps to empower your health journey. Here's what you can explore:
    
    *   **Interactive Patient Data Input**: Easily enter your health parameters like Glucose, BMI, Blood Pressure, and more.
    *   **Dynamic Prediction Threshold**: Adjust the model's sensitivity to see how changes impact your risk classification.
    *   **Instant Prediction & Interpretation**: Get an immediate prediction (Diabetic/Non-Diabetic) with a detailed explanation of what each factor means for you.
    *   **Personalized Health Score**: Receive an intuitive 0-100 risk score, giving you a quick snapshot of your overall health status related to diabetes.
    *   **Interactive Exploratory Data Analysis (EDA)**: Dive into the underlying dataset with dynamic charts like correlation heatmaps, feature distributions, and comparative plots to understand the data that trained our model.
    *   **Patient Cohort Comparison**: See how your individual health metrics stack up against the average values of 'Diabetic' and 'Non-Diabetic' patient groups.
    *   **Educational Insights / Dynamic Glossary**: Unravel complex medical terms and understand their significance with our built-in glossary.
    *   **Actionable Recommendations / Goal Setter**: If you're at risk, discover personalized, data-driven suggestions for lifestyle changes that could reduce your diabetes probability.
    *   **Feedback Submission**: Share your thoughts and help us continuously improve the app!
    *   **Usage Tracking & Admin Dashboard**: (For administrators) Monitor application usage and review valuable user feedback.

    We believe that informed decisions lead to better health. Start exploring to take control of your well-being today!\n
    ***Disclaimer:*** *This app is for informational purposes only and not a substitute for professional medical advice. Consult a healthcare provider for any health concerns.*
    """)

st.header("Enter Patient Data:")

# Create input widgets for each feature
col1, col2, col3 = st.columns(3)

with col1:
    pregnancies = st.number_input(
        "Number of Pregnancies",
        min_value=0, max_value=17, value=3, step=1, key='main_pregnancies'
    )
    glucose = st.number_input(
        "Glucose Level (mg/dL)",
        min_value=0, max_value=200, value=120, step=1, key='main_glucose'
    )
    blood_pressure = st.number_input(
        "Blood Pressure (mm Hg)",
        min_value=0, max_value=122, value=72, step=1, key='main_blood_pressure'
    )

with col2:
    skin_thickness = st.number_input(
        "Skin Thickness (mm)",
        min_value=0, max_value=99, value=23, step=1, key='main_skin_thickness'
    )
    insulin = st.number_input(
        "Insulin Level (mu U/ml)",
        min_value=0, max_value=270, value=100, step=1, key='main_insulin'
    )
    bmi = st.number_input(
        "BMI (Body Mass Index)",
        min_value=0.0, max_value=70.0, value=32.0, step=0.1, key='main_bmi'
    )

with col3:
    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.07, max_value=2.5, value=0.372, step=0.001, format="%.3f", key='main_dpf'
    )
    age = st.number_input(
        "Age (years)",
        min_value=21, max_value=81, value=29, step=1, key='main_age'
    )

st.markdown("--- ")

# Dynamic Prediction Threshold Slider
prediction_threshold = st.slider(
    "Set Prediction Probability Threshold (for 'Diabetic' classification)",
    min_value=0.01,
    max_value=0.99,
    value=0.5, # Default threshold
    step=0.01,
    help="Adjust this threshold to change the sensitivity of the 'Diabetic' classification. A higher threshold makes the model more conservative (fewer positive predictions), a lower threshold makes it more sensitive (more positive predictions)."
)

predict_button = st.button("Predict Diabetes Risk")

# --- Prediction Logic ---
if predict_button:
    # Collect user inputs into a dictionary
    user_input_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    final_input_scaled = preprocess_input(user_input_data, imputation_medians, final_feature_columns, loaded_robust_scaler, loaded_standard_scaler, INSULIN_UPPER_BOUND)

    # Make prediction
    prediction_proba = loaded_model.predict_proba(final_input_scaled)
    # Classify based on the dynamic threshold
    if prediction_proba[0][1] >= prediction_threshold:
        prediction_label = 'Diabetic'
        display_message = st.error
        prob_display = f"Probability of Diabetes: **{prediction_proba[0][1]:.2f}**"
    else:
        prediction_label = 'Non-Diabetic'
        display_message = st.success
        prob_display = f"Probability of Diabetes: **{prediction_proba[0][1]:.2f}** (below threshold)"


    st.markdown("### Prediction Result:")
    display_message(f"The patient is predicted to be **{prediction_label}**.")
    st.write(prob_display)

    st.warning("This prediction is for informational purposes only and does not constitute medical advice.")

    # Display interpretation
    st.markdown("### Personalized Interpretation and Advice:")
    st.info(generate_interpretation(user_input_data, prediction_label, prediction_proba, prediction_threshold))

    # Store results in session state for access in other sections (e.g., Actionable Recommendations)
    st.session_state.prediction_label = prediction_label
    st.session_state.prediction_proba = prediction_proba
    st.session_state.user_input_data = user_input_data

    # Calculate and display Health Score
    health_score = calculate_health_score(user_input_data, imputation_medians, INSULIN_UPPER_BOUND)
    st.markdown(f"### Your Personalized Health Score: **{health_score:.2f}** / 100")
    if health_score >= 60:
        st.warning("A high health score indicates a higher overall risk based on your input features.")
    else:
        st.info("A lower health score suggests a relatively lower overall risk based on your input features.")

st.markdown("--- ")

# --- Interactive EDA Section ---
with st.expander("Interactive Exploratory Data Analysis (EDA)"):
    st.header("Explore the Dataset")

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr_matrix = original_df.corr(numeric_only=True) # Use numeric_only=True for robustness
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                         color_continuous_scale=px.colors.sequential.Viridis)
    fig_corr.update_layout(height=700, width=800)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Feature Distributions
    st.subheader("Feature Distributions")
    # Filter original_df.columns to include only numeric for distribution plots
    numeric_cols = original_df.select_dtypes(include=np.number).columns.drop('Outcome', errors='ignore')
    eda_feature_dist = st.selectbox(
        "Select a feature to view its distribution:",
        options=numeric_cols
    )
    hist_type = st.radio("Choose plot type:", ('Histogram', 'Density Plot'), horizontal=True)
    segment_by_outcome = st.checkbox("Segment by Outcome?")

    if hist_type == 'Histogram':
        if segment_by_outcome:
            fig_hist = px.histogram(original_df, x=eda_feature_dist, color='Outcome', marginal="box",
                                    title=f"Distribution of {eda_feature_dist} by Outcome")
        else:
            fig_hist = px.histogram(original_df, x=eda_feature_dist, marginal="box",
                                    title=f"Distribution of {eda_feature_dist}")
    else: # Density Plot
        if segment_by_outcome:
             fig_hist = px.histogram(original_df, x=eda_feature_dist, color='Outcome', histnorm='probability density',
                                      title=f"Density Plot of {eda_feature_dist} by Outcome")
        else:
            fig_hist = px.histogram(original_df, x=eda_feature_dist, histnorm='probability density',
                                    title=f"Density Plot of {eda_feature_dist}")

    st.plotly_chart(fig_hist, use_container_width=True)

    # Comparative Box/Violin Plots
    st.subheader("Comparative Box/Violin Plots")
    eda_feature_comp = st.selectbox(
        "Select a feature for comparison between Outcome groups:",
        options=numeric_cols,
        key='comp_feature'
    )
    plot_type_comp = st.radio("Choose plot type:", ('Box Plot', 'Violin Plot'), horizontal=True, key='comp_plot_type')

    if plot_type_comp == 'Box Plot':
        fig_comp = px.box(original_df, x='Outcome', y=eda_feature_comp, color='Outcome',
                          title=f"Box Plot of {eda_feature_comp} by Outcome")
    else:
        fig_comp = px.violin(original_df, x='Outcome', y=eda_feature_comp, color='Outcome',
                             title=f"Violin Plot of {eda_feature_comp} by Outcome")
    st.plotly_chart(fig_comp, use_container_width=True)

    # Feature Importance Plot
    st.subheader("Feature Importance from Gradient Boosting Model")
    feature_importance_df = pd.DataFrame({
        'Feature': final_feature_columns, # Use loaded final_feature_columns
        'Importance': loaded_model.feature_importances_
    }).sort_values(by='Importance', ascending=True) # Sort ascending for horizontal bar chart

    fig_feat_imp = px.bar(feature_importance_df, x='Importance', y='Feature',
                          orientation='h', title='Feature Importance (Gradient Boosting Classifier)')
    st.plotly_chart(fig_feat_imp, use_container_width=True)


# --- Patient Cohort Comparison Section ---
with st.expander("Patient Cohort Comparison"):
    st.header("Compare Your Data with Cohort Averages")

    # Prepare data for cohort analysis: apply similar preprocessing steps as the model
    cohort_analysis_df = original_df.copy()

    # Replace 0s with NaNs for relevant features (as done in notebook for preprocessing)
    features_to_impute_zeros_global = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] # Defined globally for this section
    for col in features_to_impute_zeros_global:
        cohort_analysis_df[col] = cohort_analysis_df[col].replace(0, np.nan)

    # Impute NaNs with overall medians stored in imputation_medians
    numerical_features_for_cohort_comparison = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    for col in numerical_features_for_cohort_comparison:
        if col in cohort_analysis_df.columns and col in imputation_medians:
            cohort_analysis_df[col] = cohort_analysis_df[col].fillna(imputation_medians[col])

    # Apply Insulin capping as done during training preprocessing
    cohort_analysis_df['Insulin'] = cohort_analysis_df['Insulin'].apply(lambda x: min(x, INSULIN_UPPER_BOUND))

    # Calculate cohort means on this cleaned data
    cohort_means = cohort_analysis_df.groupby('Outcome')[numerical_features_for_cohort_comparison].mean()

    st.subheader("Average Feature Values by Outcome Group:")
    st.dataframe(cohort_means.rename(index={0: 'Non-Diabetic Cohort', 1: 'Diabetic Cohort'}).round(2))

    st.subheader("Your Data vs. Cohort Averages:")

    # Get current user input data from widgets
    current_user_input = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    comparison_data = []
    for feature in numerical_features_for_cohort_comparison:
        user_val = current_user_input[feature]
        non_diabetic_mean = cohort_means.loc[0, feature]
        diabetic_mean = cohort_means.loc[1, feature]

        non_diabetic_comparison = ""
        if user_val > non_diabetic_mean:
            non_diabetic_comparison = f"Higher ({user_val:.1f} vs {non_diabetic_mean:.1f})"
        elif user_val < non_diabetic_mean:
            non_diabetic_comparison = f"Lower ({user_val:.1f} vs {non_diabetic_mean:.1f})"
        else:
            non_diabetic_comparison = f"Same ({user_val:.1f})"

        diabetic_comparison = ""
        if user_val > diabetic_mean:
            diabetic_comparison = f"Higher ({user_val:.1f} vs {diabetic_mean:.1f})"
        elif user_val < diabetic_mean:
            diabetic_comparison = f"Lower ({user_val:.1f} vs {diabetic_mean:.1f})"
        else:
            diabetic_comparison = f"Same ({user_val:.1f})"

        comparison_data.append({
            "Feature": feature,
            "Your Value": f"{user_val:.1f}",
            "Vs. Non-Diabetic Cohort": non_diabetic_comparison,
            "Vs. Diabetic Cohort": diabetic_comparison
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df)
    st.info("Compare your values for each feature against the average values of 'Non-Diabetic' (Outcome=0) and 'Diabetic' (Outcome=1) cohorts. This can help identify which of your measurements align more closely with one group or the other.")

# --- What-If Scenario Analysis Section ---
with st.expander("What-If Scenario Analysis"):
    st.header("Adjust Parameters and See Impact")
    st.write("Change the values below to see how the prediction outcome and probability shift.")

    col_w1, col_w2, col_w3 = st.columns(3)

    with col_w1:
        pregnancies_w = st.number_input(
            "Number of Pregnancies",
            min_value=0, max_value=17, value=pregnancies, step=1, key='whatif_pregnancies'
        )
        glucose_w = st.number_input(
            "Glucose Level (mg/dL)",
            min_value=0, max_value=200, value=glucose, step=1, key='whatif_glucose'
        )
        blood_pressure_w = st.number_input(
            "Blood Pressure (mm Hg)",
            min_value=0, max_value=122, value=blood_pressure, step=1, key='whatif_blood_pressure'
        )

    with col_w2:
        skin_thickness_w = st.number_input(
            "Skin Thickness (mm)",
            min_value=0, max_value=99, value=skin_thickness, step=1, key='whatif_skin_thickness'
        )
        insulin_w = st.number_input(
            "Insulin Level (mu U/ml)",
            min_value=0, max_value=270, value=insulin, step=1, key='whatif_insulin'
        )
        bmi_w = st.number_input(
            "BMI (Body Mass Index)",
            min_value=0.0, max_value=70.0, value=bmi, step=0.1, key='whatif_bmi'
        )

    with col_w3:
        dpf_w = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.07, max_value=2.5, value=dpf, step=0.001, format="%.3f", key='whatif_dpf'
        )
        age_w = st.number_input(
            "Age (years)",
            min_value=21, max_value=81, value=age, step=1, key='whatif_age'
        )

    whatif_predict_button = st.button("Run What-If Analysis")

    if whatif_predict_button:
        whatif_input_data = {
            'Pregnancies': pregnancies_w,
            'Glucose': glucose_w,
            'BloodPressure': blood_pressure_w,
            'SkinThickness': skin_thickness_w,
            'Insulin': insulin_w,
            'BMI': bmi_w,
            'DiabetesPedigreeFunction': dpf_w,
            'Age': age_w
        }

        final_whatif_scaled = preprocess_input(whatif_input_data, imputation_medians, final_feature_columns, loaded_robust_scaler, loaded_standard_scaler, INSULIN_UPPER_BOUND)

        whatif_prediction_proba = loaded_model.predict_proba(final_whatif_scaled)

        if whatif_prediction_proba[0][1] >= prediction_threshold:
            whatif_prediction_label = 'Diabetic'
            whatif_display_message = st.error
            whatif_prob_display = f"Probability of Diabetes: **{whatif_prediction_proba[0][1]:.2f}**"
        else:
            whatif_prediction_label = 'Non-Diabetic'
            whatif_display_message = st.success
            whatif_prob_display = f"Probability of Diabetes: **{whatif_prediction_proba[0][1]:.2f}** (below threshold)"


        st.markdown("#### What-If Prediction Result:")
        whatif_display_message(f"The patient in this scenario is predicted to be **{whatif_prediction_label}**.")
        st.write(whatif_prob_display)

        st.info("Note: This 'what-if' result is based on the modified inputs and should be interpreted cautiously.")

        # Display interpretation for what-if
        st.markdown("### Personalized Interpretation and Advice for 'What-If' Scenario:")
        st.info(generate_interpretation(whatif_input_data, whatif_prediction_label, whatif_prediction_proba, prediction_threshold))


# --- Educational Insights / Dynamic Glossary Section ---
with st.expander("Educational Insights / Dynamic Glossary"):
    st.header("Understanding Diabetes and Key Terms")
    st.write("Here's some information to help you understand the terms used in this application and related to diabetes:")

    # Add an image example within this section
    st.image("diabetes.jpeg", caption='Diabetes Awareness', width=300)
    st.markdown("You can also use `use_column_width=True` to make the image fit the column, like this:")
    st.image("diabetics1.webp", caption='Blood Sugar Levels Explained', width=400)

    glossary = {
        "Pregnancies": {
            "definition": "Number of times pregnant.",
            "impact": "Higher numbers of pregnancies can sometimes be associated with an increased risk of developing type 2 diabetes, especially if there was a history of gestational diabetes."
        },
        "Glucose": {
            "definition": "Plasma glucose concentration a 2 hours in an oral glucose tolerance test. This measures the amount of sugar in your blood.",
            "impact": "High glucose levels are a primary indicator of diabetes. Consistently elevated glucose suggests the body isn't processing sugar effectively."
        },
        "BloodPressure": {
            "definition": "Diastolic blood pressure (mm Hg). This is the pressure in the arteries when the heart rests between beats.",
            "impact": "High blood pressure (hypertension) often co-occurs with diabetes and is a significant risk factor for cardiovascular complications."
        },
        "SkinThickness": {
            "definition": "Triceps skin fold thickness (mm). Used to estimate body fat.",
            "impact": "Increased skin fold thickness can correlate with higher body fat, which is a known risk factor for insulin resistance and type 2 diabetes."
        },
        "Insulin": {
            "definition": "2-Hour serum insulin (mu U/ml). Measures how much insulin the body is producing.",
            "impact": "Abnormal insulin levels (too high or too low for glucose levels) can indicate insulin resistance, impaired insulin production, or other issues related to diabetes."
        },
        "BMI": {
            "definition": "Body Mass Index (weight in kg / (height in m)^2). A measure of body fat based on height and weight.",
            "impact": "High BMI (overweight or obesity) is a major risk factor for developing type 2 diabetes due to its association with insulin resistance."
        },
        "DiabetesPedigreeFunction": {
            "definition": "A function that scores likelihood of diabetes based on family history.",
            "impact": "A higher DPF indicates a stronger genetic predisposition to diabetes, meaning family history plays a more significant role in an individual's risk."
        },
        "Age": {
            "definition": "Age in years.",
            "impact": "The risk of developing type 2 diabetes generally increases with age, especially after 45."
        },
        "Outcome": {
            "definition": "The target variable, indicating whether the patient has diabetes (1) or not (0).",
            "impact": "This is what the model is trying to predict."
        },
        "NewBMI (Categorical)": {
            "definition": "Categorical representation of BMI (Underweight, Normal, Overweight, Obesity 1-3).",
            "impact": "This feature helps the model understand the severity of weight status in a more granular way than just the continuous BMI value."
        },
        "NewInsulinScore (Categorical)": {
            "definition": "Categorical score for insulin (Normal or Abnormal).",
            "impact": "Simplifies insulin levels into a more interpretable category, indicating whether insulin is within a healthy range."
        },
        "NewGlucose (Categorical)": {
            "definition": "Categorical score for glucose (Low, Normal, Overweight, Secret).",
            "impact": "Provides a quick categorization of glucose levels, which helps the model identify ranges associated with different diabetes risks."
        }
    }

    selected_term = st.selectbox("Select a term to learn more:", list(glossary.keys()))

    if selected_term:
        term_info = glossary[selected_term]
        st.subheader(selected_term)
        st.write(f"**Definition:** {term_info['definition']}")
        st.write(f"**Impact on Diabetes:** {term_info['impact']}")

    st.markdown("--- ")
    st.write("For more detailed medical information, please consult a healthcare professional or reputable medical sources.")


# --- Actionable Recommendations / Goal Setter Section ---
with st.expander("Actionable Recommendations / Goal Setter"):
    st.header("Your Personalized Health Goals and Recommendations")
    st.write("This section will provide personalized health goals and recommendations to improve your diabetes risk.")

    # Only provide recommendations if the user is predicted to be Diabetic
    if 'prediction_label' in st.session_state and st.session_state.prediction_label == 'Diabetic':
        st.subheader("How to potentially move to 'Non-Diabetic' Status:")
        st.write("We'll simulate small, realistic changes to your input values to see how they impact the prediction.")

        # Get the initial user input data (from the main prediction part)
        initial_user_input_data = st.session_state.user_input_data

        # Features to iterate on and their step sizes
        features_to_adjust = {
            'Glucose': {'step': 5, 'min_val': 70, 'max_attempts': 10, 'unit': 'mg/dL'},
            'BMI': {'step': 0.5, 'min_val': 18.5, 'max_attempts': 10, 'unit': ''},
            'BloodPressure': {'step': 2, 'min_val': 60, 'max_attempts': 5, 'unit': 'mm Hg'},
            'Insulin': {'step': 10, 'min_val': 16, 'max_attempts': 5, 'unit': 'mu U/ml'},
            'Age': {'step': 1, 'min_val': initial_user_input_data['Age']-10, 'max_attempts': 5, 'unit': 'years'} # Age adjustment is tricky, typically not modifiable directly
        }

        st.markdown("###### Minimal Changes to Reduce Diabetes Probability:")
        found_recommendations = False

        for feature, params in features_to_adjust.items():
            temp_input_data = initial_user_input_data.copy()
            original_value = temp_input_data[feature]
            current_prediction_probability = st.session_state.prediction_proba[0][1]

            if current_prediction_probability < prediction_threshold:
                continue

            for i in range(params['max_attempts']):
                new_value = original_value - (i + 1) * params['step']
                if new_value < params['min_val']:
                    new_value = params['min_val']

                temp_input_data[feature] = new_value

                adjusted_input_scaled = preprocess_input(temp_input_data, imputation_medians, final_feature_columns, loaded_robust_scaler, loaded_standard_scaler, INSULIN_UPPER_BOUND)
                adjusted_prediction_proba = loaded_model.predict_proba(adjusted_input_scaled)

                if adjusted_prediction_proba[0][1] < prediction_threshold * 0.95:
                    st.success(f"*   **Reduce {feature} by approximately {original_value - new_value:.1f} {params['unit']}** (from {original_value:.1f} to {new_value:.1f}): This could reduce your diabetes probability to {adjusted_prediction_proba[0][1]:.2f} and potentially change your status to 'Non-Diabetic'.")
                    found_recommendations = True
                    break
                elif new_value == params['min_val'] and adjusted_prediction_proba[0][1] >= prediction_threshold:
                    st.info(f"*   Even with {feature} at its practical minimum ({new_value:.1f} {params['unit']}), your predicted probability ({adjusted_prediction_proba[0][1]:.2f}) remains above the threshold. Consider broader lifestyle changes.")
                    found_recommendations = True
                    break

        if not found_recommendations and current_prediction_probability >= prediction_threshold:
            st.write("It's challenging to find simple, isolated changes to flip your prediction based on these parameters. A holistic approach to health management is recommended. Focus on overall healthy eating, regular physical activity, and stress management.")

        st.markdown("### General Health Goals:->\n\n" + general_health_goals_markdown_content)
    else:
        st.info("The model currently predicts a 'Non-Diabetic' status for this profile. Continue maintaining a healthy lifestyle to reduce future risks!")


# --- Feedback Section ---
# Initialize session state variable for the text area's *content*
if 'feedback_content_to_display' not in st.session_state:
    st.session_state.feedback_content_to_display = ""
# Define the callback function for the submit button
def submit_feedback_callback():
    # Retrieve the current text from the text area using its key
    current_feedback_text = st.session_state.feedback_text_widget_key

    if current_feedback_text:
        conn = None
        try:
            conn = get_db_connection()
            if conn is None: # Handle case where connection failed
                st.error("Failed to connect to the database.")
                return
            cursor = conn.cursor()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute("INSERT INTO feedback_logs (timestamp, feedback_text) VALUES (?, ?)",
                               (timestamp, current_feedback_text))
            conn.commit()
            st.success("Thank you for your feedback!")
            # Clear the text area's content by updating the session state variable
            # that controls its 'value' argument. This is safe within a callback.
            st.session_state.feedback_content_to_display = ""
        except Exception as e:
            st.error(f"Error storing feedback: {e}")
        finally:
            if conn:
                conn.close()
    else:
        st.info("Please enter some feedback before submitting.")

with st.expander("Submit Feedback"):
    st.header("Help Us Improve!")
    st.write("We appreciate your feedback to make this application better.")

    st.text_area(
        "Your Feedback (optional)",
        height=100,
        key='feedback_text_widget_key', # This key will hold the user's current input
        value=st.session_state.feedback_content_to_display # This controls the displayed content
    )

    # The button calls the callback function
    st.button("Submit Feedback", on_click=submit_feedback_callback, key='submit_feedback_button')


# --- Admin/Metrics Dashboard Section ---
with st.expander("Admin/Metrics Dashboard"):
    st.header("Application Metrics and Feedback")

    st.subheader("User Feedback Logs")
    feedback_df = fetch_feedback_logs()
    if not feedback_df.empty:
        st.dataframe(feedback_df)
    else:
        st.info("No feedback submitted yet.")

    st.subheader("Application Usage Statistics")
    usage_df = fetch_usage_logs()
    if not usage_df.empty:
        total_visits = len(usage_df)
        unique_sessions = usage_df['session_id'].nunique()
        st.write(f"Total visits recorded: **{total_visits}**")
        st.write(f"Unique sessions: **{unique_sessions}**")
        st.dataframe(usage_df)
    else:
        st.info("No usage data recorded yet.")
