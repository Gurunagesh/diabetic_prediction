# Empowering Your Health: Diabetes Risk Predictor App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.streamlit.app)

## Project Overview

This interactive Streamlit application is designed to help individuals understand and manage their potential diabetes risk based on key health metrics. Built with advanced machine learning (Gradient Boosting Classifier), the app provides not only a prediction but also personalized insights, educational resources, actionable recommendations, and a robust administrative dashboard for feedback and usage tracking.

Our goal is to empower users with data-driven insights, encouraging proactive health management and informed decision-making concerning diabetes risk. The application employs a comprehensive preprocessing pipeline, including handling missing values, outlier treatment, and feature engineering, consistent with the model's training process.

## Implemented Features

### 1. Interactive Patient Data Input

*   **Description**: A user-friendly interface allows you to easily input various health parameters related to diabetes risk, such as `Pregnancies`, `Glucose`, `Blood Pressure`, `Skin Thickness`, `Insulin`, `BMI`, `Diabetes Pedigree Function`, and `Age`.
*   **Impact**: Enables personalized risk assessment based on individual health profiles.

### 2. Dynamic Prediction Threshold

*   **Description**: A slider widget empowers users to adjust the probability threshold for classifying a patient as 'Diabetic' or 'Non-Diabetic'. This allows exploration of the model's sensitivity.
*   **Impact**: Provides flexibility in risk interpretation, catering to different levels of risk aversion or clinical contexts.

### 3. Instant Prediction & Interpretation

*   **Description**: Upon input and threshold selection, the app provides an immediate prediction (Diabetic/Non-Diabetic) along with the probability of diabetes. A detailed interpretation explains the prediction and highlights the significance of each input feature in your personalized risk profile.
*   **Impact**: Offers immediate clarity on diabetes risk and helps users understand the contributing factors.

### 4. Personalized Health Score / Risk Indicator

*   **Description**: A numerical 'Health Score' (0-100) is calculated based on the user's input features, providing an intuitive summary of overall diabetes risk. Higher scores indicate increased risk.
*   **Impact**: Complements the binary prediction with a quantifiable and easily digestible metric for overall risk assessment.

### 5. Interactive Exploratory Data Analysis (EDA)

*   **Description**: An expandable section provides dynamic visualizations of the dataset used to train the model, offering insights into data patterns and relationships.
    *   **Correlation Heatmap**: Visualizes the correlation matrix between all numerical features.
    *   **Feature Distributions**: Interactive histograms and density plots to explore the distribution of individual features, with an option to segment by `Outcome` (Diabetic/Non-Diabetic).
    *   **Comparative Box/Violin Plots**: Compares feature distributions between 'Diabetic' and 'Non-Diabetic' groups.
    *   **Feature Importance**: Displays the importance of each feature as learned by the Gradient Boosting Classifier model.
*   **Impact**: Enhances transparency and allows users to understand the underlying data characteristics and model's decision-making process.

### 6. Patient Cohort Comparison

*   **Description**: This section displays mean feature values for both 'Diabetic' and 'Non-Diabetic' cohorts from the original dataset. It then compares the user's input values against these cohort averages, highlighting whether individual metrics are higher, lower, or similar to each group.
*   **Impact**: Contextualizes personal health metrics against population averages, aiding in understanding relative risk.

### 7. Educational Insights / Dynamic Glossary

*   **Description**: An interactive glossary provides definitions and explanations for key medical terms and features used in the application (e.g., Glucose, BMI, Insulin, Diabetes Pedigree Function). Each entry details its impact related to diabetes.
*   **Impact**: Improves user literacy regarding diabetes and its associated terminology, promoting a deeper understanding of health concepts.

### 8. Actionable Recommendations / Goal Setter

*   **Description**: For users predicted to be 'Diabetic' (or with a high probability), this feature simulates small, realistic adjustments to individual features (e.g., reducing Glucose, BMI) to identify minimal changes that could potentially shift the prediction to 'Non-Diabetic' or significantly lower the diabetes probability. It also provides general health goals.
*   **Impact**: Offers personalized, data-driven advice and practical steps for lifestyle changes, motivating users towards better health outcomes.

### 9. Feedback Submission

*   **Description**: A dedicated section allows users to submit optional feedback on the application, which is stored in an SQLite/PostgreSQL database.
*   **Impact**: Provides a channel for continuous improvement based on user experience and suggestions.

### 10. Usage Tracking & Admin Dashboard

*   **Description**: The application logs each unique session to a database. An administrative dashboard is available (primarily for developers/administrators) to monitor usage statistics (total visits, unique sessions) and review submitted user feedback.
*   **Impact**: Supports application monitoring, performance analysis, and data-driven improvements by providing insights into user engagement and feedback.

## Live Demo

Experience the app live: [Live Demo - Diabetes Prediction App](https://your-streamlit-app-url.streamlit.app) *(Replace with your deployed app URL)*

## How to Run Locally

Follow these steps to set up and run the application on your local machine:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/diabetic_prediction.git
cd diabetic_prediction
```
*(Replace `yourusername` and `diabetic_prediction` with your actual GitHub details)*

### 2. Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv
# On Windows
venc\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

Install all required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Database Setup

The application uses an SQLite database (`app_data.db`) by default for logging feedback and usage. You need to initialize this database:

```bash
python db_setup.py
```
This will create `app_data.db` and set up the necessary tables (`feedback_logs`, `usage_logs`).

**For Production (PostgreSQL)**:
If you intend to deploy with a PostgreSQL database, the application is configured to use a `DATABASE_URL` environment variable. Set this variable with your PostgreSQL connection string (e.g., `postgresql://user:password@host:port/dbname`). If `DATABASE_URL` is present, `psycopg2` will be used to connect. Ensure `psycopg2` is installed (`pip install psycopg2-binary`) and uncommented in `requirements.txt` if you choose this option.

### 5. Run the Streamlit Application

```bash
streamlit run diabetes_prediction_app.py
```

This command will open the application in your default web browser.

## Model Artifacts and Data

The machine learning model (`gbc_model.pkl`), scalers (`robust_scaler.pkl`, `standard_scaler.pkl`), imputation medians (`imputation_medians.pkl`), and final feature columns (`final_feature_columns.pkl`) are all pre-trained and saved as pickle files in the project root.

The dataset (`diabetes.csv`) used for training and EDA is also included in the repository.

## Deployment

This application can be easily deployed to platforms like [Streamlit Community Cloud](https://streamlit.io/cloud) or [Hugging Face Spaces](https://huggingface.co/spaces). Simply link your GitHub repository, and the platform will handle the deployment using the `requirements.txt` and `diabetes_prediction_app.py` files. Ensure your `requirements.txt` includes all necessary packages and `psycopg2-binary` if using an external PostgreSQL database.

## Contributing / Feedback

We welcome contributions, suggestions, and feedback! If you have any ideas, bug reports, or want to contribute to the code, please feel free to:

*   Submit feedback directly through the app's 'Submit Feedback' section.
*   Open an issue on the [GitHub repository](https://github.com/yourusername/diabetic_prediction/issues).
*   Fork the repository and submit a pull request with your enhancements.

## Disclaimer

This Diabetes Risk Predictor is intended for informational and educational purposes only. It is not designed to provide medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional for any health concerns or before making any decisions related to your health or treatment. The predictions and interpretations generated by this application are based on statistical models and historical data and should not be considered a substitute for professional medical guidance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
