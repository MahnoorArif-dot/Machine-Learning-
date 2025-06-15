import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Apply page configuration
st.set_page_config(page_title="🎓 Student Performance Predictor", layout="wide")

# Custom CSS styling for aesthetics
st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
        }
        .stSidebar {
            background-color: #eef2f7;
        }
        .block-container {
            padding-top: 2rem;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .reportview-container .markdown-text-container {
            font-size: 1.1rem;
        }
        .stButton>button {
            background-color: #2c3e50;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
section = st.sidebar.radio("Choose a Section", ["Introduction", "EDA", "Modeling","Predictor", "Conclusion"])

# Sidebar file uploader
st.sidebar.markdown("### 📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Load data if available
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';')  # You can use sep=',' for other datasets
    st.sidebar.success("✅ File uploaded successfully!")
    uploaded_file.seek(0)

# Title
st.title("🎓 Student Performance Predictor App")

# Introduction Section
if section == "Introduction":
    st.markdown("### 🧾 Introduction")
    st.info("""
        Welcome to the **Student Grade Predictor App** – an intelligent tool designed for educators, 
        advisors, and students to analyze and predict academic performance using various attributes.
    """)
    st.image("https://images.unsplash.com/photo-1577896851231-70ef18881754", width=600, caption="Education and Insights")

elif section == "EDA":
    st.header("📊 Exploratory Data Analysis")

    if df is not None:
        st.markdown("---")

        # File Overview
        with st.expander("📄 File Overview", expanded=True):
            st.write("**Filename:**", uploaded_file.name)
            st.write("**Shape:**", df.shape)
            st.write("**Columns:**", df.columns.tolist())

        # Show dataframe preview
        with st.expander("🧾 Data Preview"):
            st.dataframe(df.head(10), use_container_width=True)

        # Buttons for key EDA steps
        st.markdown("### 🔍 Explore Data Sections")

        colA, colB, colC = st.columns(3)
        if colA.button("Summary Stats"):
            st.subheader("📋 Summary Statistics")
            st.write(df.describe())

        if colB.button("Missing Values"):
            st.subheader("🧩 Missing Values")
            st.write(df.isnull().sum())

        if colC.button("Download Cleaned CSV"):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "cleaned_data.csv", "text/csv")

        st.markdown("---")

        # Tabs for visualizations
        tab1, tab2, tab3 = st.tabs(["📈 Distribution", "📉 Correlation", "📊 Boxplots"])

        with tab1:
            st.subheader("🎯 Final Grade Distribution (G3)")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.histplot(df['G3'], bins=20, kde=True, ax=ax1, color='skyblue')
            ax1.set_title("Distribution of Final Grade (G3)")
            st.pyplot(fig1)

        with tab2:
            st.subheader("🔗 Correlation Matrix")
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            corr_matrix = df.corr(numeric_only=True)
            sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".2f", ax=ax2)
            st.pyplot(fig2)

            st.markdown("#### 📌 Feature Correlation with G3")
            corr_with_target = corr_matrix['G3'].sort_values(ascending=False)
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            corr_with_target.drop('G3').plot(kind='barh', color='salmon', ax=ax3)
            ax3.set_title("Correlation with Final Grade (G3)")
            st.pyplot(fig3)

        with tab3:
            st.subheader("📦 Boxplots")
            st.markdown("#### G3 by Gender")
            fig4, ax4 = plt.subplots()
            sns.boxplot(x='sex', y='G3', data=df, palette='Set2', ax=ax4)
            st.pyplot(fig4)

            st.markdown("#### G3 by Family Relationship")
            fig5, ax5 = plt.subplots()
            sns.boxplot(x='famrel', y='G3', data=df, palette='coolwarm', ax=ax5)
            st.pyplot(fig5)

            st.markdown("#### G3 by Study Time")
            fig6, ax6 = plt.subplots()
            sns.boxplot(x='studytime', y='G3', data=df, palette='Blues', ax=ax6)
            st.pyplot(fig6)

            st.markdown("#### G1, G2, G3 Overview")
            fig7, ax7 = plt.subplots()
            sns.boxplot(data=df[['G1', 'G2', 'G3']], palette='pastel', ax=ax7)
            st.pyplot(fig7)

        # Pass/Fail Analysis
        with st.expander("📘 Pass/Fail Distribution"):
            df['pass'] = (df['G3'] >= 10).astype(int)
            fig8, ax8 = plt.subplots()
            sns.countplot(x='pass', data=df, palette='Accent', ax=ax8)
            ax8.set_xticklabels(['Fail (0)', 'Pass (1)'])
            st.pyplot(fig8)

    else:
        st.warning("⚠️ Please upload a CSV file to perform EDA.")


elif section == "Modeling":
    st.header("🧠 Modeling and Evaluation")

    if df is not None:
        with st.spinner("Preparing data and training models..."):

            # ---------------------- Encoding ----------------------
            binary_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus',
                           'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                           'higher', 'internet', 'romantic']
            for col in binary_cols:
                df[col] = df[col].map({'yes': 1, 'no': 0, 'GP': 1, 'MS': 0,
                                       'F': 1, 'M': 0, 'U': 1, 'R': 0, 'LE3': 1, 'GT3': 0,
                                       'T': 1, 'A': 0})

            nominal_cols = ['Mjob', 'Fjob', 'reason', 'guardian']
            df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

            # ---------------------- Splitting ----------------------
            X = df.drop('G3', axis=1)
            y = df['G3']

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            # ---------------------- Scaling ----------------------
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # ---------------------- Models ----------------------
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.metrics import mean_squared_error, r2_score

            lr = LinearRegression()
            lr.fit(X_train_scaled, y_train)
            y_pred_lr = lr.predict(X_test_scaled)

            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)

            gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            gb.fit(X_train, y_train)
            y_pred_gb = gb.predict(X_test)

            # ---------------------- GridSearch ----------------------
            from sklearn.model_selection import GridSearchCV
            param_grid_rf = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5]
            }

            grid_rf = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid_rf, cv=3, scoring='r2', n_jobs=-1)
            grid_rf.fit(X_train, y_train)
            best_rf = grid_rf.best_estimator_
            y_pred_best_rf = best_rf.predict(X_test)

        # ---------------------- Metrics ----------------------
        st.subheader("🔍 Model Performance Comparison")
        results = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting'],
            'R2 Score': [r2_score(y_test, y_pred_lr),
                         r2_score(y_test, y_pred_rf),
                         r2_score(y_test, y_pred_gb)]
        })

        st.dataframe(results)

        st.bar_chart(data=results.set_index('Model'))

        st.markdown("#### ✅ Best Random Forest Parameters")
        st.code(grid_rf.best_params_)

        # ---------------------- Feature Importance ----------------------
        import matplotlib.pyplot as plt
        import seaborn as sns

        importances = pd.Series(best_rf.feature_importances_, index=X.columns)
        top_features = importances.sort_values(ascending=False).head(10)

        st.subheader("🔎 Top 10 Important Features (Random Forest)")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        top_features.plot(kind='barh', color='teal', ax=ax1)
        st.pyplot(fig1)

        # ---------------------- Actual vs Predicted ----------------------
        st.subheader("📌 Actual vs Predicted Final Grades")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.scatter(y_test, y_pred_best_rf, alpha=0.6, color='orange')
        ax2.plot([0, 20], [0, 20], '--r')
        ax2.set_xlabel("Actual Grade")
        ax2.set_ylabel("Predicted Grade")
        st.pyplot(fig2)

        # ---------------------- Residuals ----------------------
        residuals = y_test - y_pred_rf
        st.subheader("📊 Residual Distribution (Random Forest)")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.histplot(residuals, bins=20, kde=True, color='purple', ax=ax3)
        st.pyplot(fig3)

        # ---------------------- Cross-validation ----------------------
        from sklearn.model_selection import cross_val_score
        st.subheader("📈 Cross-Validation Results (Random Forest)")
        cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='r2')
        st.write("Fold-wise R2 Scores:", cv_scores)
        st.write("Mean CV R2 Score:", round(cv_scores.mean(), 4))

        # ---------------------- Feature Heatmap ----------------------
        st.subheader("🌡️ Feature Importance Heatmap")
        fig4, ax4 = plt.subplots(figsize=(10, 1))
        sns.heatmap(importances.values.reshape(1, -1), cmap='viridis',
                    annot=False, cbar=True, xticklabels=importances.index, ax=ax4)
        ax4.set_yticks([])
        ax4.set_title("Random Forest Feature Importances")
        plt.xticks(rotation=90)
        st.pyplot(fig4)

    else:
        st.warning("⚠️ Please upload data in the EDA section before proceeding to Modeling.")

elif section == "Predictor":
    # Custom CSS for styling
    st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .prediction-result {
            font-size: 1.5rem;
            color: #2e7d32;
            padding: 1.5rem;
            background-color: #e8f5e9;
            border-radius: 10px;
            margin: 1rem 0;
            text-align: center;
        }
        .feature-section {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

    # Load the trained pipeline
    @st.cache_data
    def load_model():
        try:
            return joblib.load('student_grade_pipeline.pkl')
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return None

    pipeline = load_model()

    # Mapping dictionaries
    binary_map = {'yes': 1, 'no': 0, 'GP': 1, 'MS': 0, 'F': 1, 'M': 0,
                'U': 1, 'R': 0, 'LE3': 1, 'GT3': 0, 'T': 1, 'A': 0}
    studytime_map = {"<2h": 1, "2-5h": 2, "5-10h": 3, ">10h": 4}
    traveltime_map = {"<15min": 1, "15-30min": 2, "30min-1h": 3, ">1h": 4}
    scale_map = {"Very low": 1, "Low": 2, "Medium": 3, "High": 4, "Very high": 5}
    edu_map = {"None": 0, "Primary": 1, "Middle": 2, "Secondary": 3, "Higher": 4}
    job_map = {'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4}
    reason_map = {'home': 0, 'reputation': 1, 'course': 2, 'other': 3}
    guardian_map = {'mother': 0, 'father': 1, 'other': 2}

    # Title
    st.title("🎓 Student Performance Predictor")

    # Tabs
    tab1, tab2 = st.tabs(["📊 Predict", "📈 Insights"])

    with tab1:
        with st.form("student_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Academic Information")
                g1 = st.slider("First period grade (G1)", 0, 20, 10)
                g2 = st.slider("Second period grade (G2)", 0, 20, 10)
                studytime = st.selectbox("Weekly study time", list(studytime_map.keys()))
                failures = st.slider("Number of past class failures", 0, 4, 0)
                absences = st.slider("Number of school absences", 0, 30, 0)
            with col2:
                st.subheader("Personal Information")
                age = st.slider("Age", 15, 22, 18)
                famrel = st.slider("Family relationship quality (1-5)", 1, 5, 3)
                health = st.slider("Health status (1-5)", 1, 5, 3)
                traveltime = st.selectbox("Travel time to school", list(traveltime_map.keys()))
                freetime = st.selectbox("Free time after school", list(scale_map.keys()))
                goout = st.selectbox("Going out with friends", list(scale_map.keys()))

            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Family Background")
                Medu = st.selectbox("Mother's education", list(edu_map.keys()))
                Fedu = st.selectbox("Father's education", list(edu_map.keys()))
                Mjob = st.selectbox("Mother's job", list(job_map.keys()))
                Fjob = st.selectbox("Father's job", list(job_map.keys()))
                reason = st.selectbox("Reason to choose school", list(reason_map.keys()))
                guardian = st.selectbox("Guardian", list(guardian_map.keys()))
            with col4:
                st.subheader("Habits & Support")
                Dalc = st.selectbox("Workday alcohol consumption", list(scale_map.keys()))
                Walc = st.selectbox("Weekend alcohol consumption", list(scale_map.keys()))
                internet = st.radio("Internet access at home", ["yes", "no"])
                higher = st.radio("Wants higher education", ["yes", "no"])
                paid = st.radio("Paid classes", ["yes", "no"])
                schoolsup = st.radio("School extra support", ["yes", "no"])
                famsup = st.radio("Family educational support", ["yes", "no"])
                activities = st.radio("Extra-curricular activities", ["yes", "no"])
                nursery = st.radio("Attended nursery school", ["yes", "no"])
                romantic = st.radio("In a romantic relationship", ["yes", "no"])

            st.subheader("School Information")
            scol1, scol2, scol3, scol4 = st.columns(4)
            with scol1:
                school = st.radio("School", ["GP", "MS"])
            with scol2:
                sex = st.radio("Sex", ["F", "M"])
            with scol3:
                address = st.radio("Address", ["U", "R"])
            with scol4:
                famsize = st.radio("Family size", ["LE3", "GT3"])
                Pstatus = st.radio("Parents' cohabitation", ["T", "A"])

            submitted = st.form_submit_button("Predict Final Grade")

            if submitted and pipeline:
                    try:
                        input_data = {
                            'school': binary_map[school],
                            'sex': binary_map[sex],
                            'age': age,
                            'address': binary_map[address],
                            'famsize': binary_map[famsize],
                            'Pstatus': binary_map[Pstatus],
                            'Medu': edu_map[Medu],
                            'Fedu': edu_map[Fedu],
                            'Mjob': Mjob,
                            'Fjob': Fjob,
                            'reason': reason,
                            'guardian': guardian,
                            'traveltime': traveltime_map[traveltime],
                            'studytime': studytime_map[studytime],
                            'failures': failures,
                            'schoolsup': binary_map[schoolsup],
                            'famsup': binary_map[famsup],
                            'paid': binary_map[paid],
                            'activities': binary_map[activities],
                            'nursery': binary_map[nursery],
                            'higher': binary_map[higher],
                            'internet': binary_map[internet],
                            'romantic': binary_map[romantic],
                            'famrel': famrel,
                            'freetime': scale_map[freetime],
                            'goout': scale_map[goout],
                            'Dalc': scale_map[Dalc],
                            'Walc': scale_map[Walc],
                            'health': health,
                            'absences': absences,
                            'G1': g1,
                            'G2': g2
                        }

                        # Convert to DataFrame
                        input_df = pd.DataFrame([input_data])

                        # One-hot encode categorical columns
                        categorical_cols = ['Mjob', 'Fjob', 'reason', 'guardian']
                        input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols)

                        # Load training feature names to match
                        features = joblib.load("features.pkl")  # list of feature names from training

                        # Add missing columns with 0 and ensure order
                        for col in features:
                            if col not in input_df_encoded.columns:
                                input_df_encoded[col] = 0
                        input_df_encoded = input_df_encoded[features]

                        # Predict
                        prediction = pipeline.predict(input_df_encoded)[0]
                        prediction = max(0, min(20, round(prediction, 1)))

                        st.markdown(f"""
                        <div class="prediction-result">
                            <strong>Predicted Final Grade (G3):</strong> {prediction}/20
                        </div>
                        """, unsafe_allow_html=True)

                        # Plot grade progression
                        fig, ax = plt.subplots(figsize=(10, 5))
                        grades = [g1, g2, prediction]
                        labels = ['G1', 'G2', 'Predicted G3']
                        colors = ['#3498db', '#3498db', '#2ecc71']
                        bars = ax.bar(labels, grades, color=colors)
                        ax.set_ylim(0, 20)
                        ax.set_ylabel('Grade')
                        ax.set_title('Grade Progression')
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom', fontsize=12)
                        st.pyplot(fig)

                        st.info(f"**Confidence Range:** {max(0, round(prediction - 2))} - {min(20, round(prediction + 2))}")
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")

elif section == "Conclusion":
    st.header("✅ Conclusion")

    st.subheader("🏁 Final Thoughts")
    st.markdown("""
This Student Performance Predictor demonstrates the potential of machine learning to forecast student academic outcomes using a variety of features such as academic history, personal background, and behavioral indicators.
""")

    st.subheader("📊 Model Evaluation Summary")

    st.markdown("**🔹 Cross-Validation (Mean R² Scores):**")
    st.markdown("""
- **Linear Regression:** 0.836 ± 0.043  
- **Random Forest Regressor:** 0.854 ± 0.064  
- **Gradient Boosting Regressor:** 0.841 ± 0.061  
""")

    st.markdown("**🔹 Test Set Performance:**")
    st.markdown("""
- **Linear Regression:**  
  - MSE: 1.476  
  - R² Score: 0.849  

- **Random Forest Regressor:**  
  - MSE: 1.589  
  - R² Score: 0.837  

- **Gradient Boosting Regressor:**  
  - MSE: 1.793  
  - R² Score: 0.816  
""")

    st.markdown("**🔹 Best Random Forest Parameters:**")
    st.code("{'max_depth': None, 'min_samples_split': 5, 'n_estimators': 100}")
    st.markdown("Best Random Forest R² Score: **0.839**")

    st.subheader("📌 Key Takeaways")
    st.markdown("""
- Random Forest achieved the best cross-validation R² score overall.
- Linear Regression performed slightly better on the test set, suggesting strong generalization.
- Gradient Boosting showed consistent but slightly lower performance.
- Feature interpretability remains crucial, especially for educational decisions.
""")

    st.subheader("🚀 Future Enhancements")
    st.markdown("""
- Incorporate more student-related behavioral and psychological data.
- Add model explainability tools like SHAP or LIME.
- Deploy improved pipelines for continuous model evaluation.
""")

    st.success("This project showcases how machine learning can support early intervention strategies and improve educational outcomes through data-driven insights.")

