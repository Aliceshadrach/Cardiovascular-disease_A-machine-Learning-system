import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Cardiovascular ML - Task 1", layout="wide")
st.title("🫀 Cardiovascular Disease ML Pipeline")
st.markdown("**COM763 Task 1 — Alice** | End-to-end data preprocessing, EDA, feature engineering & model training")

# ─── Sidebar: Upload CSV ────────────────────────────────────────────────────
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload `cardio_train.csv`", type=["csv"])

if uploaded_file is None:
    st.info("👈 Please upload `cardio_train.csv` in the sidebar to get started.")
    st.stop()

# ─── 1. Load Data ─────────────────────────────────────────────────────────
st.header("1. Load & Inspect Data")
df = pd.read_csv(uploaded_file, sep=';')
st.success(f"Dataset loaded: **{df.shape[0]:,} rows × {df.shape[1]} columns**")
st.dataframe(df.head(), use_container_width=True)

with st.expander("📊 Summary Statistics"):
    st.dataframe(df.describe(), use_container_width=True)

with st.expander("🔢 Data Types"):
    st.dataframe(df.dtypes.rename("dtype").to_frame(), use_container_width=True)

# ─── 2. Preprocessing ─────────────────────────────────────────────────────
st.header("2. Preprocessing")

# Convert age
df['age'] = (df['age'] / 365).round().astype(int)
st.markdown("✅ **Age** converted from days → years")

# Missing values
missing = df.isnull().sum()
st.markdown(f"✅ Missing values check — total missing: **{missing.sum()}**")
st.dataframe(missing.rename("missing").to_frame().T, use_container_width=True)

# Impute
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
st.markdown("✅ Median imputation applied")

# ─── 3. EDA ───────────────────────────────────────────────────────────────
st.header("3. Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Feature Distributions")
    fig, ax = plt.subplots(figsize=(10, 10))
    df.hist(ax=ax, bins=20, edgecolor='black')
    plt.suptitle("Distribution of Numerical Features")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─── 4. Outlier Handling ──────────────────────────────────────────────────
st.header("4. Outlier Handling")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Before Outlier Removal")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, ax=ax)
    ax.set_title("Boxplot of Features (Before)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

Q1 = df_imputed.quantile(0.25)
Q3 = df_imputed.quantile(0.75)
IQR = Q3 - Q1
outlier_condition = ((df_imputed < (Q1 - 1.5 * IQR)) | (df_imputed > (Q3 + 1.5 * IQR)))
df_no_outliers = df_imputed[~outlier_condition.any(axis=1)].copy()

with col2:
    st.subheader("After Outlier Removal")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_no_outliers, ax=ax)
    ax.set_title("Boxplot of Features (After)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.info(f"Rows removed: **{len(df_imputed) - len(df_no_outliers):,}** | Remaining: **{len(df_no_outliers):,}**")

# ─── 5. Feature Engineering ───────────────────────────────────────────────
st.header("5. Feature Engineering")

def categorize_age(age):
    if age < 21:
        return 'Young'
    elif age < 35:
        return 'Adult'
    else:
        return 'Middle_Aged'

def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

df_no_outliers.loc[:, 'BMI'] = df_no_outliers['weight'] / ((df_no_outliers['height'] / 100) ** 2)
df_no_outliers.loc[:, 'Age_Category'] = df_no_outliers['age'].apply(categorize_age).astype('category')
df_no_outliers.loc[:, 'BMI_Category'] = df_no_outliers['BMI'].apply(categorize_bmi).astype('category')

st.markdown("✅ **BMI** computed from weight/height²")
st.markdown("✅ **Age_Category** and **BMI_Category** created")
st.dataframe(df_no_outliers[['age', 'Age_Category', 'BMI', 'BMI_Category']].head(10), use_container_width=True)

# ─── 6. Encoding ──────────────────────────────────────────────────────────
st.header("6. Encoding Categorical Variables")

df_encoded = pd.get_dummies(df_no_outliers, columns=['Age_Category', 'BMI_Category'], prefix=['Age', 'BMI'])

le_age = LabelEncoder()
le_bmi = LabelEncoder()
df_no_outliers.loc[:, 'Age_Category_Label'] = le_age.fit_transform(df_no_outliers['Age_Category'])
df_no_outliers.loc[:, 'BMI_Category_Label'] = le_bmi.fit_transform(df_no_outliers['BMI_Category'])

st.markdown("✅ **One-Hot Encoding** applied (for modelling) | **Label Encoding** also stored")
st.dataframe(df_encoded.head(5), use_container_width=True)

# ─── 7. Scaling ───────────────────────────────────────────────────────────
st.header("7. Feature Scaling")

numeric_columns = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'BMI']
scaler = StandardScaler()
df_scaled = df_encoded.copy()
df_scaled[numeric_columns] = scaler.fit_transform(df_scaled[numeric_columns])

fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=df_scaled[numeric_columns], ax=ax)
ax.set_title('Box Plots of Scaled Numeric Features')
ax.set_xlabel('Features')
ax.set_ylabel('Scaled Values')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ─── 8. Model Training ────────────────────────────────────────────────────
st.header("8. Model Training — Linear Regression Pipeline")

# Use cardio (target) column — notebook uses 'Outcome' but dataset has 'cardio'
target_col = 'cardio' if 'cardio' in df_scaled.columns else df_scaled.columns[-1]
st.markdown(f"Target column: **`{target_col}`**")

X = df_scaled.drop(columns=[target_col])
y = df_scaled[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.markdown(f"Train: **{X_train.shape}** | Test: **{X_test.shape}**")

preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

if st.button("🚀 Train Linear Regression Model"):
    with st.spinner("Training..."):
        lr_pipeline.fit(X_train, y_train)
        preds = lr_pipeline.predict(X_test)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE",  f"{mae:.4f}")
    col2.metric("RMSE", f"{rmse:.4f}")
    col3.metric("R²",   f"{r2:.4f}")

    # Prediction scatter plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, preds, alpha=0.5, edgecolors='k', linewidths=0.4, color='royalblue')
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, 'r--', label='Perfect Prediction')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Save model
    model_path = "linear_regression_model.pkl"
    joblib.dump(lr_pipeline, model_path)
    st.success(f"✅ Model saved to `{model_path}`")

    with open(model_path, "rb") as f:
        st.download_button("⬇️ Download Model (.pkl)", f, file_name=model_path)
