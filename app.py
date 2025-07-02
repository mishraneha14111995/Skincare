import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("influencer_skincare_survey_dubai.csv")

df = load_data()

# Sidebar
st.sidebar.title("Dashboard Navigation")
tabs = ["Data Visualization", "Classification", "Clustering", "Association Rule Mining", "Regression"]
selected_tab = st.sidebar.radio("Go to", tabs)

# Data Preprocessing for Classification
def preprocess_classification(data):
    df = data.copy()
    le = LabelEncoder()
    categorical_cols = ["Gender", "Income (AED)", "Employment Status", "Location in Dubai",
                        "Skincare Purchase Frequency", "Follow Influencer Recommendations",
                        "Trusted Influencer Tier", "Preferred Platform", "Persuasive Ad Format",
                        "Browse Time", "Willingness to Try New Brand",
                        "Number of Skincare Influencers Followed", "Interest in Curated Skincare Box",
                        "Willing to Pay Premium", "One Platform for Shopping Inspiration"]
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    df = df.drop(columns=["Favorite Influencer or Brand", "Purchase Influencers",
                          "Skin Concerns", "Purchase Discouragement Factors",
                          "Skincare Products Used"])  # Drop multiselect columns for classification
    X = df.drop("Purchased After Influencer Promotion", axis=1)
    y = le.fit_transform(df["Purchased After Influencer Promotion"])
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Visualization Tab
if selected_tab == "Data Visualization":
    st.title("📊 Data Visualization")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Gender Distribution")
    fig1 = px.pie(df, names="Gender", title="Gender Distribution")
    st.plotly_chart(fig1)

    st.subheader("Income Distribution")
    fig2 = px.histogram(df, x="Income (AED)", color="Gender", barmode="group")
    st.plotly_chart(fig2)

    st.subheader("Spending vs. Age")
    fig3 = px.scatter(df, x="Age", y="Average Monthly Spend (AED)", color="Gender", trendline="ols")
    st.plotly_chart(fig3)

# Classification Tab
elif selected_tab == "Classification":
    st.title("🤖 Classification Models")
    X_train, X_test, y_train, y_test = preprocess_classification(df)
    model_dict = {
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    selected_model = st.selectbox("Select Model", list(model_dict.keys()))
    model = model_dict[selected_model]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    if st.checkbox("Show Confusion Matrix"):
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig_cm)

    st.subheader("ROC Curve")
    fig_roc, ax = plt.subplots()
    for name, m in model_dict.items():
        m.fit(X_train, y_train)
        y_score = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig_roc)
