# --- Set-Up ---

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# --- Data Pre-Processing ---
# read in s
s = pd.read_csv("social_media_usage.csv")

# define clean_sm
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

# create ss and drop missing values
ss = pd.DataFrame({
    "sm_li": clean_sm(s["web1h"]),
    "income": np.where(s["income"] > 9, np.nan, s["income"]),
    "educ2": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "par": np.where(s["par"] > 2, np.nan, np.where(s["par"] == 2, 0, s["par"])),
    "marital": np.where(s["marital"] > 6, np.nan, np.where(s["marital"].isin([2,3,4,5,6]), 0, s["marital"])),
    "female": np.where(s["gender"] > 3, np.nan, np.where(s["gender"]==2, 1, 0)),
    "age": np.where(s["age"] > 97, np.nan, s["age"])
})

ss = ss.dropna()

# --- Train-Test Split ---
# Target (y) and feature(s) selection (X)
y = ss["sm_li"]
X = ss[["income", "educ2", "par", "marital", "female", "age"]]

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                    y,
                                                    stratify=y,       
                                                    test_size=0.2,    
                                                    random_state=987) 
# --- Model Training ---
# Initialize algorithm
lr = LogisticRegression(class_weight = "balanced")

# Fit algorithm to training data
lr.fit(X_train, y_train)

# --- Model Predictions and App ---
st.title("LinkedIn Usage Prediction App")

with st.sidebar:
    st.header("LinkedIn Usage Prediction")
    st.markdown(
        """
        ### Instructions:
        1. Use the dropdowns and sliders to input values for demographic features about potential LinkedIn users:
        2. Click the "Predict LinkedIn Usage" button to see the prediction.
        3. The app will display whether the individual is predicted to use LinkedIn and the probability of usage.

        Expand the sidebar for readability if needed.
        """
        )
    
    income_map = {
        1: "Less than $10,000",
        2: "10 to under $20,000",
        3: "20 to under $30,000",
        4: "30 to under $40,000",
        5: "40 to under $50,000",
        6: "50 to under $75,000",
        7: "75 to under $100,000",
        8: "100 to under $150,000",
        9: "$150,000+"}
    
    income_input = st.selectbox("Select Income Level:",
                                options=list(income_map.values()))

    educ_map = {
        1: "Less than high school",
        2: "High school incomplete",
        3: "High school graduate",
        4: "Some college, no degree",
        5: "Two-year associate degree",
        6: "Four-year college degree",
        7: "Some postgraduate schooling",
        8: "Postgraduate or professional degree"}
    
    educ_input = st.selectbox("Select Education Level:",
                               options=list(educ_map.values()))
    
    par_input = st.selectbox("Parental Status (of Children Under 18):",
                             options=["Not a parent", "Parent"])
    
    marital_input = st.selectbox("Marital Status:",
                                 options=["Not married", "Married"])
    
    female_input = st.selectbox("Gender:",
                               options=["Not Female", "Female"])
    
    age_input = st.slider("Age:", min_value=18, max_value=97, value=30)

    income_input = [key for key, val in income_map.items() if val == income_input]
    educ_input = [key for key, val in educ_map.items() if val == educ_input]
    par_input = 0 if par_input == "Not a parent" else 1
    marital_input = 0 if marital_input == "Not married" else 1
    female_input = 0 if female_input == "Not Female" else 1

    if st.button("Predict LinkedIn Usage"):
        input_features = pd.DataFrame([[income_input[0],
                                    educ_input[0],
                                    par_input,
                                    marital_input,
                                    female_input,
                                    age_input]])
        prediction = lr.predict(input_features.values)
        st.write("Predicted LinkedIn Usage:", "Yes" if prediction[0] == 1 else "No")
        prob = (lr.predict_proba(input_features.values)[0][1]) * 100
        st.write(f"Probability of LinkedIn Usage: {prob:.2f} %")

tab1, tab2 = st.tabs(["LinkedIn Usage Analysis", "Model Performance Metrics"])

with tab1:
    st.header("LinkedIn Usage Analysis")
    st.markdown("""
                ###### This section provides visualizations of LinkedIn usage based on demographic features. There are 1,250 samples in the dataset.
                """)
    # Plot overall LinkedIn use %
    ss["user"] = ss["sm_li"].map({0: 'No', 1: 'Yes'})
    st.markdown("""
                ##### Overall LinkedIn Usage (%)
                """)
    st.altair_chart(
        alt.Chart(ss).transform_aggregate(Count='count()', groupby=['user']).transform_joinaggregate(TotalUsers='sum(Count)').\
            transform_calculate(PercentOfTotal='datum.Count / datum.TotalUsers * 100')\
                .mark_bar().encode(
                    x = alt.X('user:N', sort='y', title='User', axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
                    y=alt.Y('PercentOfTotal:Q', title='Percentage (%)'),
                    color=alt.Color('user:N', legend=None, scale=alt.Scale(scheme='paired')),
                    tooltip=[alt.Tooltip('user:N', title='User'), alt.Tooltip('PercentOfTotal:Q', format='.2f', title='%')]))
    
    # Plot LinkedIn use % by Income Level
    ss["income_label"] = ss["income"].map(income_map)
    income_grouped = ss.groupby(["income_label"], as_index = False)["sm_li"].mean()
    income_grouped["sm_li"] = income_grouped["sm_li"] * 100
    income_order = ["Less than $10,000",
                    "10 to under $20,000",
                    "20 to under $30,000",
                    "30 to under $40,000",
                    "40 to under $50,000",
                    "50 to under $75,000",
                    "75 to under $100,000",
                    "100 to under $150,000",
                    "$150,000+"]
    income_bar = alt.Chart(income_grouped).mark_bar().encode(
        x=alt.X('income_label', title = 'Income', sort = income_order, axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
        y=alt.Y('sm_li', title = 'LinkedIn Use by Income (%)'),
        color = alt.Color('income_label', legend=None, scale=alt.Scale(scheme='purpleblue')),
        tooltip=[alt.Tooltip('income_label', title='Income Level'), alt.Tooltip('sm_li', format='.2f', title = 'user %')])
    income_line = alt.Chart(income_grouped).mark_line(color='red').transform_window(
        rolling_mean='mean(sm_li)',
        frame=[-1, 0]).encode(
            x=alt.X('income_label', sort = income_order),
            y='rolling_mean:Q')
    income_chart = income_bar + income_line
    st.markdown("""
                ##### LinkedIn Usage Rate (%) by Income Level
                """)
    st.altair_chart(income_chart)

    # Plot LinkedIn use % by Education Level
    ss["education"] = ss["educ2"].map(educ_map)
    educ_grouped = ss.groupby(["education"], as_index = False)["sm_li"].mean()
    educ_grouped["sm_li"] = educ_grouped["sm_li"] * 100
    educ_order = [
        "Less than high school",
        "High school incomplete",
        "High school graduate",
        "Some college, no degree",
        "Two-year associate degree",
        "Four-year college degree",
        "Some postgraduate schooling",
        "Postgraduate or professional degree"]
    educ_bar = alt.Chart(educ_grouped).mark_bar().encode(
        x=alt.X('education', title = 'Education Level', sort = educ_order, axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
        y=alt.Y('sm_li', title = 'LinkedIn Use by Education Level (%)'),
        color=alt.Color('education', legend = None, scale=alt.Scale(scheme='purpleblue')),
        tooltip=[alt.Tooltip('education', title='Education Level'), alt.Tooltip('sm_li', format='.2f', title = 'user %')])
    educ_line = alt.Chart(educ_grouped).mark_line(color='red').transform_window(
        rolling_mean='mean(sm_li)',
        frame=[-1, 0]).encode(
            x=alt.X('education', sort = educ_order),
            y='rolling_mean:Q')
    educ_chart = educ_bar + educ_line
    st.markdown("""
                ##### LinkedIn Usage Rate (%) by Education Level
                """)
    st.altair_chart(educ_chart)

    # Plot interaction between Income and Education Levels
    st.markdown("""
                ##### LinkedIn Usage Rate (%) by Income × Education Level
                """)
    st.altair_chart(alt.Chart(ss).transform_calculate(
        user_pct = "datum.sm_li * 100").mark_rect().encode(
            x=alt.X("income:O", title="Income Level"),
            y=alt.Y("educ2:O", title="Education Level"),
            color=alt.Color("mean(user_pct):Q",
                            scale=alt.Scale(scheme="purpleblue"),
                            title="Mean LinkedIn Use %"),
            tooltip=[
                alt.Tooltip("income_label:O", title="Income"),
                alt.Tooltip("education:O", title="Education"),
                alt.Tooltip("mean(user_pct):Q", title="use %", format=",.2f"),
                alt.Tooltip("count()", title="N")]).properties(
                    width=400,
                    height=300))
    
    # Plot LinkedIn use by Parental Status
    ss["parent"] = ss["par"].map({0: 'Non-Parent', 1: 'Parent'})
    par_grouped = ss.groupby(["parent"], as_index = False)["sm_li"].mean()
    par_grouped["sm_li"] = par_grouped["sm_li"] * 100
    st.markdown("""
                ##### LinkedIn Use by Non-Parent vs Parent (%)
                """)
    st.altair_chart(alt.Chart(par_grouped).mark_bar().encode(
        x=alt.X('parent', title = 'Parent', axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
        y=alt.Y('sm_li', title = 'LinkedIn Use by Non-Parent vs Parent (%)'),
        color=alt.Color('parent', legend = None, scale=alt.Scale(scheme='paired')),
        tooltip = [alt.Tooltip('parent', title='Parental Status'), alt.Tooltip('sm_li', format='.2f', title = 'user %')]))
    
    # Plot Plot LinkedIn use % by Marital Status
    ss["married"] = ss["marital"].map({0: 'Not Married', 1: 'Married'})
    married_grouped = ss.groupby(["married"], as_index = False)["sm_li"].mean()
    married_grouped["sm_li"] = married_grouped["sm_li"] * 100
    st.markdown("""
                ##### LinkedIn Usage Rate (%) by Marital Status
                """)
    st.altair_chart(alt.Chart(married_grouped).mark_bar().encode(
        x=alt.X('married', title = 'Marital Status', axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
        y=alt.Y('sm_li', title = 'LinkedIn Use by Marital Status (%)'),
        color=alt.Color('married', legend = None, scale=alt.Scale(scheme='paired')),
        tooltip=['married', alt.Tooltip('sm_li', format='.2f', title = 'user %')]))
    
    # Plot LinkedIn use % by Gender
    ss["gender"] = ss["female"].map({0: 'Not Female', 1: 'Female'})
    gender_grouped = ss.groupby(["gender"], as_index = False)["sm_li"].mean()
    gender_grouped["sm_li"] = gender_grouped["sm_li"] * 100
    st.markdown("""
                ##### LinkedIn Usage Rate (%) by Female
                """)
    st.altair_chart(alt.Chart(gender_grouped).mark_bar().encode(
        x=alt.X('gender', title = 'Female', axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
        y=alt.Y('sm_li', title = 'LinkedIn Use by Female (%)'),
        color=alt.Color('gender', legend = None, scale=alt.Scale(scheme='paired')),
        tooltip=['gender', alt.Tooltip('sm_li', format='.2f', title='user %')]))
    
    # Plot LinkedIn use % by Age (Binned)
    bins = [18, 28, 38, 48, 58, 68, 78, 88, 98]
    labels = ['18-27', '28-37', '38-47', '48-57', '58-67', '68-77', '78-87', '88-97']
    ss['age_bin'] = pd.cut(ss['age'], bins=bins, labels=labels, right=False)
    age_binned_grouped = ss.groupby('age_bin', as_index=False)['sm_li'].mean()
    age_binned_grouped['sm_li'] = age_binned_grouped['sm_li'] * 100
    age_bar = alt.Chart(age_binned_grouped).mark_bar().encode(
        x=alt.X('age_bin:O', title='Age Group', sort=labels, axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
        y=alt.Y('sm_li:Q', title='LinkedIn Use by Age Group (%)'),
        color=alt.Color('age_bin:O', legend=None),
        tooltip=['age_bin', alt.Tooltip('sm_li', format='.2f', title = 'user %')])
    age_line = alt.Chart(age_binned_grouped).mark_line(color='red').transform_window(
        rolling_mean='mean(sm_li)',
        frame=[-1, 0]).encode(
            x=alt.X('age_bin:O'),
            y='rolling_mean:Q')
    final_age = age_bar + age_line
    st.markdown("""
                ##### LinkedIn Usage Rate (%) by Age Group
                """)
    st.altair_chart(final_age)

with tab2:
    st.header("Model Performance Metrics")
    st.markdown("""
                ###### This section provides performance metrics of the Logistic Regression model used for predicting LinkedIn usage.
                """)
    y_pred = lr.predict(X_test)
    st.markdown("""
                ##### Model Accuracy:
                """)
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    cm = confusion_matrix(y_test, y_pred, labels = [1,0])
    confusion_df_viz = pd.DataFrame(cm,
            columns=["Predicted LinkedIn User", "Predicted Non-LinkedIn User"],
            index=["Actual LinkedIn User","Actual Non-LinkedIn User"]).style.background_gradient(cmap="Blues")
    st.markdown("""
                ##### Confusion Matrix:
                """)
    st.dataframe(confusion_df_viz)

    confusion_df = pd.DataFrame(cm,
            columns=["Predicted positive", "Predicted negative"],
            index=["Actual positive", "Actual negative"])

    # calculate precision for positive cases (LinkedIn use)
    def precision_positive(dataframe):
        pos_prec_score = dataframe.loc["Actual positive","Predicted positive"] / (
            dataframe.loc["Actual positive","Predicted positive"] + \
                dataframe.loc["Actual negative","Predicted positive"])
        return pos_prec_score

    # calculate precision for negative cases (non-LinkedIn use)
    def precision_negative(dataframe):
        neg_prec_score = dataframe.loc["Actual negative","Predicted negative"] / (
            dataframe.loc["Actual negative","Predicted negative"] + \
                dataframe.loc["Actual positive","Predicted negative"])
        return neg_prec_score
    
    # calculate recall for positive cases (LinkedIn use)
    def recall_positive(dataframe):
        pos_rec_score = dataframe.loc["Actual positive","Predicted positive"] / (
            dataframe.loc["Actual positive","Predicted positive"] + \
                dataframe.loc["Actual positive", "Predicted negative"])
        return pos_rec_score
    
    # calculate recall for negative cases (non-LinkedIn use)
    def recall_negative(dataframe):
        neg_rec_score = dataframe.loc["Actual negative","Predicted negative"] / (
            dataframe.loc["Actual negative","Predicted negative"] + \
                dataframe.loc["Actual negative", "Predicted positive"])
        return neg_rec_score
    
    # calculate f1 for positive cases (LinkedIn use)
    def f1_positive(dataframe):
        pos_prec_score = dataframe.loc["Actual positive","Predicted positive"] / (
            dataframe.loc["Actual positive","Predicted positive"] + \
                dataframe.loc["Actual negative","Predicted positive"])
        pos_rec_score = dataframe.loc["Actual positive","Predicted positive"] / (
            dataframe.loc["Actual positive","Predicted positive"] + \
                dataframe.loc["Actual positive", "Predicted negative"])
        pos_f1 = 2*(pos_prec_score*pos_rec_score)/(pos_prec_score+pos_rec_score)
        return pos_f1
    
    # calculate f1 for negative cases (non-LinkedIn use)
    def f1_negative(dataframe):
        neg_prec_score = dataframe.loc["Actual negative","Predicted negative"] / (
            dataframe.loc["Actual negative","Predicted negative"] + \
                dataframe.loc["Actual positive","Predicted negative"])
        neg_rec_score = dataframe.loc["Actual negative","Predicted negative"] / (
            dataframe.loc["Actual negative","Predicted negative"] + \
                dataframe.loc["Actual negative", "Predicted positive"])
        neg_f1 = 2*(neg_prec_score*neg_rec_score)/(neg_prec_score+neg_rec_score)
        return neg_f1
    
    linkedin_user_metrics = pd.DataFrame({
        "pos/neg": "LinkedIn Users",
        "precision": [round(precision_positive(confusion_df), 2)],
        "recall":    [round(recall_positive(confusion_df), 2)],
        "f1":        [round(f1_positive(confusion_df), 2)]})
    
    non_linkedin_user_metrics = pd.DataFrame({
        "pos/neg": "Non-Users",
        "precision": [round(precision_negative(confusion_df), 2)],
        "recall":    [round(recall_negative(confusion_df), 2)],
        "f1":        [round(f1_negative(confusion_df), 2)]})
    
    model_metrics = pd.concat([linkedin_user_metrics, non_linkedin_user_metrics], ignore_index=True)
    
    metrics_melted = model_metrics.melt( id_vars = "pos/neg", value_vars = ["precision", "recall", "f1"])

    st.text("")
    st.markdown("""
                ##### Model Metrics by LinkedIn User vs Non-User Predictions
                """)
    st.altair_chart(
        alt.Chart(metrics_melted).mark_bar().encode(
            x=alt.X('variable:N', title = None, axis=alt.Axis(labelAngle=-45, labelOverlap=False)), 
            y=alt.Y('value:Q', title = 'Metric Score'),
            column = alt.Column('pos/neg:N', header=alt.Header(title='Model Metrics', titleOrient='bottom')),
            color= alt.Color('variable:N', legend = None, scale=alt.Scale(scheme="purpleblue")),
            tooltip=[alt.Tooltip('variable:N', title='metric'), alt.Tooltip('value:Q', format='.2f', title='score')]).properties(
                width=150))
    
    auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Model AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve on Test Set")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    st.markdown("""
                ##### ROC Curve On Test Set
                """)
    st.pyplot(plt)