{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Streamlit App: Customer Segmentation Explorer\
# Developed for academic use - simple, intuitive, and practical.\
\
import streamlit as st\
import pandas as pd\
import joblib\
from sklearn.preprocessing import StandardScaler\
\
# Load trained ML model, scaler, and cleaned dataset\
# These files must be saved in the same directory as this script\
model = joblib.load("rf_classifier.pkl")  # Random Forest or best performing classifier\
scaler = joblib.load("scaler.pkl")        # The same scaler used to standardize RFM values\
df = pd.read_csv("cleaned_rfm_data.csv")  # Dataset containing CustomerID, Recency, Frequency, Monetary\
\
# Streamlit app settings\
st.set_page_config(page_title="Customer Segment Explorer", layout="centered")\
st.title("Customer Segment Prediction & Lookup")\
\
st.write("""\
Use this tool to either:\
- Look up an existing customer by ID and view their RFM stats and predicted segment\
- Or manually input new customer RFM values to predict their segment\
""")\
\
# Option for the user to choose mode\
option = st.radio("Choose an option:", ["Lookup by CustomerID", "Predict Segment Manually"])\
\
# Option 1: Lookup by CustomerID\
if option == " Lookup by CustomerID":\
    customer_id = st.number_input("Enter CustomerID:", min_value=0)\
    if st.button("Fetch Details"):\
        if customer_id in df['CustomerID'].values:\
            # Filter the record for the entered customer\
            customer_data = df[df['CustomerID'] == customer_id][['Recency', 'Frequency', 'Monetary']]\
            st.write("### RFM Values for Customer:")\
            st.dataframe(customer_data)\
\
            # Scale and predict the segment\
            scaled_input = scaler.transform(customer_data)\
            prediction = model.predict(scaled_input)\
\
            st.success(f"Predicted Segment: \{prediction[0]\}")\
\
            # Explain the meaning of the predicted segment\
            st.markdown("**Segment Description:**")\
            if prediction[0] == 0:\
                st.info("Segment 0: Inactive or low-value customers")\
            elif prediction[0] == 1:\
                st.info("Segment 1: Frequent shoppers with lower spending")\
            elif prediction[0] == 2:\
                st.info("Segment 2: Loyal, high-frequency buyers")\
            elif prediction[0] == 3:\
                st.info("Segment 3: High-value and highly engaged customers")\
            else:\
                st.warning("Segment not recognized.")\
        else:\
            st.error("CustomerID not found in the dataset. Please try a different ID.")\
\
# Option 2: Manual RFM Input\
elif option == " Predict Segment Manually":\
    # Simple form inputs for each of the RFM scores\
    recency = st.number_input("Recency (days since last purchase):", min_value=0)\
    frequency = st.number_input("Frequency (number of purchases):", min_value=0)\
    monetary = st.number_input("Monetary (total amount spent):", min_value=0.0, format="%.2f")\
\
    if st.button("Predict Segment"):\
        # Create a single-row DataFrame with the user input\
        input_data = pd.DataFrame([[recency, frequency, monetary]], columns=["Recency", "Frequency", "Monetary"])\
\
        # Apply scaling as done during training\
        scaled_input = scaler.transform(input_data)\
        prediction = model.predict(scaled_input)\
\
        st.success(f"Predicted Segment: \{prediction[0]\}")\
\
        st.markdown("**Segment Description:**")\
        if prediction[0] == 0:\
            st.info("Segment 0: Inactive or low-value customers")\
        elif prediction[0] == 1:\
            st.info("Segment 1: Frequent shoppers with lower spending")\
        elif prediction[0] == 2:\
            st.info("Segment 2: Loyal, high-frequency buyers")\
        elif prediction[0] == 3:\
            st.info("Segment 3: High-value and highly engaged customers")\
        else:\
            st.warning("Segment not recognized.")\
\
# Simple footer with author note\
st.markdown("""\
---\
 Developed by Gunjan Tanwar\
""")\
}