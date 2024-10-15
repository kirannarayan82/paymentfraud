import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import numpy as np

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv('transactions.csv')
    return data

data = load_data()

# Display the dataset
st.title('Fraud Detection in Payments')
st.write('Dataset:')
st.write(data.head())

# Feature selection
features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
data=data.iloc[0:10,:]
X = data[features]

# Isolation Forest
st.subheader('Isolation Forest')
if st.button('Run Isolation Forest'):
    iso_forest = IsolationForest(contamination=0.01)
    data['anomaly_iso'] = iso_forest.fit_predict(X)
    # Display the results
    st.subheader('Anomalies Detected')
    st.write(data[data['anomaly_iso'] == -1])

# KMeans Clustering
st.subheader('KMeans Clustering')
if st.button('Run KMeans'):
    kmeans = KMeans(n_clusters=2)
    data['cluster'] = kmeans.fit_predict(X)
    # Display the results
    st.subheader('Anomalies Detected')
    st.write(data[data['cluster']==1])

# Interquartile Range (IQR)
st.subheader('Interquartile Range (IQR)')
if st.button('Run IQR'):
    Q1 = data[features].quantile(0.25)
    Q3 = data[features].quantile(0.75)
    IQR = Q3 - Q1
    kmeans = KMeans(n_clusters=2)
    data['anomaly_iqr'] = ((data[features] < (Q1 - 1.5 * IQR)) | (data[features] > (Q3 + 1.5 * IQR))).any(axis=1)
    st.write(data[data['anomaly_iqr']])


