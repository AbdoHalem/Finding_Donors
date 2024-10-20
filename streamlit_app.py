from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sb
import joblib
from sklearn.cluster import KMeans

st.title('🤖 Machine Learning App')
st.info('This is app predicts the annual salary of a person whether is more than 50K dollars or less !')

with st.expander("Data"):
    st.write('**Raw Data**')
    data = pd.read_csv("census.csv")
    data

with st.expander("Data Visualization and analysis"):
    st.image('output.png', caption='Relations between each column and number of people with annual salary > 50K dollars', use_column_width=True)
    st.write("""### These graphs show that
1- People who are older than 30 years old and younger than 50 years old are the most profitable for more than 50 thousand annually

2- People who have his private buisness are the most profitable for more than 50 thousand annually

3- People whose education level is 'Bachelors' are the most profitable for more than 50 thousand annually, followed by those with 'HS-grade', then those with 'Some-college'

4- People whose education num is '13' are the most profitable for more than 50 thousand annually, followed by those with '9', then those with '10'

5- People who is married are the most profitable for more than 50 thousand annually

6- People with occupation 'Exec-managerial' and 'Prof-Speciality' are the most profitable for more than 50 thousand annually

7- People with race 'white' are the most profitable for more than 50 thousand annually

8- Males are the most profitable for more than 50 thousand annually

9- People with 'capital-gain' 0 are the most profitable for more than 50 thousand annually

10- People with 'capital-loss' 0 are the most profitable for more than 50 thousand annually

11- People who work 40 hours per week are the most profitable for more than 50 thousand annually

12- People whose native-country is 'united-states' are the most profitable for more than 50 thousand annually""")
    
with st.sidebar:
    st.header('Input Features')
    age = st.slider('age (year)', 1, 90)
    workclass = st.selectbox('workclass', data['workclass'].unique())
    education_level = st.selectbox('education_level', data['education_level'].unique())
    education_num = st.slider('education-num (integer)', 1, 20, 10)
    marital_status = st.selectbox('marital-status', data['marital-status'].unique())
    occupation = st.selectbox('occupation', data['occupation'].unique())
    relationship = st.selectbox('relationship', data['relationship'].unique())
    race = st.selectbox('race', data['race'].unique())
    sex = st.selectbox('sex', data['sex'].unique())
    capital_gain = st.slider('capital-gain (dolar)', 0.0, 50000.0)
    capital_loss = st.slider('capital-loss (dolar)', 0.0, 50000.0)
    hours_per_week = st.slider('hours-per-week (hours)', 1, 100)
    native_country = st.selectbox('native-country', data['native-country'].unique())

    # Create a DataFrame for the input features
    features = {'age': age,
            'workclass': workclass,
            'education_level': education_level,
            'education-num': education_num,
            'marital-status': marital_status,
            'occupation': occupation,
            'relationship': relationship,
            'race': race,
            'sex': sex,
            'capital-gain': capital_gain,
            'capital-loss': capital_loss,
            'hours-per-week': hours_per_week,
            'native-country': native_country
            }
    input_df = pd.DataFrame(features, index=[0])

with st.expander('Input features'):
    input_df

# Load the model and preprocessing objects
model = joblib.load('charity_model.pkl')
encoder = joblib.load('encoder.pkl')
normalizer = joblib.load('normalizer.pkl')

def predection_fun(row, model, encoder, normalizer):
    # Add predicted row to the original dataframe
    original_data = pd.read_csv("census.csv")
    original_data = pd.concat([original_data, row], ignore_index=True)

    # First fill nan value in 'income' column of the last row
    original_data['income'] = original_data['income'].fillna("<=50K").astype(str)
    # Convert income to numerical values
    original_data.loc[original_data['income'] == "<=50K", 'income'] = 0
    original_data.loc[original_data['income'] == ">50K", 'income'] = 1
    original_data['income'] = original_data['income'].astype(int)

    # Get the mean of numerical columns
    numerical_cols = original_data.select_dtypes(include= ['float64', 'int64']).copy()
    mean_col = original_data[numerical_cols.columns].mean()
    null_columns = original_data[numerical_cols.columns]
    null_columns = null_columns.columns[null_columns.isnull().sum() > 0]

    # Fill nan values with the mean of its column
    original_data[null_columns] = original_data[null_columns].fillna(mean_col[null_columns])
    # Drop nan values in categorical data
    catg_col = original_data.drop(numerical_cols.columns, axis=1).columns
    original_data = original_data.dropna(subset = catg_col)

    # Apply log transform for skewed data as preprocessing
    skewed_cols = abs(numerical_cols.skew() > 0.5)
    skewed_cols = skewed_cols[skewed_cols == True].index
    original_data[skewed_cols] = np.log1p(original_data[skewed_cols])

    # Normalize the numerical data    
    original_data[numerical_cols.columns] = normalizer.transform(numerical_cols)

    # Convert categorical data to numerical
    categorical_cols = original_data.drop(numerical_cols.columns, axis=1).columns
    feature_coded = encoder.transform(original_data[categorical_cols])
    encoded_df = pd.DataFrame(feature_coded, columns=encoder.get_feature_names(categorical_cols), index=original_data.index)
    original_data = pd.concat([original_data[numerical_cols.columns], encoded_df], axis=1)

    # Data Clustering: First remove 'income' column
    if 'income' in original_data.columns:
        original_data.drop('income', axis=1, inplace=True)

    # Check for NaNs again before clustering
    if original_data.isnull().values.any():
        raise ValueError("Data still contains NaN values")
    
    kmeans = KMeans(n_clusters = 7, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(original_data)
    cluster = pd.Series(data=y_kmeans)
    cluster.index = original_data.index
    original_data["Cluster"] = cluster
    row_processed = original_data.iloc[original_data.shape[0]-1, :].to_frame().T    # Convert Series to DataFrame
    output = model.predict(row_processed)
    return output

# Make predictions using the refactored function
if st.button('Predict'):
    prediction = predection_fun(input_df, model, encoder, normalizer)
    if(prediction == 1):
        st.write("# This person's annual salary is expected to be higher than 50k$")
    else:
        st.write("# This person's annual salary is expected to be less than 50k$")


    
