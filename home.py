import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
from sklearn.decomposition import PCA
import seaborn as sns
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import streamlit as st
from datetime import date
# Path to the directory where chunks are saved
output_directory = 'data/'

# List all chunk files in the directory
chunk_files = [f for f in os.listdir(output_directory) if f.endswith('.csv')]

# Sort the files if necessary to ensure they are processed in the correct order
chunk_files.sort()  # Optional, but helps if filenames contain a numeric or sequential pattern

# Initialize an empty list to hold the DataFrame chunks
df_list = []

# Loop over each chunk file and read it into a DataFrame
for chunk_file in chunk_files:
    chunk_path = os.path.join(output_directory, chunk_file)
    chunk_df = pd.read_csv(chunk_path)
    df_list.append(chunk_df)  # Add the chunk to the list

simi = pd.read_csv('similarity.csv')
simi = simi.drop(columns='Unnamed: 0')
# Concatenate all chunks into a single DataFrame
data = pd.concat(df_list, ignore_index=True)
data.reset_index(drop=True, inplace=True)
data = data.drop(columns='Unnamed: 0')
data['Огноо'] = pd.to_datetime(data['Огноо'])
data['Үүсгэгдсэн огноо'] = pd.to_datetime(data['Үүсгэгдсэн огноо'])

data['Хэрэглэгчийн нийт худалдан авалт'] = data.groupby('Картны дугаар')['Дүн'].transform('sum')

data['Хүйс'] = data['Хүйс'].str.strip().str.capitalize()

min_date = data['Огноо'].min().date()  # Convert to datetime.date format
max_date = data['Огноо'].max().date()  # Convert to datetime.date format

start_date, end_date = st.sidebar.slider(
    "Үндсэн Огноо",
    min_value=min_date,
    max_value=max_date,
    value=(min_date,max_date)
)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

min_created_date = data['Үүсгэгдсэн огноо'].min().date()
max_created_date = data['Үүсгэгдсэн огноо'].max().date()

start_created, end_created = st.sidebar.slider(
    "Картын Үүсгэгдсэн огноо",
    min_value=min_created_date,
    max_value=max_created_date,
    value=(min_created_date, max_created_date)
)

start_created = pd.to_datetime(start_created)
end_created = pd.to_datetime(end_created)

purchase_min, purchase_max = st.sidebar.slider(
    "Худалдан авалтын дүн",
    min_value=int(data['Хэрэглэгчийн нийт худалдан авалт'].min()),
    max_value=int(data['Хэрэглэгчийн нийт худалдан авалт'].max()),
    value=(int(data['Хэрэглэгчийн нийт худалдан авалт'].min()), int(data['Хэрэглэгчийн нийт худалдан авалт'].max()))
)

card_rate_min, card_rate_max = st.sidebar.slider(
    "Картны хувь",
    min_value=3,
    max_value=10,
    value=(3, 10)
)

genders = st.sidebar.multiselect(
    "Хүйс",
    options=data['Хүйс'].unique(),
    default=data['Хүйс'].unique()
)

age_min = int(data['Нас'].min())
age_max = int(data['Нас'].max())

age_min, age_max = st.sidebar.slider(
    "Нас",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max)
)

branches = st.sidebar.multiselect(
    "Салбар",
    options=data['Салбар'].unique(),
    default=data['Салбар'].unique()
)

segment_selection = st.sidebar.multiselect("Сегмент",
                                           options=data['Segment'].unique(),
                                          default=data['Segment'].unique())

basket_avg = data.groupby('Картны дугаар')['Дүн'].mean().reset_index()
basket_avg = basket_avg.rename(columns={'Дүн': 'Сагсны дундаж'})  # Rename the column for clarity
data = pd.merge(data, basket_avg, on='Картны дугаар', how='left')

bas_min, bas_max = st.sidebar.slider(
    "Сагсны дундаж",
    min_value=float(data['Сагсны дундаж'].min()),
    max_value=float(data['Сагсны дундаж'].max()),
    value=(float(data['Сагсны дундаж'].min()), float(data['Сагсны дундаж'].max()))
)


filtered_data = data[
    (data['Огноо'] >= start_date) & (data['Огноо'] <= end_date) &
    (data['Үүсгэгдсэн огноо'] >= start_created) & (data['Үүсгэгдсэн огноо'] <= end_created) &
    (data['Хэрэглэгчийн нийт худалдан авалт'] >= purchase_min) & 
    (data['Хэрэглэгчийн нийт худалдан авалт'] <= purchase_max) &
    (data['Картны хувь'] >= card_rate_min) & (data['Картны хувь'] <= card_rate_max) &
    (data['Хүйс'].isin(genders)) &
    (data['Нас'] >= age_min) & (data['Нас'] <= age_max) &
    (data['Салбар'].isin(branches)) &
    (data['Segment'].isin(segment_selection)) &
    (data['Сагсны дундаж'] >= bas_min) &
    (data['Сагсны дундаж'] <= bas_max)
]



filtered_data['Барааны нэр'] = filtered_data['Барааны нэр'].str.strip().str.upper()
simi['Product'] = simi['Product'].str.strip().str.upper()

simi = simi.rename(columns={'Top_10_Similar_Products': 'Санал болгох бараанууд'})
filtered_data = filtered_data.merge(simi, left_on='Барааны нэр', right_on='Product', how='left')

st.write(filtered_data.head(10))

