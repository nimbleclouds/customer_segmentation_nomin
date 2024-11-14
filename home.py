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
output_directory = 'data/'
chunk_files = [f for f in os.listdir(output_directory) if f.endswith('.csv')]
chunk_files.sort()  # Optional, but helps if filenames contain a numeric or sequential pattern
df_list = []
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
filtered_data = filtered_data.rename(columns={'Segment':'RFM Сегмент',
                                              'Recency':'Сүүлд худалдан авалт хийснээс хойш хоног (R)',
                                              'Frequency':'Худалдан авалтын давтамж (F)',
                                              'Monetary':'Нийт худалдан авалтын дүн (M)'})
filtered_data = filtered_data.drop(columns = ['Product','Хэрэглэгчийн нийт худалдан авалт'])



###########################################################################################################################3



metric = st.selectbox("Choose a metric to bin", ['Картны хувь', 'Нас', 'Хүйс', 'Нийт худалдан авалтын дүн (M)'])
apply_button = st.button("Apply")
def create_bins(filtered_df, metric):
    if metric == 'Картны хувь':
        unique_card_rates = sorted(filtered_df['Картны хувь'].unique())
        if len(unique_card_rates) >= 3 and len(unique_card_rates) <= 10:
            bins = pd.cut(filtered_df['Картны хувь'], bins=7, labels=[f'Bin {i+1}' for i in range(7)])
            filtered_df['Картны хувь_Bins'] = bins
            st.write(filtered_df[['Картны хувь', 'Картны хувь_Bins']])

    elif metric == 'Хүйс':
        bins = pd.cut(filtered_df['Хүйс'], bins=2, labels=['Male', 'Female'])
        filtered_df['Хүйс_Bins'] = bins
        st.write(filtered_df[['Хүйс', 'Хүйс_Bins']])

    elif metric == 'Нас':
        min_Нас = filtered_df['Нас'].min()
        max_Нас = filtered_df['Нас'].max()
        Нас_bins = [min_Нас, (max_Нас - min_Нас) / 3 + min_Нас, (max_Нас - min_Нас) * 2 / 3 + min_Нас, max_Нас]
        Нас_labels = ['Bin 1', 'Bin 2', 'Bin 3']
        bins = pd.cut(filtered_df['Нас'], bins=Нас_bins, labels=Нас_labels)
        filtered_df['Нас_Bins'] = bins
        st.write(filtered_df[['Нас', 'Нас_Bins']])

    elif metric == 'Нийт худалдан авалтын дүн (M)':
        min_amount = filtered_df['Нийт худалдан авалтын дүн (M)'].min()
        max_amount = filtered_df['Нийт худалдан авалтын дүн (M)'].max()
        amount_bins = [min_amount, (max_amount - min_amount) / 3 + min_amount, (max_amount - min_amount) * 2 / 3 + min_amount, max_amount]
        amount_labels = ['Bin 1', 'Bin 2', 'Bin 3']
        bins = pd.cut(filtered_df['Нийт худалдан авалтын дүн (M)'], bins=amount_bins, labels=amount_labels)
        filtered_df['Нийт худалдан авалтын дүн (M)_Bins'] = bins
        st.write(filtered_df[['Нийт худалдан авалтын дүн (M)', 'Нийт худалдан авалтын дүн (M)_Bins']])
    return filtered_df
    
if apply_button:
    updated_filtered_df = create_bins(filtered_df, metric)
    st.write(updated_filtered_df.sample(10))
st.write(filtered_data.sample(10))

