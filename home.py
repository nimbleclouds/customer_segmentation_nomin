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

# Optionally, you can reset the index after concatenation
data.reset_index(drop=True, inplace=True)
data = data.drop(columns='Unnamed: 0')
# Ensure that datetime columns are properly converted to datetime format
data['Огноо'] = pd.to_datetime(data['Огноо'])
data['Үүсгэгдсэн огноо'] = pd.to_datetime(data['Үүсгэгдсэн огноо'])

# Add total purchase column
data['Хэрэглэгчийн нийт худалдан авалт'] = data.groupby('Картны дугаар')['Дүн'].transform('sum')

# Standardize Gender (strip whitespace and capitalize)
data['Хүйс'] = data['Хүйс'].str.strip().str.capitalize()

# Date range for Огноо (Purchase Date)
min_date = data['Огноо'].min().date()  # Convert to datetime.date format
max_date = data['Огноо'].max().date()  # Convert to datetime.date format

start_date, end_date = st.sidebar.slider(
    "Үндсэн Огноо",
    min_value=min_date,
    max_value=max_date,
    value=(min_date,max_date)
)

# Convert the selected date range back to pandas Timestamp for filtering
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Created date range for Үүсгэгдсэн огноо
min_created_date = data['Үүсгэгдсэн огноо'].min().date()
max_created_date = data['Үүсгэгдсэн огноо'].max().date()

start_created, end_created = st.sidebar.slider(
    "Картын Үүсгэгдсэн огноо",
    min_value=min_created_date,
    max_value=max_created_date,
    value=(min_created_date, max_created_date)
)

# Convert the selected created date range back to pandas Timestamp for filtering
start_created = pd.to_datetime(start_created)
end_created = pd.to_datetime(end_created)

# Purchase amount (Хэрэглэгчийн нийт худалдан авалт) range slicer
purchase_min, purchase_max = st.sidebar.slider(
    "Хэрэглэгчийн нийт худалдан авалт",
    min_value=int(data['Хэрэглэгчийн нийт худалдан авалт'].min()),
    max_value=int(data['Хэрэглэгчийн нийт худалдан авалт'].max()),
    value=(int(data['Хэрэглэгчийн нийт худалдан авалт'].min()), int(data['Хэрэглэгчийн нийт худалдан авалт'].max()))
)

# Card rate percentage (Картны хувь) slicer (3-10%)
card_rate_min, card_rate_max = st.sidebar.slider(
    "Картны хувь",
    min_value=3,
    max_value=10,
    value=(3, 10)
)

# Gender (Хүйс) multi-select
genders = st.sidebar.multiselect(
    "Хүйс",
    options=data['Хүйс'].unique(),
    default=data['Хүйс'].unique()
)

# Age range as a slider
age_min = int(data['Нас'].min())
age_max = int(data['Нас'].max())

age_min, age_max = st.sidebar.slider(
    "Нас",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max)
)

# Location (Салбар) multi-select
branches = st.sidebar.multiselect(
    "Салбар",
    options=data['Салбар'].unique(),
    default=data['Салбар'].unique()
)

segment_selection = st.sidebar.multiselect("Сегмент",
                                           options=data['Segment'].unique(),
                                          default=data['Segment'].unique())

filtered_data = data[
    (data['Огноо'] >= start_date) & (data['Огноо'] <= end_date) &
    (data['Үүсгэгдсэн огноо'] >= start_created) & (data['Үүсгэгдсэн огноо'] <= end_created) &
    (data['Хэрэглэгчийн нийт худалдан авалт'] >= purchase_min) & 
    (data['Хэрэглэгчийн нийт худалдан авалт'] <= purchase_max) &
    (data['Картны хувь'] >= card_rate_min) & (data['Картны хувь'] <= card_rate_max) &
    (data['Хүйс'].isin(genders)) &
    (data['Нас'] >= age_min) & (data['Нас'] <= age_max) &
    (data['Салбар'].isin(branches)) &
    (data['Segment'].isin(segment_selection))
]

simi = simi.rename(columns={'Top_10_Similar_Products': 'Санал болгох бараанууд'})
filtered_data = filtered_data.merge(simi, left_on='Барааны нэр', right_on='Product', how='left')

segment_customers = filtered_data[filtered_data['Segment'].isin(segment_selection)]
segment_products = segment_customers['Барааны нэр'].value_counts()
top_products = segment_products.head(10)

segment_revenue = segment_customers.groupby('Барааны нэр')['Дүн'].sum()
top_revenue_products = segment_revenue.sort_values(ascending=False).head(10)

def format_similarity(products):
    if isinstance(products, list):
        return ', '.join(products) if products else 'No similar products'
    return 'No similar products'

top_products_with_similarity = filtered_data[filtered_data['Барааны нэр'].isin(top_products.index)]
top_products_with_similarity['Санал болгох бараанууд'] = top_products_with_similarity['Санал болгох бараанууд'].apply(format_similarity)

top_revenue_products_with_similarity = filtered_data[filtered_data['Барааны нэр'].isin(top_revenue_products.index)]
top_revenue_products_with_similarity['Санал болгох бараанууд'] = top_revenue_products_with_similarity['Санал болгох бараанууд'].apply(format_similarity)

st.write("### Давтамж өндөр бараанууд")
# Displaying the top products and their recommendations as a list
for index, row in top_products_with_similarity[['Барааны нэр', 'Санал болгох бараанууд']].iterrows():
    st.write(f"**{row['Барааны нэр']}**")
    st.write(f"  - **Recommended Products:** {row['Санал болгох бараанууд']}")
    
st.write("### Борлуулалтын дүн өндөр бараанууд")
# Displaying the top revenue products and their recommendations as a list
for index, row in top_revenue_products_with_similarity[['Барааны нэр', 'Санал болгох бараанууд']].iterrows():
    st.write(f"**{row['Барааны нэр']}**")
    st.write(f"  - **Recommended Products:** {row['Санал болгох бараанууд']}")
