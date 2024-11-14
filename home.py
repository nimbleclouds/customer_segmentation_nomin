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

# Concatenate all chunks into a single DataFrame
data = pd.concat(df_list, ignore_index=True)
data = data[data['Огноо']>'2024-05-31']
# Optionally, you can reset the index after concatenation
data.reset_index(drop=True, inplace=True)

# Ensure that datetime columns are properly converted to datetime format
data['Огноо'] = pd.to_datetime(data['Огноо'])
data['Үүсгэгдсэн огноо'] = pd.to_datetime(data['Үүсгэгдсэн огноо'])

# Add total purchase column
data['Хэрэглэгчийн нийт худалдан авалт'] = data.groupby('Картны дугаар')['Дүн'].transform('sum')

# Standardize Gender (strip whitespace and capitalize)
data['Хүйс'] = data['Хүйс'].str.strip().str.capitalize()

# Move all filters to the sidebar

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

segment_customers = filtered_data[filtered_data['Segment'].isin(segment_selection)]
segment_products = segment_customers['Барааны нэр'].value_counts()
top_products = segment_products.head(10) 
segment_revenue = segment_customers.groupby('Барааны нэр')['Дүн'].sum()
top_revenue_products = segment_revenue.sort_values(ascending=False).head(10)

dx = filtered_data[['Баримтны дугаар', 'Барааны нэр', 'Тоо ширхэг']]
dx.loc[:, 'purchased'] = 1
invoices = dx['Баримтны дугаар'].unique()
products = dx['Барааны нэр'].unique()
user_to_id = {user: idx for idx, user in enumerate(invoices)}
product_to_id = {product: idx for idx, product in enumerate(products)}
row_indices = dx['Баримтны дугаар'].map(user_to_id).values
col_indices = dx['Барааны нэр'].map(product_to_id).values
data = dx['purchased'].values 
user_item_sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(len(invoices), len(products)))
svd = TruncatedSVD(n_components=50)  # You can adjust the number of components
user_embeddings = svd.fit_transform(user_item_sparse_matrix)
product_embeddings = svd.components_.T  # Transpose to get product embeddings
product_similarity = cosine_similarity(product_embeddings)
product_similarity_df = pd.DataFrame(product_similarity, index=products, columns=products)

def recommend_products(product_name, top_n=5):
    similar_products = product_similarity_df[product_name].sort_values(ascending=False)
    similar_products = similar_products[similar_products.index != product_name]
    return similar_products.head(top_n)


selected_product = st.selectbox('Choose a product', options = products)

if st.button('Product selection'):
    recommended_products = recommend_products(selected_product, top_n=5)
    st.write(f'Сонгогдсон бараанд хамгийн өндөр корреляцитай бараанууд:')
    st.write(recommended_products)


st.write(f'Сегментийн хувьд хамгийн өндөр борлуулалттай бараанууд:')
st.write(top_products)
st.write(f'Сегментийн хувьд хамгийн өндөр дүнтэй бараанууд:')
st.write(top_revenue_products)
st.write(filtered_data.sample(50))
