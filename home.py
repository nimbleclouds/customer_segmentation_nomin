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
import plotly.express as px
import math
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
data = data[data['Огноо']>'2024-07-01']
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
basket_avg = data.groupby('Картны дугаар')['Дүн'].mean(numeric_only=True).reset_index()
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
aggregated_df = filtered_data.groupby('Картны дугаар').agg({
    'Нас':'last',
    'Тоо ширхэг': 'sum', 
    'Үнэ': 'median',  
    'Дүн': 'sum',    
    'Картны хувь': 'last',   
    'Хүйс': 'last',
}).reset_index()

rfms = filtered_data.groupby(['Картны дугаар','Сүүлд худалдан авалт хийснээс хойш хоног (R)', 'Худалдан авалтын давтамж (F)', 'Нийт худалдан авалтын дүн (M)']).last().reset_index()[['Картны дугаар','Сүүлд худалдан авалт хийснээс хойш хоног (R)', 'Худалдан авалтын давтамж (F)', 'Нийт худалдан авалтын дүн (M)']]

aggregated_df = aggregated_df.merge(rfms, on='Картны дугаар', how='left')

numerical_cols = ['Тоо ширхэг', 'Үнэ', 'Дүн', 'Нас','Картны хувь', 'Сүүлд худалдан авалт хийснээс хойш хоног (R)', 'Худалдан авалтын давтамж (F)', 'Нийт худалдан авалтын дүн (M)']
categorical_cols = ['Хүйс']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ]
)

fd_pca = preprocessor.fit_transform(aggregated_df)
pca = PCA(random_state=123, svd_solver='full')
pca.fit(fd_pca)
cumsum = np.cumsum(pca.explained_variance_ratio_)
reqd_expl_var = 0.9
reqd_n_comp = np.argmax(cumsum >= reqd_expl_var) + 1


plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(cumsum) + 1), cumsum, linewidth=2, color='blue', linestyle='-', alpha=0.8)
plt.xlabel("Принципал Компонент Тоо")
plt.ylabel("Нийлбэр Тайлбарлагдсан Дисперси ")
plt.title("Нийлбэр Тайлбарлагдсан Дисперси vs. Принципал Компонент Тоо")
plt.plot(reqd_n_comp, cumsum[reqd_n_comp - 1], marker='o', markersize=9, color='red')

def find_flatten_point(cumsum):
    differences = np.diff(cumsum)
    second_differences = np.diff(differences)
    flatten_point_index = np.argmax(second_differences) + 1
    return flatten_point_index

flatten_index = find_flatten_point(cumsum)
x_value_flatten = flatten_index
y_value_flatten = cumsum[flatten_index - 1]

plt.plot(x_value_flatten, y_value_flatten, marker='o', markersize=9, color='green')
plt.annotate(f"{x_value_flatten}", xy=(x_value_flatten, y_value_flatten), xytext=(x_value_flatten, y_value_flatten - 0.15),
arrowprops=dict(arrowstyle="->"), ha="center", color="green", weight="bold")
circle_flatten = plt.Circle((x_value_flatten, y_value_flatten), 0.00, color='green', fill=False)
plt.gca().add_patch(circle_flatten)
x_value = reqd_n_comp
y_value = cumsum[reqd_n_comp - 1]
plt.annotate(f"{x_value}", xy=(x_value, y_value), xytext=(x_value, y_value - 0.15),
arrowprops=dict(arrowstyle="->"), ha="center", color="red", weight="bold")
circle = plt.Circle((x_value, y_value), 0.00, color='red', fill=False)
plt.gca().add_patch(circle)
plt.text(reqd_n_comp + 2, cumsum[reqd_n_comp - 1] - 0.18, "(Сонгогдсон ПК)", ha='right', va='top', color='red')
plt.text(x_value_flatten + 1.6, y_value_flatten - 0.18, "(Хэрэгцээт ПК)", ha='right', va='top', color='green')
plt.xticks(np.arange(1, len(cumsum) + 1, 1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid(True, alpha=0.2)
st.pyplot(plt)


############

pca = PCA(n_components = reqd_n_comp, random_state = 123, svd_solver = 'full')
pca.fit(fd_pca)
pcs = pca.transform(fd_pca)
df_pcs = pd.DataFrame(data=pcs, columns=[f'PC{i+1}' for i in range(reqd_n_comp)])
st.write(f"Цомхотголын дараах өгөгдөлд үлдсэн онцлог чанарын тоо: {len(df_pcs.columns)}")


#############


def compute_scores(data, k_range):
    wcss_scores = []
    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=123)
        kmeans.fit(data)
        wcss_scores.append(kmeans.inertia_)
        kmeans_labels = kmeans.labels_
        if len(set(kmeans_labels)) > 1: # ensuring at least 2 clusters for silhouette score
            silhouette_scores.append(silhouette_score(data, kmeans_labels))
        else:
            silhouette_scores.append(0) # setting silhouette score to 0 if only 1 cluster
    return wcss_scores, silhouette_scores

def find_elbow_point(wcss):
    differences = np.diff(wcss)
    second_differences = np.diff(differences)
    elbow_point_index = np.where(second_differences > 0)[0][0] + 1
    return elbow_point_index

k_range = range(2, 11)
wcss_scores, silhouette_scores = compute_scores(df_pcs, k_range)
elbow_point_index = find_elbow_point(wcss_scores)
best_k_index = np.argmax(silhouette_scores)
best_k = k_range[best_k_index]
plt.style.use("fivethirtyeight")
fig, axes = plt.subplots(1, 2, figsize=(20, 5))
axes[0].plot(k_range, wcss_scores, color='blue')
axes[0].scatter(k_range[elbow_point_index], wcss_scores[elbow_point_index], color='red', marker='o', s=500, label='Elbow Point')
axes[0].set_title('Elbow Аргачлал')
axes[0].set_xlabel('Кластерын тоо')
axes[0].set_ylabel('Квадрат Зөрүүний Нийлбэр (WCSS)')
axes[0].legend()
axes[1].plot(k_range, silhouette_scores, color='blue')
axes[1].scatter(best_k, silhouette_scores[best_k_index], color='red', marker='o', label='Best Silhouette Score', s=500)
axes[1].set_title('Silhouette Аргачлал')
axes[1].set_xlabel('Кластерын тоо')
axes[1].set_ylabel('Silhouette оноо')
axes[1].legend()
plt.tight_layout()
st.pyplot(plt)

plt.style.use("default")

max_score = max(silhouette_scores)
max_index = silhouette_scores.index(max_score)
max_clusters = k_range[max_index]
for i, score in zip(k_range, silhouette_scores):
    st.code(f"{i} кластерт оноогдсон Silhuoette оноо: {round(score, 4)}")
st.code(f"Хамгийн өндөр Silhuoette оноо: {round(max_score, 4)} (ашиглагдсан кластерын тоо: {max_clusters})")

df_with_clusters = aggregated_df.copy()

model_k2 = KMeans(n_clusters= 2, random_state=123)
cluster_labels_k2 = model_k2.fit_predict(df_pcs)
df_with_clusters['Cluster_2'] = cluster_labels_k2

model_k3 = KMeans(n_clusters= 3, random_state=123)
cluster_labels_k3 = model_k3.fit_predict(df_pcs)
df_with_clusters['Cluster_3'] = cluster_labels_k3


pca_2 = PCA(n_components=2)
df_pcs_2 = pca_2.fit_transform(fd_pca)
df_pcs_2 = pd.DataFrame(data=df_pcs_2, columns=['PC1','PC2'])
df_pcs_2['Cluster_2'] = cluster_labels_k2
df_pcs_2['Cluster_3'] = cluster_labels_k3

##############################################3333

def plot_cluster_solution(df_pcs, cluster_labels, centroids, title, custom_colors):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_pcs, x='PC1', y='PC2', hue=cluster_labels, palette=custom_colors, alpha=0.5)
    plt.title(title)
    plt.xlabel("Принципал Компонент 1")
    plt.ylabel("Принципал Компонент 2")
    plt.grid(False)
    plt.tight_layout()
    sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], marker='X', s=250, color='red', label='Centroids')
    plt.legend(title='Кластерын нэр')
    
custom_colors_2 = ['#ffc000', '#0070c0']
plot_cluster_solution(df_pcs_2, 'Cluster_2', model_k2.cluster_centers_,
"2 кластерд суурилсан солюшн", custom_colors_2)
st.pyplot(plt)
custom_colors_3 = ['#ffc000', '#0070c0', '#ff33cc']
plot_cluster_solution(df_pcs_2, 'Cluster_3', model_k3.cluster_centers_,
"3 кластерд суурилсан солюшн", custom_colors_3)
st.pyplot(plt)
df_3d = aggregated_df.loc[:,['Худалдан авалтын давтамж (F)', 'Нас', 'Нийт худалдан авалтын дүн (M)']]
df_3d['Cluster_2'] = cluster_labels_k2
df_3d['Cluster_3'] = cluster_labels_k3

st.subheader("Нас, Давтамж, Дүн-д тулгуурласан кластер тайлбарлах")
def plot_3d_scatter(df, cluster_col, title):
    fig = px.scatter_3d(df, x='Худалдан авалтын давтамж (F)', y='Нас', z='Нийт худалдан авалтын дүн (M)', color=cluster_col)
    fig.update_layout(title=title, scene=dict(xaxis_title='Худалдан авалтын давтамж (F)', yaxis_title='Нас', zaxis_title='Нийт худалдан авалтын дүн (M)'))
    st.plotly_chart(fig)
plot_3d_scatter(df_3d, 'Cluster_2', 'K-means Clustering (Cluster 2)')
plot_3d_scatter(df_3d, 'Cluster_3', 'K-means Clustering (Cluster 3)')


#####################3

st.subheader("Кластерын дистрибюшн")
def plot_freq_pie_chart(df, column_name, pie_colors, font_colors, ax=None):
    value_counts = df[column_name].value_counts()
    proportion = value_counts / len(df)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    patches, texts, autotexts = ax.pie(
        proportion, 
        labels=[f"{label} ({value})" for label, value in value_counts.items()],
        autopct='%1.1f%%', 
        startangle=140,
        colors=[pie_colors.get(label, 'gray') for label in value_counts.index],
        textprops={'fontweight': 'bold'}
    )
    for text, font_color in zip(autotexts, font_colors):
        text.set_color(font_color)
    ax.set_title(f'Frequency of {column_name}', fontsize=16, pad=20)
    ax.axis('equal')
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plot_freq_pie_chart(
    df_with_clusters, 'Cluster_2',
    pie_colors={0: '#ffc000', 1: '#0070c0'}, 
    font_colors=['white', 'black'],
    ax=axes[0]
)
plot_freq_pie_chart(
    df_with_clusters, 'Cluster_3',
    pie_colors={0: '#ffc000', 1: '#0070c0', 2: '#ff33cc'}, 
    font_colors=['black', 'black', 'white'],
    ax=axes[1]
)
plt.tight_layout()
st.pyplot(plt)

###########################
st.subheader("Өгөгдлийн тоон үзүүлэлтүүдийн кластер дундаж")
def cluster_analysis_by_cols(df, cols_for_analysis, cluster_column, colors, suptitle):
    num_cols = 4  
    num_rows = (len(cols_for_analysis) + num_cols - 1) // num_cols
    plt.figure(figsize=(20, 5 * num_rows))
    bar_width = 0.35
    for i, column in enumerate(cols_for_analysis):
        plt.subplot(num_rows, num_cols, i + 1)
        for cluster, color in colors.items():
            cluster_data = df[df[cluster_column] == cluster]
            x_values = np.array([cluster - bar_width / 4, cluster + bar_width / 4])
            y_values = np.array([cluster_data[column].mean()] * 2)
            plt.bar(x_values, y_values, color=color, width=bar_width / 2, label=f'Cluster {cluster}')
        overall_mean = df[column].mean()
        mean_20_above = overall_mean * 1.2
        mean_20_below = overall_mean * 0.8
        plt.axhline(y=overall_mean, color='black', linestyle='-')
        plt.text(0.5, overall_mean, f'Overall Mean: {overall_mean:.2f}', color='black',
                 fontsize=9, fontweight='bold', ha='center', va='bottom')
        plt.axhline(y=mean_20_above, color='green', linestyle='-')
        plt.text(0.5, mean_20_above, f'20% Above Mean: {mean_20_above:.2f}', color='green',
                 fontsize=9, fontweight='bold', ha='center', va='bottom')
        plt.axhline(y=mean_20_below, color='red', linestyle='-')
        plt.text(0.5, mean_20_below, f'20% Below Mean: {mean_20_below:.2f}', color='red',
                 fontsize=9, fontweight='bold', ha='center', va='bottom')
        plt.title(column, fontweight='bold')
        plt.xlabel('Cluster')
        plt.ylabel(f'Mean {column}')
        plt.xticks(list(colors.keys()), list(colors.keys()))  # Set x-axis labels to cluster values
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle(suptitle, fontsize=20, fontweight='bold')
    st.pyplot(plt)


st.subheader("2 кластер солюшн задаргаа")
reqd_cols = df_with_clusters.drop(columns='Хүйс').columns
reqd_cols = [item for item in reqd_cols if item not in ['Cluster_2', 'Cluster_3']]
cluster_analysis_by_cols(df = df_with_clusters.drop(columns='Хүйс'),
                         cols_for_analysis = reqd_cols,
                         cluster_column = 'Cluster_2',
                         colors = {0: '#ffc000', 1: '#0070c0'},
                         suptitle = None)

st.subheader("3 кластер солюшн задаргаа")
cluster_analysis_by_cols(df = df_with_clusters,
                         cols_for_analysis = reqd_cols,
                         cluster_column = 'Cluster_3',
                         colors = {0: '#ffc000', 1: '#0070c0', 2: '#ff33cc'},
                         suptitle = None)
st.divider()
st.subheader("Үндсэн датасет")
st.write(df_with_clusters)
num_users = df_with_clusters['Картны дугаар'].nunique()  # Count of unique Картны дугаар
total_spent = df_with_clusters['Дүн'].sum()  # Sum of Дүн
total_items = df_with_clusters['Тоо ширхэг'].sum()  # Sum of Тоо ширхэг
col1, col2, col3 = st.columns(3)
col1.metric("Нийт хэрэглэгчдийн тоо", num_users)
col2.metric("Нийт худалдан авалтын дүн", f"{total_spent:,.2f}")
col3.metric("Нийт худалдан авалтын тоо ширхэг", total_items)
cols_for_dist_plots = [
    'Нас', 'Тоо ширхэг', 'Үнэ', 'Дүн', 'Картны хувь',
    'Сүүлд худалдан авалт хийснээс хойш хоног (R)', 
    'Худалдан авалтын давтамж (F)', 'Нийт худалдан авалтын дүн (M)'
]
num_cols = 3  # You can adjust the number of columns
num_rows = math.ceil(len(cols_for_dist_plots) / num_cols)  # Calculate rows needed
plt.figure(figsize=(20, 5 * num_rows))
plt.subplots_adjust(hspace=0.5, wspace=0.5)
for i, column in enumerate(cols_for_dist_plots, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.histplot(df_with_clusters, x=column, kde=True, bins=20)
    plt.title(f"Distribution of {column}")
st.pyplot(plt)

st.subheader("2 кластер солюшн центройд")
st.write(df_with_clusters.groupby('Cluster_2').mean(numeric_only=True))

st.subheader("3 кластер солюшн центройд")
st.write(df_with_clusters.groupby('Cluster_3').mean(numeric_only=True))
st.divider()

####################################################################################################################
bins1 = pd.cut(filtered_data['Картны хувь'], bins=7, labels=[f'Bin {i+1}' for i in range(7)])
filtered_data['Картны хувь_Bins'] = bins1
bins2 = pd.cut(filtered_data['Нас'], bins=3, labels=['Bin 1', 'Bin 2', 'Bin 3'])
filtered_data['Нас_Bins'] = bins2
bins3 = pd.cut(filtered_data['Нийт худалдан авалтын дүн (M)'], bins=3, labels=['Bin 1', 'Bin 2', 'Bin 3'])
filtered_data['Нийт худалдан авалтын дүн (M)_Bins'] = bins3

metric1 = st.sidebar.selectbox("Бүлэглэх чанар сонгох", ['Картны хувь', 'Нас', 'Нийт худалдан авалтын дүн (M)'])
apply_button = st.button("Apply")

def process_bins(df, metric):
    if metric == 'Картны хувь':
        bin_column = 'Картны хувь_Bins'
    elif metric == 'Нас':
        bin_column = 'Нас_Bins'
    elif metric == 'Нийт худалдан авалтын дүн (M)':
        bin_column = 'Нийт худалдан авалтын дүн (M)_Bins'
    unique_bins = df[bin_column].unique()
    num_bins = len(unique_bins)
    columns = st.columns(num_bins)
    for idx, bin_value in enumerate(unique_bins):
        with columns[idx]:  # Assigning a column for each bin
            st.subheader(f"Analysis for {bin_value} - {metric}")
            df_bin = df[df[bin_column] == bin_value]
            aggregated_df1 = df_bin.groupby('Картны дугаар').agg({ 
                'Нас': 'last',
                'Тоо ширхэг': 'sum', 
                'Үнэ': 'median',  
                'Дүн': 'sum',    
                'Картны хувь': 'last',   
                'Хүйс': 'last',
            }).reset_index()
            rfms1 = df_bin.groupby(['Картны дугаар', 'Сүүлд худалдан авалт хийснээс хойш хоног (R)', 
                                   'Худалдан авалтын давтамж (F)', 'Нийт худалдан авалтын дүн (M)']).last().reset_index()
            rfms1 = rfms1[['Картны дугаар', 'Сүүлд худалдан авалт хийснээс хойш хоног (R)', 
                         'Худалдан авалтын давтамж (F)', 'Нийт худалдан авалтын дүн (M)']]
            aggregated_df1 = aggregated_df1.merge(rfms1, on='Картны дугаар', how='left')
            numerical_cols = ['Тоо ширхэг', 'Үнэ', 'Дүн', 'Нас', 'Картны хувь', 
                              'Сүүлд худалдан авалт хийснээс хойш хоног (R)', 
                              'Худалдан авалтын давтамж (F)', 'Нийт худалдан авалтын дүн (M)']
            categorical_cols = ['Хүйс']
            preprocessor = ColumnTransformer(
                transformers=[('num', StandardScaler(), numerical_cols),
                             ('cat', OneHotEncoder(), categorical_cols)]
            )
            fd_pca = preprocessor.fit_transform(aggregated_df1)
            xpca = PCA(random_state=123, svd_solver='full')
            xpca.fit(fd_pca)  # Fit PCA
            cumsum = np.cumsum(xpca.explained_variance_ratio_)
            reqd_expl_var = 0.9
            reqd_n_comp = np.argmax(cumsum >= reqd_expl_var) + 1  # Number of components that explain 90% variance
            xpca = PCA(n_components=reqd_n_comp, random_state=123, svd_solver='full')
            pcs = xpca.fit_transform(fd_pca)  # Fit and transform
            df_pcs = pd.DataFrame(data=pcs, columns=[f'PC{i+1}' for i in range(reqd_n_comp)])
            st.write(f"Цомхотголын дараах өгөгдөлд үлдсэн онцлог чанарын тоо: {len(df_pcs.columns)}")
            kmeans = KMeans(n_clusters=3, random_state=123)
            kmeans_labels = kmeans.fit_predict(df_pcs)
            aggregated_df1['Cluster'] = kmeans_labels
            pca_2 = PCA(n_components=2)
            df_pcs_2 = pca_2.fit_transform(fd_pca)  # Using the transformed fd_pca for 2D visualization
            df_pcs_2 = pd.DataFrame(data=df_pcs_2, columns=['PC1', 'PC2'])
            df_pcs_2['Cluster'] = kmeans_labels
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df_pcs_2, x='PC1', y='PC2', hue='Cluster', palette='Set2')
            st.pyplot(plt)
            st.write(aggregated_df1)
            st.write(aggregated_df1.groupby('Cluster').mean(numeric_only=True))

if apply_button:
    process_bins(filtered_data, metric1)
