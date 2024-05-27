import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import lightgbm as lgb
import warnings
from itertools import combinations
import plotly.express as px
from matplotlib.pyplot import figure
from skimpy import skim
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy
from sklearn.cluster import DBSCAN
from itertools import product
from sklearn.metrics import silhouette_score
import warnings
# Disable all warnings
warnings.filterwarnings ('ignore')
df = pd.read_csv('Mall_Customers.csv')
df.head()
skim(df)
df.info()
df.columns
df.describe()
df.isnull().sum()
import missingno as msno
msno.matrix(df, figsize=(10, 6), sparkline=False, color=(0.24, 0.77, 0.77))
plt.title('Null Values Heatmap', fontsize=16)
plt.show()
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
fig, axes = plt.subplots(nrows=len(num_cols), ncols=1, figsize=(12, 8 * len(num_cols)))
colors = sns.color_palette("viridis", n_colors=len(num_cols))
for i, (col, color) in enumerate(zip(num_cols, colors)):
    sns.histplot(data=df, x=col, kde=True, color=color, ax=axes[i], bins=30)
    axes[i].set_title(f'Histogram for {col}', fontsize=16, color=color)
    axes[i].set_xlabel(col, fontsize=14, color=color)
    axes[i].set_ylabel('Frequency', fontsize=14, color=color)
    axes[i].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
def create_countplots(df):
    obj_cols = df.select_dtypes(include=['object']).columns
    
    fig, axes = plt.subplots(nrows=len(obj_cols), ncols=1, figsize=(12, 8 * len(obj_cols)))

    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for i, col in enumerate(obj_cols):
        sns.set_theme(style="whitegrid")
        ax = sns.countplot(data=df, x=col, palette="Set3", ax=axes[i])
        ax.set_title(f'Countplot for {col}', fontsize=16)
        ax.set_xlabel(col, fontsize=14)
        ax.set_ylabel('Count', fontsize=14)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=12)

        total_count = len(df)
        for p in ax.patches:
            percent = (p.get_height() / total_count) * 100
            ax.annotate(f'{percent:.2f}%', (p.get_x() + p.get_width() / 2, p.get_height() + 5),
                        ha='center', fontsize=12, color='black')

    plt.tight_layout()
    plt.show()

create_countplots(df)
def create_boxplots(df, categorical_column):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    fig, axes = plt.subplots(nrows=len(num_cols), ncols=1, figsize=(12, 8 * len(num_cols)))
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for i, col in enumerate(num_cols):
        sns.set_theme(style="whitegrid")  # Set Seaborn theme for better aesthetics
        ax = sns.boxplot(data=df, y=col, x=categorical_column, ax=axes[i], palette="Set3")
        ax.set_title(f'Boxplot for {col} grouped by {categorical_column}', fontsize=16)
        ax.set_xlabel(categorical_column, fontsize=14)
        ax.set_ylabel(col, fontsize=14)
        sns.swarmplot(data=df, x=categorical_column, y=col, color=".25", ax=axes[i])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=12)
    plt.tight_layout()
    plt.suptitle(f'Advanced Boxplots for Numerical Columns Grouped by {categorical_column}', fontsize=20)
    plt.show()

create_boxplots(df, "Gender")
def one_hot_encode_columns(df, columns_to_encode):
    """
    Perform one-hot encoding for specified columns in a DataFrame.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - columns_to_encode: list
        List of column names to be one-hot encoded.

    Returns:
    - DataFrame
        The DataFrame with specified columns one-hot encoded.
    """
    df_encoded = pd.get_dummies(df, columns=columns_to_encode, prefix=columns_to_encode)
    df_encoded = df_encoded.astype(int)
    return df_encoded


df2 = one_hot_encode_columns(df, columns_to_encode=['Gender'])
df.info()
df2.head()
df_train=df2.drop('CustomerID',axis=1)
df_train.corr()
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(df_train.corr(), annot=True, cmap='coolwarm', linewidths=.5, fmt=".2f", cbar_kws={"shrink": 0.75})
heatmap.set_title('Correlation Heatmap', fontsize=18)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, fontsize=12)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=12)
plt.show()
def scatter_plots_for_numerical_columns(df, hue_column=None):
    """
    Create scatter plots for each numerical column in the DataFrame.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - hue_column: str or None, optional
        The column to use for coloring points. If None, no coloring is applied.

    Returns:
    - None
        Displays the scatter plots.
    """
    # Get the numerical columns of the DataFrame
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        sns.set_theme(style="whitegrid")  
        plt.figure(figsize=(10, 6))
        scatter = sns.scatterplot(data=df, x=col, y="Annual Income (k$)", hue=hue_column)
        scatter.set_title(f'Scatter Plot for {col}', fontsize=16)
        scatter.set_xlabel(col, fontsize=14)
        scatter.set_ylabel('Annual Income (k$)', fontsize=14)
        if hue_column:
            scatter.legend(title=hue_column, title_fontsize=12)

        plt.show()

scatter_plots_for_numerical_columns(df, hue_column='Gender')
df3 = df.drop(columns=['CustomerID'],axis=1)
df_km=df3.copy(deep=True)
df_km.info()
from sklearn.preprocessing import LabelEncoder
# Create an instance of the LabelEncoder class
le = LabelEncoder()
# Get a list of categorical columns
categorical_cols = df_km.select_dtypes(include='object').columns
# Apply the label encoder to each categorical column
for col in categorical_cols:
    df_km[col] = le.fit_transform(df_km[col])
# select the features
X = df_km
#Scaling Data

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = KMeans(random_state=1)
visualizer = KElbowVisualizer(model, k=(2,10))
visualizer.fit(X)
visualizer.show()
plt.show()
model = KMeans(random_state=1)
visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette')
visualizer.fit(X)
visualizer.show()
plt.show()
ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

# Iterate over different numbers of clusters
for num_clusters in range_n_clusters:
    # Initialize KMeans with the current number of clusters
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50, random_state=42)
    
    # Fit KMeans to the scaled data
    kmeans.fit(X)
    
    # Append the sum of squared distances (inertia) to the list
    ssd.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range_n_clusters, ssd, marker='o')
plt.title('Elbow Curve for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Distances (Inertia)')
plt.show()
# create a k-means object with the optimal number of clusters
optimal_k = 4 # number of clusters where the elbow is
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
# Adding the clusters to the dataframe
df_km['cluster'] = y_kmeans
df_km.head()
sns.scatterplot(x='Spending Score (1-100)', y='Age', hue='cluster', data=df_km, palette="deep")
plt.title('Clusters of Customers')
plt.xlabel('Spending Score')
plt.ylabel('Age')
plt.show()
sns.scatterplot(x='Annual Income (k$)', y='Age', hue='cluster', data=df_km, palette="deep")
plt.title('Clusters of Customers')
plt.xlabel('Annual Income')
plt.ylabel('Age')
plt.show()
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='cluster', data=df_km, palette="deep")
plt.title('Clusters of Customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df_km['Annual Income (k$)'],
            df_km['Gender'],
            df_km['Spending Score (1-100)'],
            c=df_km['cluster'], cmap='viridis', s=50)

ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Gender')
ax.set_zlabel('Spending Score (1-100)')
ax.set_title('3D Scatter Plot with Clusters')
colorbar = plt.colorbar(scatter)
colorbar.set_label('Cluster', rotation=270, labelpad=15)
plt.show()
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df_km['Annual Income (k$)'],
            df_km['Age'],
            df_km['Spending Score (1-100)'],
            c=df_km['cluster'], cmap='viridis', s=50)

ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Age')
ax.set_zlabel('Spending Score (1-100)')
ax.set_title('3D Scatter Plot with Clusters')
colorbar = plt.colorbar(scatter)
colorbar.set_label('Cluster', rotation=270, labelpad=15)
plt.show()
df_km.groupby('cluster').describe()
df4 = df.drop(columns=['CustomerID'],axis=1)
df_hr = df4.copy(deep=True)
df_hr.info()
categorical_cols = df_hr.select_dtypes(include='object').columns
for col in categorical_cols:
    df_hr[col] = le.fit_transform(df_hr[col])
df_hr.info()
model = AgglomerativeClustering(n_clusters=None,distance_threshold=0)
cluster_labels = model.fit_predict(df_hr)
cluster_labels
linkage_matrix = hierarchy.linkage(model.children_)
linkage_matrix[:][:5]
plt.figure(figsize=(30,10))
hierarchy.set_link_color_palette(['r','grey', 'b', 'grey', 'm', 'grey', 'g', 'grey', 'orange']) 
dn = hierarchy.dendrogram(linkage_matrix,truncate_mode='level',p=15, color_threshold=23) 
model = AgglomerativeClustering(n_clusters=4)
df_hr['cluster'] = model.fit_predict(df_hr)
cluster_palette = sns.color_palette('pastel', as_cmap=True)

plt.figure(figsize=(10, 8))
scatter = sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='cluster',
    data=df_hr,
    palette=cluster_palette,
    edgecolor='k',  
    s=100,          
)

plt.title('Clusters of Customers', fontsize=16)
plt.xlabel('Annual Income (k$)', fontsize=14)
plt.ylabel('Spending Score (1-100)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)  
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
sns.scatterplot(x='Spending Score (1-100)', y='Age', hue='cluster', data=df_hr, palette="deep")
plt.title('Clusters of Customers')
plt.xlabel('Spending Score')
plt.ylabel('Age')
plt.show()
sns.scatterplot(x='Annual Income (k$)', y='Age', hue='cluster', data=df_hr, palette="deep")
plt.title('Clusters of Customers')
plt.xlabel('Annual Income')
plt.ylabel('Age')
plt.show()
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='cluster', data=df_hr, palette="deep")
plt.title('Clusters of Customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df_hr['Age'],
           df_hr['Annual Income (k$)'],
           df_hr['Spending Score (1-100)'],
           c=df_hr['cluster'], 
           s=35, edgecolor='k', cmap=plt.cm.Set1)

ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
ax.set_title('3D Scatter Plot with Clusters')
colorbar = plt.colorbar(scatter)
colorbar.set_label('Cluster', rotation=270, labelpad=15)
plt.show()
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df_hr['Annual Income (k$)'],
           df_hr['Gender'],
           df_hr['Spending Score (1-100)'],
           c=df_hr['cluster'], 
           s=35, edgecolor='k', cmap=plt.cm.Set1)

ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Gender')
ax.set_zlabel('Spending Score (1-100)')
ax.set_title('3D Scatter Plot with Clusters')
colorbar = plt.colorbar(scatter)
colorbar.set_label('Cluster', rotation=270, labelpad=15)
plt.show()




