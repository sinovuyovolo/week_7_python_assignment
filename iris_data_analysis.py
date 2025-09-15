# -------------------------
# Iris Data Analysis Script
# -------------------------

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

# Enable better visuals
sns.set(style="whitegrid")

# Task 1: Load and Explore the Dataset

try:
    # Load the Iris dataset from sklearn
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("Dataset loaded successfully!\n")

    # Display the first few rows
    print(" First 5 rows of the dataset:")
    print(data.head())

    # Check data types and missing values
    print("\n Data Info:")
    print(data.info())

    print("\n Missing values in each column:")
    print(data.isnull().sum())

except FileNotFoundError:
    print(" Error: File not found.")
except Exception as e:
    print(f" An unexpected error occurred: {e}")

# Task 1.5: Cleaning the Dataset (no missing values in Iris, but let's ensure that)
data.dropna(inplace=True)

# Task 2: Basic Data Analysis

# Basic statistics
print("\n Basic Statistics:")
print(data.describe())

# Group by species and calculate mean of each feature
print("\n Mean of features grouped by species:")
grouped = data.groupby('species').mean()
print(grouped)

# Task 3: Data Visualization

# 1. Line chart showing average petal length per species (example of trend)
plt.figure(figsize=(8, 5))
sns.lineplot(data=grouped[['petal length (cm)']].reset_index(), x='species', y='petal length (cm)', marker='o')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Bar chart: average sepal width per species
plt.figure(figsize=(8, 5))
sns.barplot(data=grouped.reset_index(), x='species', y='sepal width (cm)', palette='pastel')
plt.title('Average Sepal Width per Species')
plt.xlabel('Species')
plt.ylabel('Sepal Width (cm)')
plt.tight_layout()
plt.show()

# 3. Histogram: distribution of petal width
plt.figure(figsize=(8, 5))
plt.hist(data['petal width (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Petal Width')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Scatter plot: sepal length vs. petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='deep')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.show()

# Observations
print("\n Observations:")
print("""
- Setosa has distinctly smaller petal length and width compared to other species.
- Versicolor and Virginica are more similar but Virginica has slightly larger petal measurements.
- Sepal length and petal length are positively correlated.
- Petal width has a relatively normal distribution.
""")
