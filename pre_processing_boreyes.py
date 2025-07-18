import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import plot_importance

#upload data set
from google.colab import files
uploaded = files.upload()

#tampilkan spesifikasi data set
for fn in uploaded.keys():
  print('File yang terupload adalah "{name}" dengan ukuran {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

df_initial = pd.read_csv("DATASET INITIAL - 1 (1).csv")
df_initial.head()

df_cleaned = df_initial.drop(columns=['HH:MM:SS'])
df_cleaned = df_cleaned[(df_initial['ROP'] != 0)]
df_cleaned = df_cleaned.dropna(subset=['ROP'])
df_cleaned.head()

columns_with_na = df_cleaned.columns[df_cleaned.isnull().any()]

for column in columns_with_na:
    df_missing = df_cleaned[df_cleaned[column].isnull()]
    df_non_missing = df_cleaned.dropna(subset=[column])

    X_train = df_non_missing.drop(column, axis=1)
    y_train = df_non_missing[column]

    model = ExtraTreesRegressor()
    model.fit(X_train, y_train)

    X_missing = df_missing.drop(column, axis=1)
    y_pred = model.predict(X_missing)

    df_cleaned.loc[df_cleaned[column].isnull(), column] = y_pred

df_final=df_cleaned
df_final.head()

df_cleaned.describe().T

column_names = df_final.columns.tolist()

describe_df=df_final.describe().round(2)
descriptive_df = pd.DataFrame(describe_df, columns=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])

describe_df=describe_df.T
describe_df['Parameter'] = column_names

describe_df = describe_df[['Parameter', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]

final_output_path = "DESCRIBE FINAL.csv"
describe_df.to_csv(final_output_path, index=False)

final_output_path
files.download("DESCRIBE FINAL.csv")

iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers = iso_forest.fit_predict(df_cleaned)

df_final_outlier = df_final[outliers == 1]

plt.figure(figsize=(20, 12))
sns.heatmap(df_final_outlier.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap Correlation in Drilling Dataset", fontsize=16, fontweight='bold')
plt.show()

X = df_final_outlier.drop(columns=['ROP'])
X = X.values #konversi ke numpy array

Y1 = df_final_outlier['ROP']
Y1 = Y1.values #konversi ke numpy array

split_index = int(0.8 * len(df_final))  # 80% train, 20% test
X_train_Y1, X_test_Y1 = X[:split_index], X[split_index:]
Y1_train, Y1_test = Y1[:split_index], Y1[split_index:]

print("Split Dataset Klaster PTEI")
print(f"X_train: {X_train_Y1.shape}, X_test: {X_test_Y1.shape}")
print(f"Y1_train: {Y1_train.shape}, Y1_test: {Y1_test.shape}")

X = df_well_outlier.drop(columns=['ROP'])
X = X.values #konversi ke numpy array

Y1 = df_well_outlier['ROP']
Y1 = Y1.values #konversi ke numpy array

split_index = int(0.8 * len(df_well))  # 80% train, 20% test
X_train_Y1, X_test_Y1 = X[:split_index], X[split_index:]
Y1_train, Y1_test = Y1[:split_index], Y1[split_index:]

print("Split Dataset Klaster PTEI")
print(f"X_train: {X_train_Y1.shape}, X_test: {X_test_Y1.shape}")
print(f"Y1_train: {Y1_train.shape}, Y1_test: {Y1_test.shape}")

model = ExtraTreesRegressor(random_state=42)
model.fit(X_train_Y1, Y1_train)

importances = model.feature_importances_

features = df_final.drop(columns=['ROP']).columns

feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Variable Importance')
plt.ylabel('Features')
plt.title('Feature Importance From Extra Tree')
plt.gca().invert_yaxis()
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')
plt.tight_layout()
plt.show()

model_lgb = lgb.LGBMRegressor(random_state=42)
model_lgb.fit(X_train_Y1, Y1_train)

importances = model_lgb.feature_importances_
features = df_final.drop(columns=['ROP']).columns

feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances from LGBoost')
plt.gca().invert_yaxis()
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')
plt.tight_layout()
plt.show()

df_final.describe().T

features = df_final.iloc[:, :-1]  # All columns except the last one
target = df_final.iloc[:, -1]   # The last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Apply IsolationForest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
X_train_i = pd.DataFrame(iso_forest.fit_predict(X_train), index=X_train.index, columns=['outlier'])
X_test_i = pd.DataFrame(iso_forest.predict(X_test), index=X_test.index, columns=['outlier'])

# Filter inliers
X_train_inliers = X_train[X_train_i['outlier'] != -1]
X_test_inliers = X_test[X_test_i['outlier'] != -1]

y_train = y_train.loc[X_train_inliers.index]
y_test = y_test.loc[X_test_inliers.index]

# Visualize boxplots for each feature before and after outlier removal
num_features = len(features.columns)

# Create a grid of subplots dynamically
num_cols = 2  # Since we are showing before and after, we keep it as 2 columns
num_rows = int(np.ceil(num_features / num_cols))  # Adjust the number of rows dynamically

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 25))

# Flatten the axes array for easier indexing
axes = axes.flatten()

plt.suptitle('Boxplots Before and After Outlier Removal', fontsize=18, weight='bold', y=1)

for i, feature in enumerate(features.columns):
    # Ensure not to access beyond available axes
  if 2 * i + 1 < len(axes): # Changed condition to check for both before and after plots
        ax_before = axes[2 * i]  # For 'before' plot
        ax_after = axes[2 * i + 1]  # For 'after' plot

        # Before outlier removal
        sns.boxplot(x=X_train[feature], ax=ax_before)
        ax_before.set_title(f'Before Outlier Removal - {feature}', fontsize=14, weight='bold')
        ax_before.set_xlabel(feature, fontsize=12, weight='bold')
        ax_before.set_ylabel('Values', fontsize=12, weight='bold')

        # After outlier removal
        sns.boxplot(x=X_train_inliers[feature], ax=ax_after)
        ax_after.set_title(f'After Outlier Removal - {feature}', fontsize=14, weight='bold')
        ax_after.set_xlabel(feature, fontsize=12, weight='bold')
        ax_after.set_ylabel('Values', fontsize=12, weight='bold')

# Adjust layout and show plot
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()

# Select the last 16 features (excluding the target column)
features = df_final.iloc[:, -2:-1]  # Last 16 features (excluding the target)
target = df_final.iloc[:, -1]        # The last column is the target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Apply IsolationForest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
X_train_i = pd.DataFrame(iso_forest.fit_predict(X_train), index=X_train.index, columns=['outlier'])
X_test_i = pd.DataFrame(iso_forest.predict(X_test), index=X_test.index, columns=['outlier'])

# Filter inliers
X_train_inliers = X_train[X_train_i['outlier'] != -1]
X_test_inliers = X_test[X_test_i['outlier'] != -1]

y_train = y_train.loc[X_train_inliers.index]
y_test = y_test.loc[X_test_inliers.index]

# Visualize boxplots for each feature before and after outlier removal
num_features = len(features.columns)

# Calculate the number of rows and columns needed for subplots
num_cols = 2  # Since we are showing before and after, we keep it as 2 columns
num_rows = int(np.ceil(num_features / num_cols))  # Adjust the number of rows dynamically

# Create subplots with enough space for all features
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 3))

# Flatten the axes array for easier indexing
axes = axes.flatten()

plt.suptitle('Boxplots Before and After Outlier Removal', fontsize=18, weight='bold', y=1)

# Create a subplot for each feature
for i, feature in enumerate(features.columns):
    if 2 * i + 1 < len(axes):  # Check if there are enough axes for both 'before' and 'after' plots
        ax_before = axes[2 * i]  # Ensure that we do not access beyond the number of axes available
        ax_before = axes[2 * i]  # For 'before' plot
        ax_after = axes[2 * i + 1]  # For 'after' plot

        # Before outlier removal
        sns.boxplot(x=X_train[feature], ax=ax_before)
        ax_before.set_title(f'Before Outlier Removal - {feature}', fontsize=14, weight='bold')
        ax_before.set_xlabel(feature, fontsize=12, weight='bold')
        ax_before.set_ylabel('Values', fontsize=12, weight='bold')

        # After outlier removal
        sns.boxplot(x=X_train_inliers[feature], ax=ax_after)
        ax_after.set_title(f'After Outlier Removal - {feature}', fontsize=14, weight='bold')
        ax_after.set_xlabel(feature, fontsize=12, weight='bold')
        ax_after.set_ylabel('Values', fontsize=12, weight='bold')

# Adjust layout and show plot
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()

final_output_path = "DATASET FINAL.csv"
df_final.to_csv(final_output_path, index=False)

final_output_path
files.download("DATASET FINAL.csv")

final_output_path = "DATASET WELL.csv"
df_well_time.to_csv(final_output_path, index=False)

final_output_path
files.download("DATASET WELL.csv")
