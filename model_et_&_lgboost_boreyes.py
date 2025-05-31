import math
import optuna
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from google.colab import files
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from matplotlib.ticker import FixedLocator
from sklearn.preprocessing import RobustScaler
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import plot_importance
from keras.callbacks import EarlyStopping

#upload data set
from google.colab import files
uploaded = files.upload()

#tampilkan spesifikasi data set
for fn in uploaded.keys():
  print('File yang terupload adalah "{name}" dengan ukuran {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

df_egs_initial = pd.read_csv("DATASET FINAL (6).csv")
df_egs_initial.head()

iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers = iso_forest.fit_predict(df_egs_initial)

df_egs_initial = df_egs_initial[outliers == 1]

"""**EXTRA TREES MODELING**"""

df_egs_extra=df_egs_initial.drop(columns=['OVP', 'DEX', 'DAS', 'TOF', 'TMU'])
df_egs_extra.head()

X = df_egs_extra.drop(columns=['ROP'])
X = X.values #konversi ke numpy array

Y1 = df_egs_extra['ROP']
Y1 = Y1.values #konversi ke numpy array

X_train_Y1, X_test_Y1, Y1_train, Y1_test = train_test_split(X, Y1, test_size=0.2, random_state=42)

print("Split Dataset Extra Tree")
print(f"X_train: {X_train_Y1.shape}, X_test: {X_test_Y1.shape}")
print(f"Y1_train: {Y1_train.shape}, Y1_test: {Y1_test.shape}")

def objective_rf(trial):
    rf = ExtraTreesRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 600, step=100),
        max_depth=trial.suggest_int("max_depth", 5, 80, step=15),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10, step=2),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4, step=1),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        random_state=42
    )


    score = cross_val_score(rf, X_train_Y1, Y1_train, cv=5, scoring='neg_root_mean_squared_error')
    return score.mean()

study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_rf, n_trials=50)

# Menampilkan hasil trial terbaik
best_trial_extra = study_rf.best_trial
print(f"Best trial: {best_trial_extra.number}")
print(f"Best value (RMSE): {-best_trial_extra.value}")
print("Best hyperparameters:")
for key, value in best_trial_extra.params.items():
    print(f"{key}: {value}")

model_xtree = ExtraTreesRegressor(n_estimators=400,
    max_depth=80,
    min_samples_split= 4,
    min_samples_leaf=1,
    max_features=None,
    random_state=42)

model_xtree.fit(X_train_Y1, Y1_train)

#metrics evaluation
Y1_test_pred_xtree = model_xtree.predict(X_test_Y1)
test_rmse = root_mean_squared_error(Y1_test, Y1_test_pred_xtree)
test_mae = mean_absolute_error(Y1_test, Y1_test_pred_xtree)
test_mape = np.mean(np.abs((Y1_test - Y1_test_pred_xtree) / Y1_test)) * 100
test_r2 = r2_score(Y1_test, Y1_test_pred_xtree)

print(f"Test  RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | MAPE: {test_mape:.2f}% | R²: {test_r2:.4f}")

ROP = df_egs_initial['ROP']

plt.figure(figsize=(8, 6))
plt.scatter(Y1_test, Y1_test_pred_xtree, c='blue', label='Testing Data') #Corrected Y1_test_pred_xtree
plt.xlabel(f'True ROP Values', fontsize=12, fontweight='bold') # Assuming ROP is defined elsewhere
plt.ylabel(f'Predicted ROP Values', fontsize=12, fontweight='bold')
plt.title(f'True vs. Predicted ROP Values of Extra Tree\nR²={float(test_r2):.4f}, MAE={float(test_mae):.4f}, MAPE={float(test_mape):.4f}%, RMSE={float(test_rmse):.4f}',fontsize=14, fontweight='bold')
plt.plot([Y1_test.min(), Y1_test.max()], [Y1_test.min(), Y1_test.max()], 'k--', lw=2, c='red', label='Correlation Line')
legend = plt.legend(loc='upper left', fontsize=10, frameon=True)

plt.setp(legend.get_texts(), weight='bold')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xticks(fontsize=10, fontweight='bold')
plt.yticks(fontsize=10, fontweight='bold')
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.tight_layout()

plt.show()

"""**WELL LOG**"""

def make_comparison_log_plot(*datasets):
    # Sort logs by depth for each dataset
    sorted_datasets = [dataset.sort_values(by='BDE') for dataset in datasets]  # Sorting by depth 'BDE'

    # Create the figure and axis for the well log data
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13, 25))

    # Plot the data for each dataset (including predicted and real ROP)
    for i, dataset in enumerate(sorted_datasets):
        # Plot ROP for actual data based on depth with green color and solid line
        ax.plot(dataset['ROP'], dataset['BDE'], label=f'Real ROP (Dataset {i + 1})', color='blue', alpha=0.8, linewidth=2)  # Actual ROP
        # Plot ROP for predicted data with a slight offset to avoid overlap, dashed line with blue color
        ax.plot(dataset['ROP'] + 5, dataset['BDE'], label=f'Predicted ROP (Dataset {i + 1})', linestyle='--', color='#c89116', alpha=1, linewidth=2)  # Predicted ROP with offset

    # Adding labels and title with increased font size for clarity
    ax.set_xlabel("ROP (ft/hr)", fontsize=20, weight='bold')
    ax.set_ylabel("Depth (ft)", fontsize=20, weight='bold')
    ax.set_title(f'ROP Comparison vs Depth (Well Log)', fontsize=26, weight='bold')

    # Invert y-axis to match well log format
    ax.invert_yaxis()

    # Add grid, legend, and labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=20, frameon=True)

    # Fine-tune layout for better spacing
    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    plt.show()

# Assume that test_data is your prediction data and df_egs_extra is the original dataset
test_data = pd.DataFrame({'BDE': X_test_Y1[:, df_egs_extra.columns.get_loc('BDE')], 'ROP': Y1_test_pred_xtree.flatten()})
test_data = test_data.sort_values(by='BDE')  # Sort the predictions by depth

# Now call the function with the real and predicted ROP data
make_comparison_log_plot(test_data)

df_egs_initial.describe().T

def make_log_plot(well_data):
    # Sort logs by depth
    well_data = well_data.sort_values(by='BDE')

    # Set top and bottom of the plot
    ztop = well_data['BDE'].min()
    zbot = well_data['BDE'].max()

    # Create subplots
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(16, 24))

    # Plot some of the well log data
    ax[0].plot(well_data['HKL'], well_data['BDE'], '-', color='green')
    ax[1].plot(well_data['RHS'], well_data['BDE'], '-', color='brown')
    ax[2].plot(well_data['ADT'], well_data['BDE'], '-', color='orange')
    ax[3].plot(well_data['ISP'], well_data['BDE'], '-', color='teal')
    ax[4].plot(well_data['ROP'], well_data['BDE'], '-', color='lime')

    for i in range(len(ax)-0):
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x')

        ax[i].xaxis.label.set_fontsize(12)
        ax[i].xaxis.label.set_fontweight('bold')
        ax[i].xaxis.set_tick_params(labelsize=10)

        ax[i].yaxis.label.set_fontsize(12)
        ax[i].yaxis.label.set_fontweight('bold')
        ax[i].yaxis.set_tick_params(labelsize=10)

        # Move x-axis labels to the top
        ax[i].xaxis.tick_top()
        # Set x-axis label at the top
        ax[i].xaxis.set_label_position('top')

        ax[0].tick_params(axis='y', labelleft=False)
        ax[0].set_ylabel("Depth (ft)", fontsize=15, weight='bold')

        ax[0].set_xlabel("Hook Load (klbs)", fontsize=12, weight='bold')
        ax[0].set_xlim(well_data['HKL'].min(), well_data['HKL'].max())
        ax[1].set_xlabel("Rotating Hours (hrs)", fontsize=12, weight='bold')
        ax[1].set_xlim(well_data['RHS'].min(), well_data['RHS'].max())
        ax[2].set_xlabel("AutoDriller Torque (kft_lb)", fontsize=12, weight='bold')
        ax[2].set_xlim(well_data['ADT'].min(), well_data['ADT'].max())
        ax[3].set_xlabel("In Slip (unitless)", fontsize=12, weight='bold')
        ax[3].set_xlim(well_data['ISP'].min(), well_data['ISP'].max())
        ax[4].set_xlabel("Rate Of Penetration (ft/hr)", fontsize=12, weight='bold')
        ax[4].set_xlim(well_data['ROP'].min(), well_data['ROP'].max())

        ax[0].set_yticklabels([])
        ax[1].set_yticklabels([])
        ax[2].set_yticklabels([])
        ax[3].set_yticklabels([])
        ax[4].set_yticklabels([])

    # Add zoom functionality
    plt.subplots_adjust(wspace=0)
    plt.show()
# Call the function with your well_data
make_log_plot(df_egs_initial)

def make_log_plot(well_data):
    # Sort logs by depth
    well_data = well_data.sort_values(by='BDE')

    # Set top and bottom of the plot
    ztop = well_data['BDE'].min()
    zbot = well_data['BDE'].max()

    # Create subplots
    fig, ax = plt.subplots(nrows=1, ncols=8, figsize=(16, 24))

    # Plot some of the well log data
    ax[0].plot(well_data['ADP'], well_data['BDE'], '-', color='green')
    ax[1].plot(well_data['HKT'], well_data['BDE'], '-', color='brown')
    ax[2].plot(well_data['OVP'], well_data['BDE'], '-', color='orange')
    ax[3].plot(well_data['MRP'], well_data['BDE'], '-', color='teal')
    ax[4].plot(well_data['DOC'], well_data['BDE'], '-', color='lime')
    ax[5].plot(well_data['MPE'], well_data['BDE'], '-', color='cyan')
    ax[6].plot(well_data['DAS'], well_data['BDE'], '-', color='pink')
    ax[7].plot(well_data['BTQ'], well_data['BDE'], '-', color='purple')

    for i in range(len(ax)-0):
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x')

        ax[i].xaxis.label.set_fontsize(12)
        ax[i].xaxis.label.set_fontweight('bold')
        ax[i].xaxis.set_tick_params(labelsize=10)

        ax[i].yaxis.label.set_fontsize(12)
        ax[i].yaxis.label.set_fontweight('bold')
        ax[i].yaxis.set_tick_params(labelsize=10)

        # Move x-axis labels to the top
        ax[i].xaxis.tick_top()
        # Set x-axis label at the top
        ax[i].xaxis.set_label_position('top')

        ax[0].tick_params(axis='y', labelleft=False)
        ax[0].set_ylabel("Depth (ft)", fontsize=15, weight='bold')

        ax[0].set_xlabel("AD Block Position (ft)", fontsize=8, weight='bold')
        ax[0].set_xlim(well_data['ADP'].min(), well_data['ADP'].max())
        ax[1].set_xlabel("Hookload Threshold (klbs)", fontsize=8, weight='bold')
        ax[1].set_xlim(well_data['HKT'].min(), well_data['HKT'].max())

        ax[2].set_xlabel("Over Pull (klbs)", fontsize=8, weight='bold')
        ax[2].set_xlim(well_data['OVP'].min(), well_data['OVP'].max())
        ax[3].set_xlabel("Motor RPM", fontsize=8, weight='bold')
        ax[3].set_xlim(well_data['MRP'].min(), well_data['MRP'].max())
        ax[4].set_xlabel("Depth of Cut (in)", fontsize=8, weight='bold')
        ax[4].set_xlim(well_data['DOC'].min(), well_data['DOC'].max())
        ax[5].set_xlabel("Min Pressure (psi)", fontsize=8, weight='bold')
        ax[5].set_xlim(well_data['MPE'].min(), well_data['MPE'].max())
        ax[6].set_xlabel("DAS Stick Slip (unitless)", fontsize=8, weight='bold')
        ax[6].set_xlim(well_data['DAS'].min(), well_data['DAS'].max())
        ax[7].set_xlabel("Bit Torque (kft_lb)", fontsize=8, weight='bold')
        ax[7].set_xlim(well_data['BTQ'].min(), well_data['BTQ'].max())
        ax[7].set_xscale('log')

        ax[0].set_yticklabels([])
        ax[1].set_yticklabels([])
        ax[2].set_yticklabels([])
        ax[3].set_yticklabels([])
        ax[4].set_yticklabels([])
        ax[5].set_yticklabels([])
        ax[6].set_yticklabels([])
        ax[7].set_yticklabels([])

    # Add zoom functionality
    plt.subplots_adjust(wspace=0)
    plt.show()
# Call the function with your well_data
make_log_plot(df_egs_initial)

def make_log_plot(well_data):
    # Sort logs by depth
    well_data = well_data.sort_values(by='BDE')

    # Set top and bottom of the plot
    ztop = well_data['BDE'].min()
    zbot = well_data['BDE'].max()

    # Create subplots
    fig, ax = plt.subplots(nrows=1, ncols=8, figsize=(16, 24))

    # Plot some of the well log data
    ax[0].plot(well_data['TDR'], well_data['BDE'], '-', color='green')
    ax[1].plot(well_data['SPP'], well_data['BDE'], '-', color='brown')
    ax[2].plot(well_data['P1S'], well_data['BDE'], '-', color='orange')
    ax[3].plot(well_data['P2S'], well_data['BDE'], '-', color='teal')
    ax[4].plot(well_data['P3S'], well_data['BDE'], '-', color='lime')
    ax[5].plot(well_data['TPS'], well_data['BDE'], '-', color='cyan')
    ax[6].plot(well_data['FIO'], well_data['BDE'], '-', color='pink')
    ax[7].plot(well_data['LNW'], well_data['BDE'], '-', color='purple')

    for i in range(len(ax)-0):
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x')

        ax[i].xaxis.label.set_fontsize(12)
        ax[i].xaxis.label.set_fontweight('bold')
        ax[i].xaxis.set_tick_params(labelsize=10)

        ax[i].yaxis.label.set_fontsize(12)
        ax[i].yaxis.label.set_fontweight('bold')
        ax[i].yaxis.set_tick_params(labelsize=10)

        # Move x-axis labels to the top
        ax[i].xaxis.tick_top()
        # Set x-axis label at the top
        ax[i].xaxis.set_label_position('top')

        ax[0].tick_params(axis='y', labelleft=False)
        ax[0].set_ylabel("Depth (ft)", fontsize=15, weight='bold')

        ax[0].set_xlabel("Top Drive Rotary (RPM)", fontsize=8, weight='bold')
        ax[0].set_xlim(well_data['TDR'].min(), well_data['TDR'].max())
        ax[1].set_xlabel("Standpipe Pressure (psi)", fontsize=8, weight='bold')
        ax[1].set_xlim(well_data['SPP'].min(), well_data['SPP'].max())

        ax[2].set_xlabel("Pump 1 strokes/min", fontsize=8, weight='bold')
        ax[2].set_xlim(well_data['P1S'].min(), well_data['P1S'].max())
        ax[3].set_xlabel("Pump 2 strokes/min", fontsize=8, weight='bold')
        ax[3].set_xlim(well_data['P2S'].min(), well_data['P2S'].max())
        ax[4].set_xlabel("Pump 3 strokes/min", fontsize=8, weight='bold')
        ax[4].set_xlim(well_data['P3S'].min(), well_data['P3S'].max())
        ax[5].set_xlabel("Trip Speed (ft/min)", fontsize=8, weight='bold')
        ax[5].set_xlim(well_data['TPS'].min(), well_data['TPS'].max())
        ax[6].set_xlabel("Flow In-Out (flow_percent)", fontsize=8, weight='bold')
        ax[6].set_xlim(well_data['FIO'].min(), well_data['FIO'].max())
        ax[7].set_xlabel("Line Wear (ton_miles)", fontsize=8, weight='bold')
        ax[7].set_xlim(well_data['LNW'].min(), well_data['LNW'].max())
        ax[7].set_xscale('log')

        ax[0].set_yticklabels([])
        ax[1].set_yticklabels([])
        ax[2].set_yticklabels([])
        ax[3].set_yticklabels([])
        ax[4].set_yticklabels([])
        ax[5].set_yticklabels([])
        ax[6].set_yticklabels([])
        ax[7].set_yticklabels([])

    # Add zoom functionality
    plt.subplots_adjust(wspace=0)
    plt.show()
# Call the function with your well_data
make_log_plot(df_egs_initial)

def make_log_plot(well_data):
    # Sort logs by depth
    well_data = well_data.sort_values(by='BDE')

    # Set top and bottom of the plot
    ztop = well_data['BDE'].min()
    zbot = well_data['BDE'].max()

    # Create subplots
    fig, ax = plt.subplots(nrows=1, ncols=8, figsize=(16, 24))

    # Plot some of the well log data
    ax[0].plot(well_data['TOF'], well_data['BDE'], '-', color='green')
    ax[1].plot(well_data['DEX'], well_data['BDE'], '-', color='brown')
    ax[2].plot(well_data['TMU'], well_data['BDE'], '-', color='orange')
    ax[3].plot(well_data['CVT'], well_data['BDE'], '-', color='teal')
    ax[4].plot(well_data['BR2'], well_data['BDE'], '-', color='lime')
    ax[5].plot(well_data['BR1'], well_data['BDE'], '-', color='cyan')
    ax[6].plot(well_data['RRM'], well_data['BDE'], '-', color='pink')
    ax[7].plot(well_data['TDT'], well_data['BDE'], '-', color='purple')

    for i in range(len(ax)-0):
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x')

        ax[i].xaxis.label.set_fontsize(12)
        ax[i].xaxis.label.set_fontweight('bold')
        ax[i].xaxis.set_tick_params(labelsize=10)

        ax[i].yaxis.label.set_fontsize(12)
        ax[i].yaxis.label.set_fontweight('bold')
        ax[i].yaxis.set_tick_params(labelsize=10)

        # Move x-axis labels to the top
        ax[i].xaxis.tick_top()
        # Set x-axis label at the top
        ax[i].xaxis.set_label_position('top')

        ax[0].tick_params(axis='y', labelleft=False)
        ax[0].set_ylabel("Depth (ft)", fontsize=15, weight='bold')

        ax[0].set_xlabel("Temp Out Flow (Fahren)", fontsize=8, weight='bold')
        ax[0].set_xlim(well_data['TOF'].min(), well_data['TOF'].max())
        ax[1].set_xlabel("D-exponent (log)", fontsize=8, weight='bold')
        ax[1].set_xlim(well_data['DEX'].min(), well_data['DEX'].max())

        ax[2].set_xlabel("Torque Motor (unitless)", fontsize=8, weight='bold')
        ax[2].set_xlim(well_data['TMU'].min(), well_data['TMU'].max())
        ax[3].set_xlabel("Conv Torque (k-lbs)", fontsize=8, weight='bold')
        ax[3].set_xlim(well_data['CVT'].min(), well_data['CVT'].max())
        ax[4].set_xlabel("Bit Rev/Min Minimum", fontsize=8, weight='bold')
        ax[4].set_xlim(well_data['BR2'].min(), well_data['BR2'].max())
        ax[5].set_xlabel("Bit Rev/Min Max", fontsize=8, weight='bold')
        ax[5].set_xlim(well_data['BR1'].min(), well_data['BR1'].max())
        ax[6].set_xlabel("Rotary Rev/Min", fontsize=8, weight='bold')
        ax[6].set_xlim(well_data['RRM'].min(), well_data['RRM'].max())
        ax[7].set_xlabel("Top Drive Torque (kft-lb)", fontsize=8, weight='bold')
        ax[7].set_xlim(well_data['TDT'].min(), well_data['TDT'].max())
        ax[7].set_xscale('log')

        ax[0].set_yticklabels([])
        ax[1].set_yticklabels([])
        ax[2].set_yticklabels([])
        ax[3].set_yticklabels([])
        ax[4].set_yticklabels([])
        ax[5].set_yticklabels([])
        ax[6].set_yticklabels([])
        ax[7].set_yticklabels([])

    # Add zoom functionality
    plt.subplots_adjust(wspace=0)
    plt.show()
# Call the function with your well_data
make_log_plot(df_egs_initial)

warnings.filterwarnings("ignore", category=DataConversionWarning)

mae_test_et = []
mape_test_et =[]
rmse_test_et =[]
r2_test_et =  []

# Split the data into features (X) and target (y)
X = df_egs_extra.iloc[:, :-1]  # All columns except the last one
y = df_egs_extra.iloc[:, -1:]   # The last column

# Define the list of possible train_sizes
possible_train_sizes = [0.7, 0.75, 0.8, 0.85, 0.9]
# Number of iterations
num_iterations = 10
for iteration in range(num_iterations):
    # Randomly select a train_size
    train_size = random.choice(possible_train_sizes)
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=1-train_size)

    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    X_train_i = pd.DataFrame(iso_forest.fit_predict(X_train), index=X_train.index, columns=['outlier'])
    X_test_i = pd.DataFrame(iso_forest.predict(X_test), index=X_test.index, columns=['outlier'])

    # Filter inliers
    X_train_inliers = X_train[X_train_i['outlier'] != -1]
    X_test_inliers = X_test[X_test_i['outlier'] != -1]

    # Predict target variable for inliers
    y_train = y_train.loc[X_train_inliers.index]
    y_test = y_test.loc[X_test_inliers.index]

    possible_cv = [3,4,5,6,7,8]
    pick_cv = random.choice(possible_cv)

    def objective_rf(trial):
      rf = ExtraTreesRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 600, step=100),
        max_depth=trial.suggest_int("max_depth", 5, 80, step=15),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10, step=2),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4, step=1),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        random_state=42
    )

      score = cross_val_score(rf, X_train_inliers, y_train, cv=pick_cv, scoring='neg_root_mean_squared_error')
      return score.mean()

    study_rf = optuna.create_study(direction="maximize")
    study_rf.optimize(objective_rf, n_trials=30)

    # Menampilkan hasil trial terbaik
    best_trial_extra_loop = study_rf.best_trial
    print(f"Best trial: {best_trial_extra_loop.number}")
    print(f"Best value (RMSE): {-best_trial_extra_loop.value}")
    print("Best hyperparameters:")
    for key, value in best_trial_extra_loop.params.items():
      print(f"{key}: {value}")

    # Display train_size and cv size for each iteration
    print(f"\nIteration {iteration + 1}:")
    print(f"Training Size: {train_size}")
    print(f"Cross-Validation Size (cv): {pick_cv}")

    best_model_xtree = ExtraTreesRegressor(
      n_estimators=best_trial_extra_loop.params['n_estimators'],
      max_depth=best_trial_extra_loop.params['max_depth'],
      min_samples_split=best_trial_extra_loop.params['min_samples_split'],
      min_samples_leaf=best_trial_extra_loop.params['min_samples_leaf'],
      max_features=best_trial_extra_loop.params['max_features'],
      random_state=42)

    # Use the best model for predictions
    best_model_xtree.fit(X_train_inliers, y_train)
    y_pred_test = best_model_xtree.predict(X_test_inliers)

    # Make predictions on the test data
    y_pred_test = best_model_xtree.predict(X_test_inliers)
    y_pred_test = pd.DataFrame(y_pred_test)

    # Calculate R2 for training and testing data
    r2_test = r2_score(y_test, y_pred_test)

    # Calculate MAE, MAPE, and RMSE
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = np.mean(np.abs((y_pred_test.values - y_test.values) / y_test.values)) * 100
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Append
    mae_test_et.append(mae_test)
    mape_test_et.append(mape_test)
    rmse_test_et.append(rmse_test)
    r2_test_et.append(r2_test)

mean_mae = np.mean(mae_test_et)
std_mae = np.std(mae_test_et)

mean_mape = np.mean(mape_test_et)
std_mape = np.std(mape_test_et)

mean_rmse = np.mean(rmse_test_et)
std_rmse = np.std(rmse_test_et)

mean_r2 = np.mean(r2_test_et)
std_r2 = np.std(r2_test_et)

print("Errors Extra Tree:")
print('mae_test_et=',mae_test_et)
print('mape_test_et=',mape_test_et)
print('rmse_test_et=',rmse_test_et)
print('r2_test_et=',r2_test_et)

print("\nMean and Std of Errors across iterations Extra Tree:")
print(f"Mean MAE: {mean_mae:.4f}, Std MAE: {std_mae:.4f}")
print(f"Mean MAPE: {mean_mape:.4f}, Std MAPE: {std_mape:.4f}")
print(f"Mean RMSE: {mean_rmse:.4f}, Std RMSE: {std_rmse:.4f}")
print(f"Mean R²: {mean_r2:.4f}, Std R²: {std_r2:.4f}")

"""**LIGHT GRADIENT BOOSTING MODELING**"""

df_egs_lgb=df_egs_initial.drop(columns=['LNW', 'HKT', 'RHS', 'ADT', 'ISP'])
df_egs_lgb.head()

X = df_egs_lgb.drop(columns=['ROP'])
X = X.values #konversi ke numpy array

Y1 = df_egs_lgb['ROP']
Y1 = Y1.values #konversi ke numpy array

X_train_Y1_lgb, X_test_Y1_lgb, Y1_train_lgb, Y1_test_lgb = train_test_split(X, Y1, test_size=0.2, random_state=42)

print("Split Dataset Light Gradient")
print(f"X_train: {X_train_Y1_lgb.shape}, X_test: {X_test_Y1_lgb.shape}")
print(f"Y1_train: {Y1_train_lgb.shape}, Y1_test: {Y1_test_lgb.shape}")

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def objective_lgb(trial):
    rf_lgb = lgb.LGBMRegressor(
    n_estimators=trial.suggest_categorical('n_estimators', [100, 200, 300, 400, 500, 600]),
    learning_rate=trial.suggest_categorical('learning_rate', [0.001, 0.01, 0.1, 0.2]),
    max_depth=trial.suggest_categorical('max_depth', [15, 20, 25]),
    min_child_samples=trial.suggest_categorical('min_child_samples', [5, 6, 7, 8]),
    num_leaves=trial.suggest_categorical('num_leaves', [30, 40, 50, 60]),
    verbosity=-1)

    score = cross_val_score(rf_lgb, X_train_Y1_lgb, Y1_train_lgb, cv=5, scoring='neg_root_mean_squared_error')
    return score.mean()

study_rf_lgb = optuna.create_study(direction="maximize")
study_rf_lgb.optimize(objective_lgb, n_trials=50)

# Menampilkan hasil trial terbaik
best_trial_lgb = study_rf_lgb.best_trial
print(f"Best trial: {best_trial_lgb.number}")
print(f"Best value (RMSE): {-best_trial_lgb.value}")
print("Best hyperparameters:")
for key, value in best_trial_lgb.params.items():
    print(f"{key}: {value}")

model_lgb = lgb.LGBMRegressor(
    n_estimators=best_trial_lgb.params['n_estimators'],
    learning_rate=best_trial_lgb.params['learning_rate'],
    max_depth=best_trial_lgb.params['max_depth'],
    min_child_samples=best_trial_lgb.params['min_child_samples'],
    num_leaves=best_trial_lgb.params['num_leaves'],
    random_state=42
)

model_lgb.fit(X_train_Y1_lgb, Y1_train_lgb)

#metrics evaluation
Y1_test_pred_lgb = model_lgb.predict(X_test_Y1_lgb)
test_rmse_lgb = root_mean_squared_error(Y1_test_lgb, Y1_test_pred_lgb)
test_mae_lgb = mean_absolute_error(Y1_test_lgb, Y1_test_pred_lgb)
test_mape_lgb = np.mean(np.abSs((Y1_test_lgb - Y1_test_pred_lgb) / Y1_test_lgb)) * 100
test_r2_lgb = r2_score(Y1_test_lgb, Y1_test_pred_lgb)

print(f"Test  RMSE: {test_rmse_lgb:.4f} | MAE: {test_mae_lgb:.4f} | MAPE: {test_mape_lgb:.2f}% | R²: {test_r2_lgb:.4f}")

ROP = df_egs_initial['ROP']

plt.figure(figsize=(8, 6))
plt.scatter(Y1_test_lgb, Y1_test_pred_lgb, c='blue', label='Testing Data') #Corrected Y1_test_pred_xtree
plt.xlabel(f'True ROP Values', fontsize=12, fontweight='bold') # Assuming ROP is defined elsewhere
plt.ylabel(f'Predicted ROP Values', fontsize=12, fontweight='bold')
plt.title(f'True vs. Predicted ROP Values of Light Gradient Boosting\nR²={float(test_r2_lgb):.4f}, MAE={float(test_mae_lgb):.4f}, MAPE={float(test_mape_lgb):.4f}%, RMSE={float(test_rmse_lgb):.4f}',fontsize=14, fontweight='bold')
plt.plot([Y1_test_lgb.min(), Y1_test_lgb.max()], [Y1_test_lgb.min(), Y1_test_lgb.max()], 'k--', lw=2, c='red', label='Correlation Line')
legend = plt.legend(loc='upper left', fontsize=10, frameon=True)

plt.setp(legend.get_texts(), weight='bold')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xticks(fontsize=10, fontweight='bold')
plt.yticks(fontsize=10, fontweight='bold')
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.tight_layout()

plt.show()

warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

mae_test_lgb = []
mape_test_lgb =[]
rmse_test_lgb =[]
r2_test_lgb =  []

# Split the data into features (X) and target (y)
X = df_egs_lgb.iloc[:, :-1]  # All columns except the last one
y = df_egs_lgb.iloc[:, -1:]   # The last column

# Define the list of possible train_sizes
possible_train_sizes = [0.7, 0.75, 0.8, 0.85, 0.9]
# Number of iterations
num_iterations = 10
for iteration in range(num_iterations):
    # Randomly select a train_size
    train_size = random.choice(possible_train_sizes)
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=1-train_size)

    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    X_train_i = pd.DataFrame(iso_forest.fit_predict(X_train), index=X_train.index, columns=['outlier'])
    X_test_i = pd.DataFrame(iso_forest.predict(X_test), index=X_test.index, columns=['outlier'])

    # Filter inliers
    X_train_inliers = X_train[X_train_i['outlier'] != -1]
    X_test_inliers = X_test[X_test_i['outlier'] != -1]

    # Predict target variable for inliers
    y_train = y_train.loc[X_train_inliers.index]
    y_test = y_test.loc[X_test_inliers.index]

    possible_cv = [3,4,5,6,7,8]
    pick_cv = random.choice(possible_cv)

    def objective_lgb(trial):
      rf_lgb = lgb.LGBMRegressor(
        n_estimators=trial.suggest_categorical('n_estimators', [100, 200, 300, 400, 500, 600]),
        learning_rate=trial.suggest_categorical('learning_rate', [0.001, 0.01, 0.1, 0.2]),
        max_depth=trial.suggest_categorical('max_depth', [15, 20, 25]),
        min_child_samples=trial.suggest_categorical('min_child_samples', [5, 6, 7, 8]),
        num_leaves=trial.suggest_categorical('num_leaves', [30, 40, 50, 60]),
        verbosity=-1)

      score = cross_val_score(rf_lgb, X_train_inliers, y_train, cv=pick_cv, scoring='neg_root_mean_squared_error')
      return score.mean()


    study_rf = optuna.create_study(direction="maximize")
    study_rf.optimize(objective_lgb, n_trials=30)

    # Menampilkan hasil trial terbaik
    best_trial_lgb_loop = study_rf.best_trial
    print(f"Best trial: {best_trial_lgb_loop.number}")
    print(f"Best value (RMSE): {-best_trial_lgb_loop.value}")
    print("Best hyperparameters:")
    for key, value in best_trial_lgb_loop.params.items():
      print(f"{key}: {value}")

    # Display train_size and cv size for each iteration
    print(f"\nIteration {iteration + 1}:")
    print(f"Training Size: {train_size}")
    print(f"Cross-Validation Size (cv): {pick_cv}")

    best_model_lgb = lgb.LGBMRegressor(
    n_estimators=best_trial_lgb.params['n_estimators'],
    learning_rate=best_trial_lgb.params['learning_rate'],
    max_depth=best_trial_lgb.params['max_depth'],
    min_child_samples=best_trial_lgb.params['min_child_samples'],
    num_leaves=best_trial_lgb.params['num_leaves'],
    random_state=42
)

    # Use the best model for predictions
    best_model_lgb.fit(X_train_inliers, y_train)
    y_pred_test = best_model_lgb.predict(X_test_inliers)

    # Make predictions on the test data
    y_pred_test = best_model_lgb.predict(X_test_inliers)
    y_pred_test = pd.DataFrame(y_pred_test)

    # Calculate R2 for training and testing data
    r2_test = r2_score(y_test, y_pred_test)

    # Calculate MAE, MAPE, and RMSE
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = np.mean(np.abs((y_pred_test.values - y_test.values) / y_test.values)) * 100
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Append
    mae_test_lgb.append(mae_test)
    mape_test_lgb.append(mape_test)
    rmse_test_lgb.append(rmse_test)
    r2_test_lgb.append(r2_test)

mean_mae_lgb = np.mean(mae_test_lgb)
std_mae_lgb = np.std(mae_test_lgb)

mean_mape_lgb = np.mean(mape_test_lgb)
std_mape_lgb = np.std(mape_test_lgb)

mean_rmse_lgb = np.mean(rmse_test_lgb)
std_rmse_lgb = np.std(rmse_test_lgb)

mean_r2_lgb = np.mean(r2_test_lgb)
std_r2_lgb = np.std(r2_test_lgb)

print("Errors Light Gradient:")
print('mae_test_lgb=',mae_test_lgb)
print('mape_test_lgb=',mape_test_lgb)
print('rmse_test_lgb=',rmse_test_lgb)
print('r2_test_lgb=',r2_test_lgb)

print("\nMean and Std of Errors across iterations Light Gradient:")
print(f"Mean MAE: {mean_mae_lgb:.4f}, Std MAE: {std_mae_lgb:.4f}")
print(f"Mean MAPE: {mean_mape_lgb:.4f}, Std MAPE: {std_mape_lgb:.4f}")
print(f"Mean RMSE: {mean_rmse_lgb:.4f}, Std RMSE: {std_rmse_lgb:.4f}")
print(f"Mean R²: {mean_r2_lgb:.4f}, Std R²: {std_r2_lgb:.4f}")

"""**BOX-PLOT FINAL**"""

mae_test_et=[1.8931149161541196, 2.2900969687993467, 2.29115621242485, 1.775054780361753, 1.928648616600817, 2.011742012974744, 1.8675949965846967, 1.895366697360323, 1.934944775132267, 1.7231824695121933]
rmse_test_et=[3.2753426108391066, 6.976089353172175, 4.6011601749328515, 3.196714841925182, 3.538669094391515, 3.494782807079815, 3.314622174831485, 3.1365800779624156, 3.432205084848217, 3.197453871377268]
mape_test_et=[21.001158198284482, 13.124075556128068, 21.037835727287277, 3.8684943755695382, 14.605204149455076, 2.904978875514809, 1.779524653746635, 2.1276373769787402, 5.465647987525492, 19.820421296360077]
r2_test_et=[0.9983817274392631, 0.9930982589984415, 0.9970942739486883, 0.9987283869421997, 0.9981364257617937, 0.9980585482356998, 0.9984232837533388, 0.9986873691093394, 0.9982926083661323, 0.9982986942783244]

mae_test_lgb=[3.3634513608547705, 2.819829983316663, 2.1611586005219907, 4.3273815828722135, 2.7496014487944493, 3.135947681958597, 3.0475706428276883, 2.9801345131650367, 2.3081841475515694, 3.9083922942612164]
rmse_test_lgb=[6.729420873161621, 4.951635020343153, 3.2784830692864193, 17.450810720832358, 4.539480909905421, 5.986029428459075, 5.416812320636693, 5.594641222619378, 3.6048424371911145, 11.405324383397222]
mape_test_lgb=[254.96563359586838, 43.077026329020924, 215.1687048725953, 57.83870869767427, 109.97246668742349, 46.28058365188916, 264.7091450223529, 477.3737475762418, 145.7532578172842, 229.25472789860234]
r2_test_lgb=[0.9927779980014033, 0.9965254173020531, 0.9981475122997866, 0.9591358309356693, 0.9966447095750847, 0.9942527489021614, 0.9958631686871391, 0.9955290058501359, 0.9981482259997012, 0.9820757074883629]

mae_test_lstm =[3.4644402051568024, 3.560144416293582, 3.4757876387415716, 5.1714807783613965, 3.6164907774398984, 4.345508963507576, 4.661254837706281, 3.57039400630713, 3.5229308108767956, 4.124294855530197]
rmse_test_lstm=[np.float64(6.466270653054848), np.float64(4.991558779482158), np.float64(4.565588291047486), np.float64(7.9386012192088895), np.float64(5.138074115793825), np.float64(5.9218784292204525), np.float64(6.04644350738038), np.float64(5.628901425402441), np.float64(4.8230575995812695), np.float64(5.769044504160664)]
mape_test_lstm=[np.float64(317.9099343890593), np.float64(537.058842691365), np.float64(868.3247760847454), np.float64(3296.421501736034), np.float64(774.3388300187767), np.float64(705.0041703956726), np.float64(843.6772961141249), np.float64(482.21023367777855), np.float64(276.3754881010851), np.float64(1846.834679649249)]
r2_test_lstm=[0.9960473081762709, 0.996774760197733, 0.9974611277132862, 0.9923808757792215, 0.99704379993224, 0.9961162157723807, 0.9959287222622756, 0.9964074780183881, 0.9969840928638416, 0.9961414714380192]

algorithm_names = ['ET','LGboost', 'LSTM']

# Creating a boxplot
plt.figure(figsize=(10, 6))
boxplot = plt.boxplot([mae_test_et, mae_test_lgb, mae_test_lstm], labels=algorithm_names)

# Set font properties for title and labels
plt.title('Distribution of MAE Scores for Different ML Algorithms', fontsize=14, weight='bold')
plt.xlabel('ML Algorithms', fontsize=12, weight='bold')
plt.ylabel('MAE Score', fontsize=12, weight='bold')

# Set font properties for tick labels
plt.xticks(fontsize=10, weight='bold')
plt.yticks(fontsize=10, weight='bold')

# Set font properties for boxplot elements (medians, boxes, whiskers, etc.)
for key in boxplot:
    for item in boxplot[key]:
        item.set_linewidth(2)  # Adjust linewidth for better visibility
# Set linewidth for the frame of the plot
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)
plt.show()

algorithm_names = ['ET','LGboost', 'LSTM']

# Creating a boxplot
plt.figure(figsize=(10, 6))
boxplot = plt.boxplot([mape_test_et, mape_test_lgb, mape_test_lstm], labels=algorithm_names)

# Set font properties for title and labels
plt.title('Distribution of MAPE Scores for Different ML Algorithms', fontsize=14, weight='bold')
plt.xlabel('ML Algorithms', fontsize=12, weight='bold')
plt.ylabel('MAPE Score', fontsize=12, weight='bold')

# Set font properties for tick labels
plt.xticks(fontsize=10, weight='bold')
plt.yticks(fontsize=10, weight='bold')

# Set font properties for boxplot elements (medians, boxes, whiskers, etc.)
for key in boxplot:
    for item in boxplot[key]:
        item.set_linewidth(2)  # Adjust linewidth for better visibility
# Set linewidth for the frame of the plot
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)
plt.show()

algorithm_names = ['ET','LGboost', 'LSTM']

# Creating a boxplot
plt.figure(figsize=(10, 6))
boxplot = plt.boxplot([rmse_test_et, rmse_test_lgb, rmse_test_lstm], labels=algorithm_names)

# Set font properties for title and labels
plt.title('Distribution of RMSE Scores for Different ML Algorithms', fontsize=14, weight='bold')
plt.xlabel('ML Algorithms', fontsize=12, weight='bold')
plt.ylabel('RMSE Score', fontsize=12, weight='bold')

# Set font properties for tick labels
plt.xticks(fontsize=10, weight='bold')
plt.yticks(fontsize=10, weight='bold')

# Set font properties for boxplot elements (medians, boxes, whiskers, etc.)
for key in boxplot:
    for item in boxplot[key]:
        item.set_linewidth(2)  # Adjust linewidth for better visibility
# Set linewidth for the frame of the plot
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)
plt.show()

algorithm_names = ['ET','LGboost', 'LSTM']

# Creating a boxplot
plt.figure(figsize=(10, 6))
boxplot = plt.boxplot([r2_test_et, r2_test_lgb, r2_test_lstm], labels=algorithm_names)

# Set font properties for title and labels
plt.title('Distribution of R² Scores for Different ML Algorithms', fontsize=14, weight='bold')
plt.xlabel('ML Algorithms', fontsize=12, weight='bold')
plt.ylabel('R² Score', fontsize=12, weight='bold')

# Set font properties for tick labels
plt.xticks(fontsize=10, weight='bold')
plt.yticks(fontsize=10, weight='bold')

# Set font properties for boxplot elements (medians, boxes, whiskers, etc.)
for key in boxplot:
    for item in boxplot[key]:
        item.set_linewidth(2)  # Adjust linewidth for better visibility
# Set linewidth for the frame of the plot
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)
plt.show()
