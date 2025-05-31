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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from matplotlib.ticker import FixedLocator
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

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

X = df_egs_initial.iloc[:, :-1].values
y = df_egs_initial.iloc[:, -1].values

X_train_Y1_lstm, X_test_Y1_lstm, Y1_train_lstm, Y1_test_lstm = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_Y1_lstm = X_train_Y1_lstm.reshape((X_train_Y1_lstm.shape[0], 1, X_train_Y1_lstm.shape[1]))
X_test_Y1_lstm = X_test_Y1_lstm.reshape((X_test_Y1_lstm.shape[0], 1, X_test_Y1_lstm.shape[1]))

scaler = RobustScaler()
X_train_Y1_lstm = scaler.fit_transform(X_train_Y1_lstm.reshape(X_train_Y1_lstm.shape[0], X_train_Y1_lstm.shape[2]))
X_test_Y1_lstm  = scaler.transform(X_test_Y1_lstm .reshape(X_test_Y1_lstm .shape[0], X_test_Y1_lstm.shape[2]))

scaler_Y = RobustScaler()
Y1_train_lstm = scaler_Y.fit_transform(Y1_train_lstm.reshape(-1, 1))  # reshape untuk target
Y1_test_lstm = scaler_Y.transform(Y1_test_lstm .reshape(-1, 1))

# Reshape back to 3D for LSTM
X_train_Y1_lstm = X_train_Y1_lstm.reshape((X_train_Y1_lstm.shape[0], 1, X_train_Y1_lstm.shape[1]))
X_test_Y1_lstm = X_test_Y1_lstm.reshape((X_test_Y1_lstm.shape[0], 1, X_test_Y1_lstm.shape[1]))

warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def objective_lstm(trial):
    global X_train_Y1_lstm, Y1_train_lstm, X_test_Y1_lstm, Y1_test_lstm
    n_units = trial.suggest_int('n_units', 32, 128)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_int('epochs', 10, 50)

    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(n_units, activation='relu', input_shape=(X_train_Y1_lstm.shape[1], X_train_Y1_lstm.shape[2]), return_sequences=True))

    # Add additional LSTM layers
    for _ in range(n_layers - 1):
        model.add(LSTM(n_units, activation='relu', return_sequences=True))

    model.add(Dropout(dropout_rate))  # Dropout layer to prevent overfitting
    model.add(Dense(1))  # Output layer for regression

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train_Y1_lstm, Y1_train_lstm, epochs=epochs, batch_size=batch_size, validation_data=(X_test_Y1_lstm, Y1_test_lstm), verbose=0)

    y_pred = model.predict(X_test_Y1_lstm)

    y_pred = y_pred.reshape(-1)  # Reshape to (samples,)

    Y1_test_lstm = Y1_test_lstm.reshape(-1)

    rmse = root_mean_squared_error(Y1_test_lstm, y_pred)

    return rmse  # We want to minimize the MSE, so we return the value

study_lstm = optuna.create_study(direction='minimize')
study_lstm.optimize(objective_lstm, n_trials=50)

# Display the best trial's results
best_trial_lstm = study_lstm.best_trial
print(f"Best trial: {best_trial_lstm.number}")
print(f"Best value (MSE): {best_trial_lstm.value}")
print("Best hyperparameters:")
for key, value in best_trial_lstm.params.items():
    print(f"{key}: {value}")

best_trial_lstm.params

model_lstm = Sequential()
model_lstm.add(LSTM(
    best_trial_lstm.params['n_units'],  # Number of units in LSTM layer
    activation='relu',
    input_shape=(X_train_Y1_lstm.shape[1], X_train_Y1_lstm.shape[2]),
    return_sequences=True
))

for _ in range(best_trial_lstm.params['n_layers'] - 1):
    model_lstm.add(LSTM(best_trial_lstm.params['n_units'], activation='relu', return_sequences=True))
model_lstm.add(Dropout(best_trial_lstm.params['dropout_rate']))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

model_lstm.summary()

model_lstm.fit(X_train_Y1_lstm, Y1_train_lstm,
               epochs=best_trial_lstm.params['epochs'],
               batch_size=best_trial_lstm.params['batch_size'],
               validation_data=(X_test_Y1_lstm, Y1_test_lstm),
               verbose=1)

# Predict on the test set
Y1_test_pred_lstm = model_lstm.predict(X_test_Y1_lstm)

Y1_test_pred_lstm = Y1_test_pred_lstm.reshape(-1, 1)

Y1_test_pred = scaler_Y.inverse_transform(Y1_test_pred_lstm)
Y1_test_actual = scaler_Y.inverse_transform(Y1_test_lstm)

# Calculate RMSE (Root Mean Squared Error)
test_rmse_lstm = np.sqrt(mean_squared_error(Y1_test_actual , Y1_test_pred ))

# Calculate MAE (Mean Absolute Error)
test_mae_lstm = mean_absolute_error(Y1_test_actual, Y1_test_pred)

# Calculate MAPE (Mean Absolute Percentage Error)
test_mape_lstm = np.mean(np.abs((Y1_test_pred - Y1_test_actual) / Y1_test_actual)) * 100

# Calculate R² (Coefficient of Determination)
test_r2_lstm = r2_score(Y1_test_actual , Y1_test_pred)

# Print evaluation metrics
print(f"Test RMSE: {test_rmse_lstm:.4f} | MAE: {test_mae_lstm:.4f} | MAPE: {test_mape_lstm:.2f}% | R²: {test_r2_lstm:.4f}")

ROP = df_egs_initial['ROP']

plt.figure(figsize=(8, 6))
plt.scatter(Y1_test_actual, Y1_test_pred, c='blue', label='Testing Data') #Corrected Y1_test_pred_xtree
plt.xlabel(f'True ROP Values', fontsize=12, fontweight='bold') # Assuming ROP is defined elsewhere
plt.ylabel(f'Predicted ROP Values', fontsize=12, fontweight='bold')
plt.title(f'True vs. Predicted ROP Values of Long-Short Term Memory\nR²={float(test_r2_lstm):.4f}, MAE={float(test_mae_lstm):.4f}, MAPE={float(test_mape_lstm):.4f}%, RMSE={float(test_rmse_lstm):.4f}',fontsize=14, fontweight='bold')
plt.plot([Y1_test_actual .min(), Y1_test_actual .max()], [Y1_test_actual.min(), Y1_test_actual .max()], 'k--', lw=2, c='red', label='Correlation Line')
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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)

# Initialize lists to store metrics
mae_test_lstm = []
mape_test_lstm = []
rmse_test_lstm = []
r2_test_lstm = []

# Now let's implement a loop to test LSTM over multiple iterations with scaling and evaluation
num_iterations = 10
for iteration in range(num_iterations):
    print(f"\nIteration {iteration + 1}:")

    # Random train-test split
    train_size = random.choice([0.7, 0.75, 0.8, 0.85, 0.9])
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=1-train_size)

    # Apply RobustScaler for feature scaling
    scaler_X = RobustScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Apply RobustScaler for target variable scaling
    scaler_y = RobustScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

    # Reshape to 3D for LSTM input
    X_train_scaled_3D = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled_3D = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    def objective_lstm(trial):
        global X_train_scaled_3D, y_train_scaled, X_test_scaled_3D, y_test_scaled
        # Hyperparameters to tune
        n_units = trial.suggest_int('n_units', 32, 128)
        n_layers = trial.suggest_int('n_layers', 1, 3)
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        epochs = trial.suggest_int('epochs', 10, 50)

        model = Sequential()
        model.add(LSTM(n_units, activation='relu', input_shape=(X_train_scaled_3D.shape[1], X_train_scaled_3D.shape[2]), return_sequences=True))

        # Add additional LSTM layers
        for _ in range(n_layers - 1):
          model.add(LSTM(n_units, activation='relu', return_sequences=True))

        model.add(Dropout(dropout_rate))  # Dropout layer to prevent overfitting
        model.add(Dense(1))  # Output layer for regression

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train_scaled_3D, y_train_scaled, epochs=epochs, batch_size=batch_size, validation_data=(X_test_scaled_3D, y_test_scaled), verbose=0)

        y_pred = model.predict(X_test_scaled_3D)
        y_pred = y_pred.reshape(-1)  # Reshape to (samples,)

        y_test_scaled = y_test_scaled.reshape(-1)

        rmse = root_mean_squared_error(y_test_scaled, y_pred)

        return rmse  # We want to minimize the MSE, so we return the value

    study_lstm = optuna.create_study(direction='minimize')
    study_lstm.optimize(objective_lstm, n_trials=30)

      # Display the best trial's results
    best_trial_lstm_loop = study_lstm.best_trial
    print(f"Best trial: {best_trial_lstm_loop.number}")
    print(f"Best value (RMSE): {best_trial_lstm_loop.value}")
    print("Best hyperparameters:")
    for key, value in best_trial_lstm_loop.params.items():
      print(f"{key}: {value}")

    # Display train_size and cv size for each iteration
    print(f"\nIteration {iteration + 1}:")
    print(f"Training Size: {train_size}")

    model_lstm_loop = Sequential()

    model_lstm_loop.add(LSTM(
      best_trial_lstm_loop.params['n_units'],  # Number of units in LSTM layer
      activation='relu',
      input_shape=(X_train_scaled_3D.shape[1], X_train_scaled_3D.shape[2]),
      return_sequences=True
    ))

    for _ in range(best_trial_lstm_loop.params['n_layers'] - 1):
      model_lstm_loop.add(LSTM(best_trial_lstm_loop.params['n_units'], activation='relu', return_sequences=True))
    model_lstm_loop.add(Dropout(best_trial_lstm_loop.params['dropout_rate']))
    model_lstm_loop.add(Dense(1))
    model_lstm_loop.compile(optimizer='adam', loss='mean_squared_error')

    model_lstm_loop.summary()

    # Fit the model
    model_lstm_loop.fit(X_train_scaled_3D, y_train_scaled,
               epochs=best_trial_lstm_loop.params['epochs'],
               batch_size=best_trial_lstm_loop.params['batch_size'],
               validation_data=(X_test_scaled_3D, y_test_scaled),
               verbose=1)

    # Make predictions on the test data
    y_pred = model_lstm_loop.predict(X_test_scaled_3D)

    # Rescale back to original scale before evaluating metrics
    y_pred_lstm_rescaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
    y_test_rescaled = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(-1)

    # Calculate the evaluation metrics
    mae_lstm = mean_absolute_error(y_test_rescaled, y_pred_lstm_rescaled)
    mape_lstm = np.mean(np.abs((y_test_rescaled - y_pred_lstm_rescaled) / y_test_rescaled)) * 100
    rmse_lstm = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_lstm_rescaled))
    r2_lstm = r2_score(y_test_rescaled, y_pred_lstm_rescaled)

    # Append metrics to lists
    mae_test_lstm.append(mae_lstm)
    mape_test_lstm.append(mape_lstm)
    rmse_test_lstm.append(rmse_lstm)
    r2_test_lstm.append(r2_lstm)

# Calculate the mean and standard deviation for each metric
mean_mae_lstm = np.mean(mae_test_lstm)
std_mae_lstm = np.std(mae_test_lstm)

mean_mape_lstm = np.mean(mape_test_lstm)
std_mape_lstm = np.std(mape_test_lstm)

mean_rmse_lstm = np.mean(rmse_test_lstm)
std_rmse_lstm = np.std(rmse_test_lstm)

mean_r2_lstm = np.mean(r2_test_lstm)
std_r2_lstm = np.std(r2_test_lstm)

print("Errors Light Gradient:")
print('mae_test_lgb=',mae_test_lstm)
print('mape_test_lgb=',mape_test_lstm)
print('rmse_test_lgb=',rmse_test_lstm)
print('r2_test_lgb=',r2_test_lstm)

# Display results
print("\nMean and Std of Errors across iterations LSTM:")
print(f"Mean MAE: {mean_mae_lstm:.4f}, Std MAE: {std_mae_lstm:.4f}")
print(f"Mean MAPE: {mean_mape_lstm:.4f}, Std MAPE: {std_mape_lstm:.4f}")
print(f"Mean RMSE: {mean_rmse_lstm:.4f}, Std RMSE: {std_rmse_lstm:.4f}")
print(f"Mean R²: {mean_r2_lstm:.4f}, Std R²: {std_r2_lstm:.4f}")
