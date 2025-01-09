"""First Best Solution"""
# #Importing Required libraries
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split

# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# import chardet

# # Detect the encoding
# with open(r"C:\Users\eDominer\Python Project\Sales Prediction\sales_data_sample.csv", 'rb') as f:
#     result = chardet.detect(f.read())

# # Use the detected encoding to read the file
# encoding = result['encoding']

# data = pd.read_csv(r"C:\Users\eDominer\Python Project\Sales Prediction\sales_data_sample.csv", encoding=encoding)

# # Preprocess data
# data['Month'] = data['MONTH_ID']
# data['Year'] = data['YEAR_ID']
# data['YearMonth'] = pd.to_datetime(data['YEAR_ID'].astype(str) + '-' + data['MONTH_ID'].astype(str) + '-01')

# # Selecting relevant features 
# features = data[['Month', 'Year', 'QUANTITYORDERED', 'PRICEEACH', 'MSRP', 'DEALSIZE']] 
# target = data['SALES']

# # Encoding DEALSIZE
# data = pd.get_dummies(data, columns=['DEALSIZE'], drop_first=True)

# # Updating features after encoding
# features = data[['Month', 'Year', 'QUANTITYORDERED', 'PRICEEACH', 'MSRP'] + list(data.columns[data.columns.str.startswith('DEALSIZE_')])]

# # Train Test Split
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# #Train Model 

# model = RandomForestRegressor()
# model.fit(X_train, y_train)

# #Predicting Output
# predictions = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, predictions)
# mae = mean_absolute_error(y_test, predictions)
# r2 = r2_score(y_test, predictions)

# print(f"Mean Squared Error: {mse}")
# print(f"Mean Absolute Error: {mae}")
# print(f"R-Squared: {r2}")

"""Second Best Solution"""

#Importing Required libraries
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import chardet
import numpy as np
import joblib


with open(r"C:\Users\eDominer\Python Project\Sales Prediction\sales_data_sample.csv", 'rb') as f:
    result = chardet.detect(f.read())

encoding = result['encoding']

data = pd.read_csv(r"C:\Users\eDominer\Python Project\Sales Prediction\sales_data_sample.csv", encoding=encoding)

data['Month'] = data['MONTH_ID']
data['Year'] = data['YEAR_ID']
data['YearMonth'] = pd.to_datetime(data['YEAR_ID'].astype(str) + '-' + data['MONTH_ID'].astype(str) + '-01')
 
features = data[['Month', 'Year', 'QUANTITYORDERED', 'PRICEEACH', 'MSRP', 'DEALSIZE']] 
target = data['SALES']

data = pd.get_dummies(data, columns=['DEALSIZE'], drop_first=True)

features = data[['Month', 'Year', 'QUANTITYORDERED', 'PRICEEACH', 'MSRP'] + list(data.columns[data.columns.str.startswith('DEALSIZE_')])]

numerical_features = ['Month', 'Year', 'QUANTITYORDERED', 'PRICEEACH', 'MSRP']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Save the scaler
scaler_filename = 'scaler.pkl'
joblib.dump(scaler, scaler_filename)

# Hyperparameter Tuning using Randomized Search
param_dist = {
    'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
    'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
random_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_distributions=param_dist, n_iter=100, cv=5, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

# Perform cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mean = -cv_scores.mean()

predictions = best_model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error (Test): {mse}")
print(f"Mean Absolute Error (Test): {mae}")
print(f"R-Squared (Test): {r2}")
print(f"Cross-Validated MSE (Train): {cv_mean}")
print(f"Best Model Parameters: {best_model.get_params()}")

# Save the model
model_filename = 'sales_prediction_model.pkl'
joblib.dump(best_model, model_filename)


"""Third Best Solution"""
# import pandas as pd
# from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.preprocessing import StandardScaler
# import chardet
# import numpy as np
# import joblib

# # Detect the encoding
# with open(r"C:\Users\eDominer\Python Project\Sales Prediction\sales_data_sample.csv", 'rb') as f:
#     result = chardet.detect(f.read())

# # Use the detected encoding to read the file
# encoding = result['encoding']

# data = pd.read_csv(r"C:\Users\eDominer\Python Project\Sales Prediction\sales_data_sample.csv", encoding=encoding)

# # Preprocess data
# data['Month'] = data['MONTH_ID']
# data['Year'] = data['YEAR_ID']
# data['Season'] = data['Month'] % 12 // 3 + 1  # Simple season feature
# data['Is_Holiday_Season'] = data['Month'].apply(lambda x: 1 if x in [11, 12] else 0)  # Holiday season feature

# # Selecting relevant features 
# features = data[['Month', 'Year', 'Season', 'Is_Holiday_Season', 'QUANTITYORDERED', 'PRICEEACH', 'MSRP', 'DEALSIZE']] 
# target = data['SALES']

# # Encoding DEALSIZE
# data = pd.get_dummies(data, columns=['DEALSIZE'], drop_first=True)

# # Updating features after encoding
# features = data[['Month', 'Year', 'Season', 'Is_Holiday_Season', 'QUANTITYORDERED', 'PRICEEACH', 'MSRP'] + list(data.columns[data.columns.str.startswith('DEALSIZE_')])]

# # Define numerical features
# numerical_features = ['Month', 'Year', 'Season', 'Is_Holiday_Season', 'QUANTITYORDERED', 'PRICEEACH', 'MSRP']

# # Train Test Split
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# # Normalize the features
# scaler = StandardScaler()
# X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
# X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# # Save the scaler
# scaler_filename = 'scaler.pkl'
# joblib.dump(scaler, scaler_filename)

# # Hyperparameter Tuning using Randomized Search
# param_dist = {
#     'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
#     'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]
# }
# random_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_distributions=param_dist, n_iter=100, cv=5, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
# random_search.fit(X_train, y_train)

# # Get the best model
# best_model = random_search.best_estimator_

# # Perform cross-validation
# cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
# cv_mean = -cv_scores.mean()

# # Predicting Output
# predictions = best_model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, predictions)
# mae = mean_absolute_error(y_test, predictions)
# r2 = r2_score(y_test, predictions)

# print(f"Mean Squared Error (Test): {mse}")
# print(f"Mean Absolute Error (Test): {mae}")
# print(f"R-Squared (Test): {r2}")
# print(f"Cross-Validated MSE (Train): {cv_mean}")
# print(f"Best Model Parameters: {best_model.get_params()}")

# # Save the model
# model_filename = 'sales_prediction_model.pkl'
# joblib.dump(best_model, model_filename)

"""Fourth Best Solution"""
# import optuna
# import lightgbm as lgb
# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# import chardet
# import numpy as np
# import joblib

# # Step 1: Detect the encoding of the CSV file
# with open(r"C:\Users\eDominer\Python Project\Sales Prediction\sales_data_sample.csv", 'rb') as f:
#     result = chardet.detect(f.read())

# # Step 2: Use the detected encoding to read the file
# encoding = result['encoding']
# data = pd.read_csv(r"C:\Users\eDominer\Python Project\Sales Prediction\sales_data_sample.csv", encoding=encoding)

# # Step 3: Preprocess data
# data['Month'] = data['MONTH_ID']
# data['Year'] = data['YEAR_ID']
# data['YearMonth'] = pd.to_datetime(data['YEAR_ID'].astype(str) + '-' + data['MONTH_ID'].astype(str) + '-01')

# # Step 4: Add lag features
# data['Sales_Lag_1'] = data['SALES'].shift(1)
# data['Sales_Lag_2'] = data['SALES'].shift(2)
# data['Sales_Lag_3'] = data['SALES'].shift(3)

# # Step 5: Fill missing values
# data = data.fillna(0)

# # Step 6: Selecting relevant features 
# features = data[['Month', 'Year', 'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_3', 'QUANTITYORDERED', 'PRICEEACH', 'MSRP']] 
# target = data['SALES']

# # Step 7: Encoding DEALSIZE
# data = pd.get_dummies(data, columns=['DEALSIZE'], drop_first=True)

# # Step 8: Update features after encoding
# features = data[['Month', 'Year', 'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_3', 'QUANTITYORDERED', 'PRICEEACH', 'MSRP'] + list(data.columns[data.columns.str.startswith('DEALSIZE_')])]

# # Step 9: Define numerical features
# numerical_features = ['Month', 'Year', 'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_3', 'QUANTITYORDERED', 'PRICEEACH', 'MSRP']

# # Step 10: Polynomial features
# poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# features_poly = poly.fit_transform(features[numerical_features])

# # Step 11: Combine polynomial features with original features
# features_combined = np.hstack([features_poly, features.drop(columns=numerical_features).values])

# # Step 12: Train Test Split
# X_train, X_test, y_train, y_test = train_test_split(features_combined, target, test_size=0.2, random_state=42)

# # Step 13: Normalize the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)  # Fit on training data
# X_test = scaler.transform(X_test)         # Transform test data

# # Step 14: Save the scaler
# scaler_filename = 'scaler.pkl'
# joblib.dump(scaler, scaler_filename)

# # Step 15: Define the objective function for optimization
# def objective(trial):
#     params = {
#         'objective': 'regression',
#         'metric': 'rmse',
#         'boosting_type': 'gbdt',
#         'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 150),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
#         'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
#         'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 1.0)
#     }
    
#     model = lgb.LGBMRegressor(**params)
#     cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
#     return -cv_scores.mean()

# # Step 16: Optimize hyperparameters
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=100)

# # Step 17: Train the model with the best parameters
# best_params = study.best_params
# model = lgb.LGBMRegressor(**best_params)
# model.fit(X_train, y_train)

# # Step 18: Make predictions
# y_pred = model.predict(X_test)

# # Step 19: Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f'Mean Squared Error: {mse}')
# print(f'Mean Absolute Error: {mae}')
# print(f'R^2 Score: {r2}')

# # Step 20: Save the model
# model_filename = 'sales_prediction_model.pkl'
# joblib.dump(model, model_filename)



