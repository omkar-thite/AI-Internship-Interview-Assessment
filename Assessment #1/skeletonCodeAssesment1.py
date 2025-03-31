import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

import xgboost as xgb

# Load appointment data
df = pd.read_csv("appointments.csv")  # Contains scheduled_time, actual_time, doctor_id, patient_id

# Feature Engineering
df['delay'] = (pd.to_datetime(df['actual_time']) - pd.to_datetime(df['scheduled_time'])).dt.total_seconds() / 60
df['hour'] = pd.to_datetime(df['scheduled_time']).dt.hour
df['day_of_week'] = pd.to_datetime(df['scheduled_time']).dt.dayofweek

# Feature engineering
df['month'] = pd.to_datetime(df['scheduled_time']).dt.month
df['is_morning'] = (df['hour'] < 12).astype(int)
df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 17)).astype(int)
df['is_evening'] = (df['hour'] >= 17).astype(int)

# Calculate doctor-specific features
df['doctor_avg_delay'] = df.groupby('doctor_id')['delay'].transform('mean')
df['daily_appointments'] = df.groupby(['doctor_id', df['scheduled_time'].dt.date])['scheduled_time'].transform('count')
df['doc_unique_patients'] = df.groupby('doctor_id')['patient_id'].transform('nunique')

# Store doctor average delays for prediction function
doctor_avg_delays = df.groupby('doctor_id')['delay'].mean().to_dict()

# Define features and target variable
features = ['doctor_id', 'hour', 'day_of_week', 'month', 'is_morning', 'is_afternoon', 
            'is_evening', 'doctor_avg_delay', 'daily_appointments', 'doc_unique_patients']
target = 'delay'

# Train AI Model
X = df[features]
y = df[target]

# Train XGBoost model
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print(f'Most important features: {feature_importance}')


# Predict delay for future appointments
def predict_wait_time(doctor_id, scheduled_time, doc_avg_delays):

    # Create a dataframe with the appointment information
    appointment = pd.DataFrame({
        'doctor_id': [doctor_id],
        'scheduled_time': [scheduled_time]
    })

      # Extract basic time features
    appointment['hour'] = scheduled_time.hour
    appointment['minute'] = scheduled_time.minute
    appointment['day_of_week'] = scheduled_time.weekday()
    
    appointment['month'] = scheduled_time.month
    appointment['is_morning'] = 1 if scheduled_time.hour < 12 else 0
    appointment['is_afternoon'] = 1 if 12 <= scheduled_time.hour < 17 else 0
    appointment['is_evening'] = 1 if scheduled_time.hour >= 17 else 0

    appointment['doctor_avg_delay'] =  df['doctor_id'].apply(lambda x: doc_avg_delays[x])
    appointment['daily_appointments'] = df.groupby(['doctor_id', df['scheduled_time'].dt.date])['scheduled_time'].transform('count')
    appointment['doc_unique_patients'] = df.groupby('doctor_id')['patient_id'].nunique()
     
    return model.predict(appointment)[0]  # Predicted delay in minutes

# Example usage
predicted_delay = predict_wait_time(doctor_id=5, scheduled_time=datetime(2024, 3, 26, 18, 30), doc_avg_delays)
print(f"Expected wait time: {predicted_delay:.2f} minutes")
