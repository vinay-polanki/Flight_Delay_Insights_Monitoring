
import os
import pandas as pd
import requests
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("🚀 Script started")

# Paths
gdrive_folder = "/Users/vinaypolanki/Desktop/CIS_695_GDrive"  # Simulating GDrive save
local_folder = "/Users/vinaypolanki/Desktop/CIS_695_Flight"
os.makedirs(gdrive_folder, exist_ok=True)
os.makedirs(local_folder, exist_ok=True)

csv_gdrive = os.path.join(gdrive_folder, "full_flight_delays.csv")
csv_local = os.path.join(local_folder, "full_flight_delays.csv")
log_file = os.path.join(gdrive_folder, "logs.txt")

API_KEY = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiI0IiwianRpIjoiMWNkNzY2YzY4NzU2YzQ4ZTQ3MGJhMTU2NmFiOTE3ODg1MjljMmJlNjExZmQ1YzQ5ZmQwMDJkMzRmOThkODNlMGUxMDY1NDA1OGY0ZTlmMzciLCJpYXQiOjE3MzkyODY1MTQsIm5iZiI6MTczOTI4NjUxNCwiZXhwIjoxNzcwODIyNTE0LCJzdWIiOiIyNDI4NSIsInNjb3BlcyI6W119.vtqjGOukH2aTa-hqkbEjtGqZPWBqZFv9QhxJ7VZDQyvghgcVVnDoG2KQBCfYp0nv7jQ1-Yx8Yky8a0VeQyo16g'

def log_message(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a') as log:
        log.write(f"[{timestamp}] {msg}\n")
    print(f"[{timestamp}] {msg}")

def fetch_flight_data():
    print("🔍 Fetching flight data from API...")
    url = f"https://www.goflightlabs.com/flight_delays?access_key={API_KEY}&delay=60&type=departures"
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json().get('data', [])
        print(f"✅ Fetched records: {len(data)}")
        return data
    except Exception as e:
        log_message(f"❌ Error: {e}")
        return []

def process_flight_data(flights):
    df = pd.json_normalize(flights)
    if df.empty:
        print("⚠️ No data to process.")
        return df
    df['departure_delay'] = (df['dep_actual_ts'] - df['dep_time_ts']) / 60
    df['departure_delay'] = df['departure_delay'].fillna(0).astype(int)
    df['delay_category'] = pd.cut(df['departure_delay'], bins=[-1, 15, 30, float('inf')],
                                  labels=['On Time', 'Short Delay', 'Long Delay'])
    df.rename(columns={'dep_icao': 'departure_airport', 'arr_icao': 'arrival_airport'}, inplace=True)
    df['departure_terminal'] = df.get('dep_terminal', 'Unknown').fillna('Unknown')
    df['arrival_terminal'] = df.get('arr_terminal', 'Unknown').fillna('Unknown')
    print(f"✅ Processed data with shape: {df.shape}")
    return df

def save_data(df):
    df.to_csv(csv_gdrive, index=False)
    df.to_csv(csv_local, index=False)
    log_message(f"✅ Saved {len(df)} records to Google Drive and Local")

def plot_all_airport_insights(df):
    def plot_bar(series, title, color):
        plt.figure(figsize=(12, 6))
        series.plot(kind='barh', color=color)
        plt.title(title)
        plt.xlabel('Flight Count')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    if 'departure_airport' in df.columns:
        dep_counts = df['departure_airport'].value_counts()
        plot_bar(dep_counts.head(25), "Top 25 Departure Airports by Flight Count", "purple")

    if 'arrival_airport' in df.columns:
        arr_counts = df['arrival_airport'].value_counts()
        plot_bar(arr_counts.head(25), "Top 25 Arrival Airports by Flight Count", "orange")

def train_models(df):
    try:
        df['is_delayed'] = (df['departure_delay'] > 15).astype(int)
        df = df[['departure_airport', 'arrival_airport', 'departure_terminal', 'arrival_terminal', 'is_delayed']].dropna()

        le = [LabelEncoder() for _ in range(4)]
        df.iloc[:, 0] = le[0].fit_transform(df.iloc[:, 0])
        df.iloc[:, 1] = le[1].fit_transform(df.iloc[:, 1])
        df.iloc[:, 2] = le[2].fit_transform(df.iloc[:, 2].astype(str))
        df.iloc[:, 3] = le[3].fit_transform(df.iloc[:, 3].astype(str))

        X = df.drop('is_delayed', axis=1)
        y = df['is_delayed']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        log_model = LogisticRegression(max_iter=200)
        log_model.fit(X_train, y_train)
        y_pred_log = log_model.predict(X_test)
        print("\n📊 Logistic Regression Report:\n")
        print(classification_report(y_test, y_pred_log))

        cm_log = confusion_matrix(y_test, y_pred_log)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_log, annot=True, fmt='d', cmap='Greens')
        plt.title('Confusion Matrix - Logistic Regression')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        tree_model = DecisionTreeClassifier(random_state=42)
        tree_model.fit(X_train, y_train)
        y_pred_tree = tree_model.predict(X_test)
        print("\n🌲 Decision Tree Report:\n")
        print(classification_report(y_test, y_pred_tree))

        cm_tree = confusion_matrix(y_test, y_pred_tree)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Oranges')
        plt.title('Confusion Matrix - Decision Tree')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    except Exception as e:
        print(f"⚠️ ML Training failed: {e}")

# Manual Run Once (not looping)
flights = fetch_flight_data()
if flights:
    df_cleaned = process_flight_data(flights)
    save_data(df_cleaned)
    plot_all_airport_insights(df_cleaned)
    train_models(df_cleaned)
else:
    print("❌ No flight data returned from API.")
