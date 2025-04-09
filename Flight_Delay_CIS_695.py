import requests
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Constants
API_KEY = ''
FOLDER_PATH = "/Users/vinaypolanki/Desktop/CIS_695_Flight"
FILE_PATH = os.path.join(FOLDER_PATH, "full_flight_delays.csv")
MODEL_PATH = os.path.join(FOLDER_PATH, "delay_model.pkl")
DEP_ENCODER_PATH = os.path.join(FOLDER_PATH, "dep_encoder.pkl")
ARR_ENCODER_PATH = os.path.join(FOLDER_PATH, "arr_encoder.pkl")
REPORT_PATH = os.path.join(FOLDER_PATH, "classification_report.csv")

# Step 1: Fetch live flight delays
def fetch_flight_data():
    url = f'https://www.goflightlabs.com/flight_delays?access_key={API_KEY}&delay=60&type=departures'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get('data', [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return []

# Step 2: Process flight data
def process_flight_data(flights):
    # Normalize JSON to dataframe
    df = pd.json_normalize(flights)
    print(f"Total Records Fetched: {len(df)}")

    selected_cols = [
        'airline_iata', 'airline_icao', 'flight_number', 'status', 'dep_icao', 'dep_terminal', 'dep_gate',
        'dep_time', 'dep_time_utc', 'dep_estimated', 'dep_actual', 'dep_estimated_utc', 'dep_actual_utc',
        'arr_icao', 'arr_terminal', 'arr_gate', 'arr_baggage', 'arr_time', 'arr_time_utc', 'arr_estimated', 'arr_actual',
        'arr_estimated_utc', 'aircraft_icao', 'duration', 'dep_time_ts', 'arr_time_ts', 'dep_actual_ts', 'arr_estimated_ts'
    ]
    
    df = df[[col for col in selected_cols if col in df.columns]]

    # Compute delay if both dep_actual_ts and dep_time_ts are available
    if 'dep_actual_ts' in df.columns and 'dep_time_ts' in df.columns:
        df['departure_delay'] = (df['dep_actual_ts'] - df['dep_time_ts']) / 60  # seconds to minutes
        df['departure_delay'] = df['departure_delay'].fillna(0).astype(int)
    else:
        df['departure_delay'] = 0

    # Rename columns for readability
    df.rename(columns={
        'dep_icao': 'departure_airport',
        'arr_icao': 'arrival_airport'
    }, inplace=True)

    # Create delay category
    df['delay_category'] = pd.cut(
        df['departure_delay'],
        bins=[-1, 30, 60, float('inf')],
        labels=['Short Delay', 'Medium Delay', 'Long Delay']
    )

    # Fill missing values
    df = fill_missing_values(df)

    # Apply column renaming map
    df = rename_columns(df)

    return df

# Step 3: Fill missing values in columns
def fill_missing_values(df):
    df['dep_terminal'] = df['dep_terminal'].fillna('Unknown')
    df['arr_terminal'] = df['arr_terminal'].fillna('Unknown')
    df['dep_gate'] = df['dep_gate'].fillna('Unknown')
    df['arr_gate'] = df['arr_gate'].fillna('Unknown')
    df['arr_baggage'] = df['arr_baggage'].fillna('Unknown')
    df['aircraft_icao'] = df['aircraft_icao'].fillna('Unknown')
    
    df['delay_category'] = df['delay_category'].fillna('Unknown')

    # Replace missing estimated and actual times with 'Not Available'
    df['arr_estimated'] = df['arr_estimated'].fillna('Not Available')
    df['arr_estimated_utc'] = df['arr_estimated_utc'].fillna('Not Available')
    df['arr_estimated_ts'] = df['arr_estimated_ts'].fillna(0).astype(int)
    
    df['dep_estimated'] = df['dep_estimated'].fillna('Not Available')
    df['dep_estimated_utc'] = df['dep_estimated_utc'].fillna('Not Available')

    df['dep_actual'] = df['dep_actual'].fillna('Not Available')
    df['dep_actual_utc'] = df['dep_actual_utc'].fillna('Not Available')
    df['dep_actual_ts'] = df['dep_actual_ts'].fillna(0).astype(int)

    df['arr_actual'] = df['arr_actual'].fillna('Not Available')

    # Replace missing duration with 0
    df['duration'] = df['duration'].fillna(0).astype(int)
    
    return df

# Step 4: Rename columns for consistency
def rename_columns(df):
    column_rename_map = {
        "airline_iata": "Airline_IATA_Code",
        "airline_icao": "Airline_ICAO_Code",
        "flight_number": "Flight_Number",
        "status": "Flight_Status",
        "departure_airport": "Departure_Airport",
        "dep_terminal": "Departure_Terminal",
        "dep_gate": "Departure_Gate",
        "dep_time": "Scheduled_Departure_Time",
        "dep_time_utc": "Scheduled_Departure_Time_(UTC)",
        "dep_estimated": "Estimated_Departure_Time",
        "dep_actual": "Actual_Departure_Time",
        "dep_estimated_utc": "Estimated_Departure_Time_(UTC)",
        "dep_actual_utc": "Actual_Departure_Time_(UTC)",
        "arrival_airport": "Arrival_Airport",
        "arr_terminal": "Arrival_Terminal",
        "arr_gate": "Arrival_Gate",
        "arr_baggage": "Baggage_Claim",
        "arr_time": "Scheduled_Arrival_Time",
        "arr_time_utc": "Scheduled_Arrival_Time_(UTC)",
        "arr_estimated": "Estimated_Arrival_Time",
        "arr_actual": "Actual_Arrival_Time",
        "arr_estimated_utc": "Estimated_Arrival_Time_(UTC)",
        "aircraft_icao": "Aircraft_Type_(ICAO_Code)",
        "duration": "Flight_Duration_(Minutes)",
        "dep_time_ts": "Departure_Timestamp",
        "arr_time_ts": "Arrival_Timestamp",
        "dep_actual_ts": "Actual_Departure_Timestamp",
        "arr_estimated_ts": "Estimated_Arrival_Timestamp",
        "departure_delay": "Departure_Delay_(Minutes)",
        "delay_category": "Delay_Category"
    }
    return df.rename(columns=column_rename_map)

# Step 5: Save Data to CSV
def save_to_csv(df):
    os.makedirs(FOLDER_PATH, exist_ok=True)
    df.to_csv(FILE_PATH, index=False)
    print(f"✅ Cleaned data saved to {FILE_PATH}")

# Step 6: Train the model
def train_model(df):
    # Classify as delayed if departure delay > 15 minutes
    df['is_delayed'] = (df['departure_delay'] > 15).astype(int)
    
    df_majority = df[df['is_delayed'] == 1]
    df_minority = df[df['is_delayed'] == 0]

    # Undersample majority class
    df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    # Encode categorical variables
    le_dep = LabelEncoder()
    le_arr = LabelEncoder()
    df_balanced['dep_encoded'] = le_dep.fit_transform(df_balanced['departure_airport'].astype(str))
    df_balanced['arr_encoded'] = le_arr.fit_transform(df_balanced['arrival_airport'].astype(str))

    # Ensure duration is numeric
    df_balanced['duration'] = pd.to_numeric(df_balanced['duration'], errors='coerce').fillna(0)

    # Split data
    X = df_balanced[['dep_encoded', 'arr_encoded', 'duration']]
    y = df_balanced['is_delayed']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # Save the model and encoders
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le_dep, DEP_ENCODER_PATH)
    joblib.dump(le_arr, ARR_ENCODER_PATH)
    
    print(f"✅ Model and encoders saved to {FOLDER_PATH}")

    return model, X_test, y_test

# Step 7: Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Save classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(REPORT_PATH)
    print(f"✅ Classification report saved to {REPORT_PATH}")

# Main function to run all steps
def main():
    flights = fetch_flight_data()
    if flights:
        df = process_flight_data(flights)
        save_to_csv(df)
        model, X_test,