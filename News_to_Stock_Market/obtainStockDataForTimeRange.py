import os
import pandas as pd
import threading
from datetime import datetime

# Define the source and destination directories
source_folder = "HistoricalData"
destination_folder = "SelectedHistoricalFolder"

# Define the time range for filtering
start_date = datetime.strptime("2006-10-09", "%Y-%m-%d")
end_date = datetime.strptime("2009-09-17", "%Y-%m-%d")

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)


def process_file(file_name):
    """Processes a single CSV file by filtering rows within the specified time range."""
    file_path = os.path.join(source_folder, file_name)
    output_path = os.path.join(destination_folder, file_name)

    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Convert the 'date' column to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Filter rows based on the date range
        filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

        # Save the filtered data to the destination folder
        filtered_df.to_csv(output_path, index=False)
        print(f"Processed {file_name} successfully.")
    except Exception as e:
        print(f"Failed to process {file_name}: {e}")


if __name__ == "__main__":
    # Get the list of CSV files in the source folder
    files = [f for f in os.listdir(source_folder) if f.endswith('.csv')]

    # Create a thread for each file
    threads = []
    for file_name in files:
        thread = threading.Thread(target=process_file, args=(file_name,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("All files have been processed.")
