import pandas as pd
import requests
import io
import re

# Get user input for filename
filename = input("Enter the filename of the CSV (e.g., 'site_coordinates.csv'): ")

# Define site coordinates
lat = input("Enter latitude: ")
lon = input("Enter longitude: ")

# Get user input for height
height = input("Enter hub height in meters (e.g., 50, 100, 150): ").strip()

# Get user input for date range
start_date = input("Enter start date (YYYYMMDD, e.g., 20180101): ").strip()
end_date = input("Enter end date (YYYYMMDD, e.g., 20251231): ").strip()  

# Build the Daily NASA POWER API URL
variables = f"WS{height}M,WD{height}M"  # WS{height}M for wind speed, WD{height}M for wind direction
url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters={variables}&community=AG&longitude={lon}&latitude={lat}&format=CSV&start={start_date}&end={end_date}"

print(f"\nFetching daily wind data at {height}m from NASA POWER API...")
response = requests.get(url)

# Check if the request was successful
if response.status_code != 200:
    print(f"Failed to fetch data. Error code: {response.status_code}")
    exit(1)

# Parse the raw CSV — NASA POWER CSVs have metadata header rows before the data
raw_text = response.text

# Find the line where actual CSV data starts (the header row contains "YEAR")
lines = raw_text.splitlines()
header_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("YEAR"))
csv_body = "\n".join(lines[header_idx:])

# Load into DataFrame
df = pd.read_csv(io.StringIO(csv_body))
df.columns = df.columns.str.strip()  # strip any accidental whitespace from column names

# Combine YEAR and DOY into a single date column
df["date"] = pd.to_datetime(df["YEAR"].astype(str) + df["DOY"].astype(str).str.zfill(3), format="%Y%j")

# Rename wind columns dynamically based on height
df = df.rename(columns={
    f"WS{height}M": "wind_speed",
    f"WD{height}M": "wind_direction_degrees",
})

# Keep only the required output columns
df = df[["date", "wind_speed", "wind_direction_degrees"]]

# Save to CSV
df.to_csv(filename, index=False)
print(f"\nSuccess! '{filename}' has been saved with {len(df)} rows.")
print(df.head())