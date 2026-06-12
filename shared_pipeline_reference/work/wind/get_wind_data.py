import pandas as pd
import requests

filename = "wind.csv"

# 1. Define site coordinates
lat = 30.06376168
lon = -94.06905776

# 2. Define exact timeframe (Format: YYYYMMDD)
start_date = "20180213"  
end_date = "20251222"    

# 3. Build the Daily NASA POWER API URL
variables = "WS50M,WD50M"  # WS{height}M for wind speed, WD{height}M for wind direction
url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters={variables}&community=AG&longitude={lon}&latitude={lat}&format=CSV&start={start_date}&end={end_date}"

print("Fetching daily wind data from NASA...")
response = requests.get(url)

if response.status_code == 200:
    # 4. Save the daily data to a CSV file
    # filename = "site_daily_wind_data.csv"
    with open(filename, "w") as f:
        f.write(response.text)
    print(f"Success! '{filename}' has been created")
else:
    print(f"Failed to fetch data. Error code: {response.status_code}")