import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from tpot import TPOTRegressor

Country = 'India'

# Import the Climate change dataset
GlobalTemp_df = pd.read_csv("GlobalLandTemperaturesByCountry.csv")

# Extract rows which contain temperature values from the country specified
IndianTemp_df = GlobalTemp_df.loc[GlobalTemp_df['Country'] == Country]

# Select only dt and AverageTemperature from the DataFrame
IndianTemp_df = IndianTemp_df.iloc[:, [0, 1]]

# Remove Nan values from the DataFrame, dropna() removes rows which contain Nan values
IndianTemp_df = IndianTemp_df.dropna()

# Convert the string format of the date to an int format which only contains the year
int_dt = []
for i in range(len(IndianTemp_df['dt'])):
    int_dt.append(int(IndianTemp_df['dt'].iloc[i][:4]))

IndianTemp = pd.DataFrame({'dt': int_dt, 'AverageTemperature': list(IndianTemp_df.iloc[:, 1])})

# Find the average temperature per year for the country specified
# Get the unique number of years present in the DataFrame
dt_unique = np.unique(np.array(IndianTemp['dt']))

# Find the average temperature values each year
IndianTemp_avg = pd.DataFrame(columns=['dt', 'AverageTemperature'])
for dt in dt_unique:
    # Select all the rows which have the year dt
    IndianTemp_dt = IndianTemp.loc[IndianTemp['dt'] == dt]

    # Find the mean temperature for each year
    mean_temp = IndianTemp_dt['AverageTemperature'].mean(axis=0)

    # Create a new Dataframe which is used to append values to IndianTemp_avg
    temp_df = pd.DataFrame([[dt, mean_temp]], columns=['dt', 'AverageTemperature'])

    # Append the new temp_df to the IndianTemp_avg DataFrame
    IndianTemp_avg = IndianTemp_avg.append(temp_df)

# Use Regression to predict the temperature in the future
year = np.array(IndianTemp_avg['dt'])
year = year.reshape(len(year), 1)
Avg_Temp = np.array(IndianTemp_avg['AverageTemperature'])
Avg_Temp = Avg_Temp.reshape(len(Avg_Temp), 1)
lr = LinearRegression()
lr.fit(year, Avg_Temp)

# Predict the temperature
print lr.predict([2100])
