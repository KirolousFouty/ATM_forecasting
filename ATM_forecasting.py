import configparser
import os
from datetime import datetime
import shutil
from prophet import Prophet
from sklearn.linear_model import LinearRegression
import mysql.connector
import matplotlib.pyplot as plt
import pandas as pd
import logging
import warnings

def read_config(config_file='config.ini'):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file '{config_file}' not found.")
    config = configparser.ConfigParser()
    config.read(config_file)
    return config['DEFAULT']

def convert_to_timestamp(df):
    df['ds_numeric'] = df['ds'].apply(lambda x: x.timestamp())
    return df

def extract_ramps(df, threshold=0):
    ramps = []
    ramp_start_idx = 0

    for i in range(1, len(df)):
        if df['yhat'].iloc[i] - df['yhat'].iloc[i - 150] > threshold:
            ramps.append(df.iloc[ramp_start_idx:i].copy())
            ramp_start_idx = i

    if ramp_start_idx < len(df) and len(df) > ramp_start_idx:
        ramps.append(df.iloc[ramp_start_idx:].copy())
    
    return ramps

def save_plot(fig, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig.savefig(os.path.join(folder, filename))
    plt.close(fig)

def create_table_if_not_exists(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS atm_cash_forecasted (
            id INT AUTO_INCREMENT PRIMARY KEY,
            atm_id INT,
            balance DECIMAL(10, 2),
            time DATETIME,
            process_time DATETIME
        )
    """)

def insert_forecasted_data(cursor, atm_id, forecast):
    process_time = datetime.now()

    for _, row in forecast.iterrows():
        cursor.execute("""
            INSERT INTO atm_cash_forecasted (atm_id, balance, time, process_time)
            VALUES (%s, %s, %s, %s)
        """, (atm_id, row['yhat'], row['ds'], process_time))

def main():
    config = read_config()

    # Extract parameters from config.ini
    mysql_user = config['mysql_user']
    mysql_password = config['mysql_password']
    mysql_host = config['mysql_host']
    mysql_port = config['mysql_port']
    mysql_db = config['mysql_db']
    startDate = config['startDate']
    min_atm_id = int(config['min_atm_id'])
    max_atm_id = int(config['max_atm_id'])
    countryHolidays = config.getboolean('countryHolidays')
    select_query = config['select_query']
    threshold = int(config['threshold'])
    nmonths = int(config['nmonths'])
    growth = config['growth']
    yearly_seasonality = config.getboolean('yearly_seasonality')
    weekly_seasonality = config.getboolean('weekly_seasonality')
    daily_seasonality = config.getboolean('daily_seasonality')
    seasonality_mode = config['seasonality_mode']
    seasonality_prior_scale = float(config['seasonality_prior_scale'])
    fourier_order = int(config['fourier_order'])

    output_folder = "output"
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    logging.getLogger('prophet').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore')

    cnx = mysql.connector.connect(
        user=mysql_user, 
        password=mysql_password, 
        host=mysql_host, 
        port=mysql_port, 
        database=mysql_db
    )
    cursor = cnx.cursor()

    # Ensure the atm_cash_forecasted table exists
    create_table_if_not_exists(cursor)

    df = pd.read_sql(select_query, cnx)
    
    for atm_id in range(min_atm_id, max_atm_id + 1):
        if atm_id not in df['atm_id'].unique():
            print(f"\nSkipped ATM ID: {atm_id}")
            continue
        atm_output_folder = os.path.join(output_folder, f'atm_id_{atm_id}')
        if not os.path.exists(atm_output_folder):
            os.makedirs(atm_output_folder)

        print(f"\nProcessing ATM ID: {atm_id}")

        # First plot
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df[df['atm_id'] == atm_id]['time'], df[df['atm_id'] == atm_id]['balance'], label=f'ATM ID {atm_id} Balance')
        plt.title(f"Simulated ATM Balance Over Time (atm_id={atm_id})")
        plt.xlabel("Date")
        plt.ylabel("Balance")
        plt.grid(True)
        plt.legend()

        start_date = pd.to_datetime(startDate)
        end_date = start_date + pd.DateOffset(months=nmonths)
        plt.xlim(start_date, end_date)

        save_plot(fig, atm_output_folder, f'ATM_{atm_id}_balance_over_time.png')

        df_ATM = df[df['atm_id'] == atm_id].copy()
        df_ATM.drop('atm_id', axis=1, inplace=True)
        df_ATM.rename(columns={'balance': 'y', 'time': 'ds'}, inplace=True)

        # Use the entire dataset for training
        m = Prophet(
            growth=growth, 
            yearly_seasonality=yearly_seasonality, 
            weekly_seasonality=weekly_seasonality, 
            daily_seasonality=daily_seasonality, 
            seasonality_mode=seasonality_mode, 
            seasonality_prior_scale=seasonality_prior_scale
        )
        m.add_seasonality(name='monthly', period=30.5, fourier_order=fourier_order)
        if countryHolidays:
            m.add_country_holidays(country_name='EG')

        m.fit(df_ATM)

        # Forecast from the last data point
        future = m.make_future_dataframe(periods=24 * 30, freq='h')  # Forecast for the next 30 days (24 * 30 hours)
        forecast = m.predict(future)

        fig1 = m.plot(forecast)
        save_plot(fig1, atm_output_folder, f'ATM_{atm_id}_forecast.png')

        fig2 = m.plot_components(forecast)
        save_plot(fig2, atm_output_folder, f'ATM_{atm_id}_components.png')

        forecast = forecast[['ds', 'yhat']]
        forecast['yhat'] = forecast['yhat'].astype(int)

        # Insert forecasted data into the database
        insert_forecasted_data(cursor, atm_id, forecast)
        cnx.commit()

        # Actual vs Forecasted Plot
        fig3 = plt.figure(figsize=(10, 6))
        plt.plot(df_ATM['ds'], df_ATM['y'], label='Actual', color='black')
        plt.plot(forecast['ds'].tail(len(forecast)), forecast['yhat'].tail(len(forecast)), label='Forecasted', color='red')
        
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Balance')
        plt.title(f'Actual vs Forecasted ATM ID {atm_id} Balance')

        save_plot(fig3, atm_output_folder, f'ATM_{atm_id}_actual_vs_forecasted.png')

        forecast = convert_to_timestamp(forecast)
        ramps = extract_ramps(forecast, threshold)

        x_intercepts = []
        for ramp in ramps:
            X = ramp['ds_numeric'].values.reshape(-1, 1)
            y = ramp['yhat'].values
            model = LinearRegression()
            model.fit(X, y)
            m = model.coef_[0]
            c = model.intercept_

            if m != 0:
                x_zero_timestamp = -c / m
                x_intercepts.append(pd.to_datetime(x_zero_timestamp, unit='s'))

        results_file = os.path.join(atm_output_folder, f"ATM_{atm_id}_results.txt")
        for i, x_zero in enumerate(x_intercepts):
            formatted_date = x_zero.strftime('%Y-%m-%d %H:%M')
            with open(results_file, 'a') as file:
                file.write(f"Ramp {i+1}: y = 0 at ds = {formatted_date}\n")

        # yhat vs ds Plot
        fig4 = plt.figure(figsize=(12, 6))
        plt.plot(forecast['ds'], forecast['yhat'])
        plt.xlabel('ds')
        plt.ylabel('yhat')
        plt.title('Graph of yhat vs ds')
        save_plot(fig4, atm_output_folder, f'ATM_{atm_id}_yhat_vs_ds.png')

    cursor.close()
    cnx.close()
    print("\n\n\nPress Enter to exit program ...")
    input()

if __name__ == "__main__":
    main()
