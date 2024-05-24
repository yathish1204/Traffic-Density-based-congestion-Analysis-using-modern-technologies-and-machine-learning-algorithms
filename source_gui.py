import streamlit as st
import googlemaps
from datetime import timedelta
from datetime import datetime

import csv
import pandas as pd

st.title("TRAFFIC DENSITY PREDICTOR")

# Enter your API key here
gmaps = googlemaps.Client(key='AIzaSyBVxrNzZGh7475ccu2-ag3ic8I76ekaiy8')

# Take start point and destination locations as user input
start = st.text_input("Enter the starting point: ")
destination = st.text_input("Enter the destination: ")

# Take depart time and depart date as user input
date = st.text_input("Enter the date of travel (YYYY-MM-DD): ")
time = st.text_input("Enter the time of travel (HH:MM AM/PM): ")

# Take mode of transport as user input
mode = st.selectbox("Enter mode of transport",options=['driving', 'walking', 'bicycling'])

# Ask if user wants historical data
col1, col2, col3 = st.columns(3)

with col2:
    historical_data = st.button("\nGET THE TRAFFIC DENSITY FOR GIVEN ROUTE")

st.markdown("---")

if historical_data:
    # Concatenate date and time into datetime format
    depart = datetime.strptime(date + ' ' + time, '%Y-%m-%d %I:%M %p')

    # Get directions and traffic data for the route
    directions_result = gmaps.directions(start, destination, mode=mode, departure_time=depart,
                                         traffic_model='best_guess')

    # Extract relevant data from the directions result
    duration = directions_result[0]['legs'][0]['duration']['text']
    distance = directions_result[0]['legs'][0]['distance']['text']
    traffic_time = directions_result[0]['legs'][0]['duration_in_traffic']['text']
    traffic_delay = directions_result[0]['legs'][0]['duration_in_traffic']['value'] - \
                    directions_result[0]['legs'][0]['duration']['value']

    # # Print the results
    # print("Route from", start, "to", destination)
    # print("Travel distance:", distance)
    # print("Travel time without traffic:", duration)
    # print("Travel time with traffic:", traffic_time)
    # print("Traffic delay:", max(traffic_delay, 0), "seconds")

    # Set up CSV file
    file_name = start.replace(" ", "-") + "-" + destination.replace(" ", "-") + ".csv"
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Departure Time', 'Start Point', 'End Point', 'Travel Distance', 'Travel Time Without Traffic',
                         'Travel Time With Traffic', 'Traffic Delay'])

        # Print historical traffic data at intervals based on the time difference between current time and depart time
        current_time = datetime.now()
        depart_time = datetime.strptime(date + ' ' + time, '%Y-%m-%d %I:%M %p')
        time_diff = (depart_time - current_time).total_seconds() // 3600  # Convert time difference to hours
        if time_diff < 72:
            interval = timedelta(minutes=5)
        elif time_diff < 150:
            interval = timedelta(minutes=10)
        elif time_diff < 250:
            interval = timedelta(minutes=15)
        else:
            interval = timedelta(hours=1)

        st.success(f"\n Collecting traffic data from {current_time.strftime('%Y-%m-%d %H:%M:%S')} to {depart_time.strftime('%Y-%m-%d %H:%M:%S')} at intervals of {interval}...")
        while current_time < depart_time:
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            new_depart_time = datetime.strptime(formatted_time, '%Y-%m-%d %H:%M:%S')
            directions_result_csv = gmaps.directions(start, destination, mode=mode, departure_time=new_depart_time)
            duration = directions_result_csv[0]['legs'][0]['duration']['text']
            distance = directions_result_csv[0]['legs'][0]['distance']['text']
            # traffic_time = directions_result_csv[0]['legs'][0]['duration_in_traffic']['text']
            traffic_delay = directions_result_csv[0]['legs'][0]['duration_in_traffic']['value'] - \
                            directions_result_csv[0]['legs'][0]['duration']['value']

            if traffic_delay < 0:
                traffic_delay = 0
            writer.writerow([formatted_time, start, destination, distance, duration, traffic_time, traffic_delay])
            current_time += interval

            # Print exit message
        st.info(f"\n..... traffic data saved to {file_name}")

        st.success("\n... using the machine learning algorithm on traffic data generated from google API....")

            #MACHINE LEARNING ALGORITHMS

        # LOADING THE DATASET AND PERFORMING EDA
        traffic_data = pd.read_csv(f"{file_name}")  # loading the dataset

        print(traffic_data)  # displaying the dataset

        print(traffic_data.columns[traffic_data.isna().any()])  # checking for NaN values

        print(traffic_data.shape)  # checking the shape of dataset
        print(traffic_data.info())  # basic information about the dataset

        print(traffic_data.describe())  # statistical description about the dataset

        #pre-processing the dataset
        st.error("....processing the dataset.....")

        traffic_data['Travel Distance'] = traffic_data['Travel Distance'].str.replace(' km', '').astype(float)

        # Convert Departure Time column to datetime object
        traffic_data['Departure Time'] = pd.to_datetime(traffic_data['Departure Time'])

        # Create a new feature for hour of the day
        traffic_data['Hour of the Day'] = traffic_data['Departure Time'].dt.hour

        # Create a new feature for hour of the day
        traffic_data['Minute of the Hour'] = traffic_data['Departure Time'].dt.minute

        # Split the traffic_dataset into training and testing sets
        X = traffic_data[['Hour of the Day', 'Minute of the Hour', 'Travel Distance']].values
        # print(X)

        y = traffic_data['Traffic Delay'].values
        # print(y)

        import pandas as pd
        from sklearn.model_selection import train_test_split
        from xgboost import XGBRegressor

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train an XGBoost model
        xgb = XGBRegressor()
        xgb.fit(X_train, y_train)

        st.info('..... training the machine learning model.....')

        # Evaluate the model on the test set
        score = xgb.score(X_test, y_test)
        print("Model accuracy score:", score)

        # Example of predicting the traffic delay for a new user input

        # Create a new feature for hour of the day
        hour_of_the_day = depart.time().hour
        # print(hour_of_the_day)

        minute_of_the_hour = depart.time().minute
        # print(minute_of_the_hour)

        average_distance = traffic_data['Travel Distance'].mean()
        # print(average_distance)

        new_input = [[hour_of_the_day, minute_of_the_hour, traffic_data['Travel Distance'].mean()]]

        print(new_input)
        # new_input_df = pd.DataFrame([new_input])
        predicted_delay = xgb.predict(new_input)
        if predicted_delay < traffic_data['Traffic Delay'].mean() - traffic_data['Traffic Delay'].std():
            traffic_density = 'LOW TRAFFIC DENSITY'
        if predicted_delay > traffic_data['Traffic Delay'].mean() + traffic_data['Traffic Delay'].std():
            traffic_density = 'HIGH TRAFFIC DENSITY'
        else:
            traffic_density = 'NORMAL TRAFFIC DENSITY'

        #COMPARING OUR MODEL WITH GOOGLE MODEL
        column1, column2 = st.columns(2)
        with column1:
            #
            st.header("MODEL RESULTS")
            st.info(f"START POINT : {start}")
            st.info(f"DESTINATION : {destination}")
            st.info(f"DEPART TIME : {depart}")
            st.warning(f"TRAVEL DISTANCE : {traffic_data['Travel Distance'].mean()}")
            st.error(f"TRAVEL TIME : {traffic_data['Travel Time With Traffic'].mode().values[0]}")
            st.success(f"TRAFFIC DENSITY : {traffic_density}")

        with column2:
            st.header("GOOGLE RESULTS")
            st.info(f"START POINT : {start}")
            st.info(f"DESTINATION : {destination}")
            st.info(f"DEPART TIME : {depart}")
            st.warning(f"TRAVEL DISTANCE : {distance}")
            st.error(f"TRAVEL TIME : {traffic_time}")

else:
    st.error("\nCLICK SUBMIT BUTTON!")



