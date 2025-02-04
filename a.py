import speech_recognition as sr
import pyttsx3
import requests
import google.generativeai as genai
from datetime import datetime, timezone, timedelta
import pytz
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from dateutil.parser import parse

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Configure Google AI SDK
genai.configure(api_key="AIzaSyC0-y4hBELFWmSUS_ZKgfCh41EwLGKzax0")

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    safety_settings=safety_settings,
    generation_config=generation_config,
)

chat_session = model.start_chat(
    history=[
        {"role": "user", "parts": ["Hi\n"]},
        {"role": "model", "parts": ["Hi! How can I help you today? \n"]},
    ]
)

# OpenWeatherMap API configuration
WEATHER_API_KEY = "14248af3777f8cf95ff3592cfd999570"
CURRENT_WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"
FORECAST_API_URL = "http://api.openweathermap.org/data/2.5/forecast"
HISTORICAL_WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/onecall/timemachine"
AQI_API_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
UV_INDEX_API_URL = "http://api.openweathermap.org/data/2.5/uvi"

IST = pytz.timezone('Asia/Kolkata')


def get_current_time_in_ist():
    cities = ["Delhi", "Mumbai", "Kolkata", "Chennai"]
    current_times = {}
    now_utc = datetime.now(timezone.utc)

    for city in cities:
        now_ist = now_utc.astimezone(IST)
        current_times[city] = now_ist.strftime('%Y-%m-%d %H:%M:%S')

    return current_times


def display_units_of_measurement():
    units = {
        "temperature": "Celsius (°C)",
        "distance": "Kilometers (km)",
        "weight": "Kilograms (kg)"
    }
    return units


def get_weather_forecast(location):
    try:
        params = {"q": location, "appid": WEATHER_API_KEY, "units": "metric"}
        response = requests.get(CURRENT_WEATHER_API_URL, params=params)
        weather_data = response.json()

        if weather_data["cod"] == 200:
            weather_description = weather_data["weather"][0]["description"]
            temperature = weather_data["main"]["temp"]
            forecast = f"The current weather in {location} is {weather_description} with a temperature of {temperature}°C."
        else:
            forecast = f"Could not retrieve weather data for {location}. Response: {weather_data}"
            print(forecast)  # Log the response for debugging
    except Exception as e:
        forecast = f"An error occurred while fetching the weather data: {str(e)}"
        print(forecast)  # Log the exception for debugging

    return forecast


def get_hourly_forecast(location):
    try:
        params = {"q": location, "appid": WEATHER_API_KEY, "units": "metric"}
        response = requests.get(FORECAST_API_URL, params=params)
        weather_data = response.json()

        if weather_data["cod"] == "200":
            forecasts = weather_data["list"][:12]  # Get the next 12 hours forecast
            forecast = f"Hourly weather forecast for {location}:\n"
            for entry in forecasts:
                time = datetime.fromtimestamp(entry["dt"], timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                weather_description = entry["weather"][0]["description"]
                temperature = entry["main"]["temp"]
                forecast += f"{time}: {weather_description}, {temperature}°C\n"
        else:
            forecast = f"Could not retrieve hourly forecast data for {location}. Response: {weather_data}"
            print(forecast)  # Log the response for debugging
    except Exception as e:
        forecast = f"An error occurred while fetching the hourly forecast data: {str(e)}"
        print(forecast)  # Log the exception for debugging

    return forecast


def get_7_day_forecast(location):
    try:
        params = {"q": location, "appid": WEATHER_API_KEY, "units": "metric"}
        response = requests.get(FORECAST_API_URL, params=params)
        weather_data = response.json()

        if weather_data["cod"] == "200":
            forecast = f"7-day weather forecast for {location}:\n"
            daily_forecasts = weather_data["list"][::8]  # Get daily forecasts (every 8th entry)
            for entry in daily_forecasts:
                date = datetime.fromtimestamp(entry["dt"], timezone.utc).strftime('%Y-%m-%d')
                weather_description = entry["weather"][0]["description"]
                temperature_min = entry["main"]["temp_min"]
                temperature_max = entry["main"]["temp_max"]
                forecast += f"{date}: {weather_description}, {temperature_min}°C - {temperature_max}°C\n"
        else:
            forecast = f"Could not retrieve 7-day forecast data for {location}. Response: {weather_data}"
            print(forecast)  # Log the response for debugging
    except Exception as e:
        forecast = f"An error occurred while fetching the 7-day forecast data: {str(e)}"
        print(forecast)  # Log the exception for debugging

    return forecast


def get_weather_alerts(location):
    try:
        params = {"q": location, "appid": WEATHER_API_KEY}
        response = requests.get(FORECAST_API_URL, params=params)
        weather_data = response.json()

        if weather_data["cod"] == "200" and "alerts" in weather_data:
            alerts = weather_data["alerts"]
            alert_message = f"Weather alerts for {location}:\n"
            for alert in alerts:
                start_time = datetime.fromtimestamp(alert["start"], timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                end_time = datetime.fromtimestamp(alert["end"], timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                description = alert["description"]
                alert_message += f"From {start_time} to {end_time}: {description}\n"
        else:
            alert_message = f"No weather alerts for {location}."
        return alert_message
    except Exception as e:
        return f"An error occurred while fetching weather alerts: {str(e)}"


def get_air_quality(location):
    try:
        # Get latitude and longitude for the location
        params = {"q": location, "appid": WEATHER_API_KEY}
        response = requests.get(CURRENT_WEATHER_API_URL, params=params)
        location_data = response.json()

        if location_data["cod"] == 200:
            lat = location_data["coord"]["lat"]
            lon = location_data["coord"]["lon"]

            params = {"lat": lat, "lon": lon, "appid": WEATHER_API_KEY}
            response = requests.get(AQI_API_URL, params=params)
            aqi_data = response.json()

            if "list" in aqi_data:
                aqi = aqi_data["list"][0]["main"]["aqi"]
                aqi_message = f"The air quality index in {location} is {aqi}."
            else:
                aqi_message = f"Could not retrieve AQI data for {location}. Response: {aqi_data}"
        else:
            aqi_message = f"Could not retrieve location data for {location}. Response: {location_data}"
    except Exception as e:
        aqi_message = f"An error occurred while fetching AQI data: {str(e)}"

    return aqi_message


def get_uv_index(location):
    try:
        # Get latitude and longitude for the location
        params = {"q": location, "appid": WEATHER_API_KEY}
        response = requests.get(CURRENT_WEATHER_API_URL, params=params)
        location_data = response.json()

        if location_data["cod"] == 200:
            lat = location_data["coord"]["lat"]
            lon = location_data["coord"]["lon"]

            params = {"lat": lat, "lon": lon, "appid": WEATHER_API_KEY}
            response = requests.get(UV_INDEX_API_URL, params=params)
            uv_data = response.json()

            if "value" in uv_data:
                uv_index = uv_data["value"]
                uv_message = f"The UV index in {location} is {uv_index}."
            else:
                uv_message = f"Could not retrieve UV index data for {location}. Response: {uv_data}"
        else:
            uv_message = f"Could not retrieve location data for {location}. Response: {location_data}"
    except Exception as e:
        uv_message = f"An error occurred while fetching UV index data: {str(e)}"

    return uv_message


def get_historical_weather(location, date):
    try:
        # Get latitude and longitude for the location
        params = {"q": location, "appid": WEATHER_API_KEY}
        response = requests.get(CURRENT_WEATHER_API_URL, params=params)
        location_data = response.json()

        if location_data["cod"] == 200:
            lat = location_data["coord"]["lat"]
            lon = location_data["coord"]["lon"]

            # Convert date to timestamp
            timestamp = int(datetime.strptime(date, "%Y-%m-%d").timestamp())

            params = {
                "lat": lat,
                "lon": lon,
                "dt": timestamp,
                "appid": WEATHER_API_KEY,
                "units": "metric"
            }
            response = requests.get(HISTORICAL_WEATHER_API_URL, params=params)
            weather_data = response.json()

            if "current" in weather_data:
                weather_description = weather_data["current"]["weather"][0]["description"]
                temperature = weather_data["current"]["temp"]
                forecast = f"The weather in {location} on {date} was {weather_description} with a temperature of {temperature}°C."
            else:
                forecast = f"Could not retrieve historical weather data for {location} on {date}. Response: {weather_data}"
                print(forecast)  # Log the response for debugging
        else:
            forecast = f"Could not retrieve location data for {location}. Response: {location_data}"
            print(forecast)  # Log the response for debugging
    except Exception as e:
        forecast = f"An error occurred while fetching the historical weather data: {str(e)}"
        print(forecast)  # Log the exception for debugging

    return forecast


def get_sunrise_sunset(location):
    try:
        params = {"q": location, "appid": WEATHER_API_KEY, "units": "metric"}
        response = requests.get(CURRENT_WEATHER_API_URL, params=params)
        weather_data = response.json()

        if weather_data["cod"] == 200:
            sunrise_timestamp = weather_data["sys"]["sunrise"]
            sunset_timestamp = weather_data["sys"]["sunset"]
            sunrise = datetime.fromtimestamp(sunrise_timestamp, timezone.utc).strftime('%H:%M:%S')
            sunset = datetime.fromtimestamp(sunset_timestamp, timezone.utc).strftime('%H:%M:%S')
            forecast = f"The sunrise in {location} is at {sunrise} UTC and the sunset is at {sunset} UTC."
        else:
            forecast = f"Could not retrieve sunrise and sunset data for {location}. Response: {weather_data}"
            print(forecast)  # Log the response for debugging
    except Exception as e:
        forecast = f"An error occurred while fetching the sunrise and sunset data: {str(e)}"
        print(forecast)  # Log the exception for debugging

    return forecast


def compare_temperatures(city1, city2, request_type):
    """
    Compare temperatures between two cities and return the highest or lowest temperature.

    Parameters:
    city1 (str): The name of the first city.
    city2 (str): The name of the second city.
    request_type (str): A string that specifies whether to return the 'max' or 'min' temperature.

    Returns:
    str: A message indicating the city with the highest or lowest temperature.
    """

    try:
        # Get temperature for city1
        params1 = {"q": city1, "appid": WEATHER_API_KEY, "units": "metric"}
        response1 = requests.get(CURRENT_WEATHER_API_URL, params=params1)
        weather_data1 = response1.json()

        # Get temperature for city2
        params2 = {"q": city2, "appid": WEATHER_API_KEY, "units": "metric"}
        response2 = requests.get(CURRENT_WEATHER_API_URL, params=params2)
        weather_data2 = response2.json()

        # Check if both cities' temperatures are available
        if weather_data1["cod"] == 200 and weather_data2["cod"] == 200:
            temp_city1 = weather_data1["main"]["temp"]
            temp_city2 = weather_data2["main"]["temp"]

            if request_type == 'max':
                # Find the city with the maximum temperature
                higher_temp_city = city1 if temp_city1 > temp_city2 else city2
                return f"The highest temperature is in {higher_temp_city} with {max(temp_city1, temp_city2)}°C."
            elif request_type == 'min':
                # Find the city with the minimum temperature
                lower_temp_city = city1 if temp_city1 < temp_city2 else city2
                return f"The lowest temperature is in {lower_temp_city} with {min(temp_city1, temp_city2)}°C."
            else:
                return "Invalid request type. Please specify 'max' or 'min'."
        else:
            return "Temperature data for one or both cities is not available."
    except Exception as e:
        return f"An error occurred while comparing temperatures: {str(e)}"


def recognize_speech_from_mic(timeout=10, phrase_time_limit=5):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    # Adjust for ambient noise
    print("Adjusting for ambient noise... Please wait.")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Listening for speech...")
        audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    print("Recognizing speech...")
    try:
        transcription = recognizer.recognize_google(audio)
        print("You said: " + transcription)

        return transcription
    except sr.WaitTimeoutError:
        print("Listening timed out while waiting for phrase to start")
        engine.say("I didn't catch that. Please try speaking again.")
        engine.runAndWait()
    except sr.RequestError:
        print("API was unreachable or unresponsive")
        engine.say("There was a problem with the speech recognition service. Please try again later.")
        engine.runAndWait()
    except sr.UnknownValueError:
        print("Unable to recognize the speech")
        engine.say("I didn't understand that. Could you please repeat?")
        engine.runAndWait()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        engine.say(f"An error occurred: {str(e)}")
        engine.runAndWait()

    return None


def parse_date(date_str):
    try:
        parsed_date = parse(date_str, fuzzy=True)
        return parsed_date.strftime("%Y-%m-%d")
    except ValueError:
        return None


def extract_locations_and_date(transcription):
    tokens = word_tokenize(transcription)
    tagged = pos_tag(tokens)
    chunks = ne_chunk(tagged)
    locations = []
    date = None

    for chunk in chunks:
        if hasattr(chunk, 'label'):
            if chunk.label() == 'GPE':
                locations.append(' '.join(c[0] for c in chunk))
        elif chunk[1] == 'CD':  # CD is the tag for cardinal numbers
            potential_date = ' '.join(tokens[tokens.index(chunk[0]):])
            parsed_date = parse_date(potential_date)
            if parsed_date:
                date = parsed_date
                break

    return locations, date


def generate_response(forecast_data, query_type):
    if query_type == "comparison":
        response = "Here is the comparison of the weather:\n"
        for location, data in forecast_data.items():
            response += f"{location}: {data}\n"
    else:
        response = forecast_data
    return response


def get_weather_forecast_for_date(location, date):
    try:
        params = {"q": location, "appid": WEATHER_API_KEY, "units": "metric"}
        response = requests.get(FORECAST_API_URL, params=params)
        weather_data = response.json()

        if weather_data["cod"] == "200":
            forecast_date = datetime.strptime(date, "%Y-%m-%d")
            forecasts = weather_data["list"]
            forecast = None
            for entry in forecasts:
                forecast_time = datetime.fromtimestamp(entry["dt"], timezone.utc)
                if forecast_time.date() == forecast_date.date():
                    weather_description = entry["weather"][0]["description"]
                    temperature = entry["main"]["temp"]
                    forecast = f"The forecast for {location} on {date} is {weather_description} with a temperature of {temperature}°C."
                    break
            if forecast is None:
                forecast = f"No forecast data available for {location} on {date}."
        else:
            forecast = f"Could not retrieve forecast data for {location}. Response: {weather_data}"
            print(forecast)  # Log the response for debugging
    except Exception as e:
        forecast = f"An error occurred while fetching the forecast data: {str(e)}"
        print(forecast)  # Log the exception for debugging

    return forecast


def main():
    previous_question = ""
    repetition_count = 0
    repetition_limit = 3

    while True:
        transcription = recognize_speech_from_mic()
        if transcription:
            if transcription.lower() == previous_question.lower():
                repetition_count += 1
                if repetition_count >= repetition_limit:
                    print("Repetition limit reached. Exiting...")
                    break
            else:
                previous_question = transcription
                repetition_count = 0

            if "weather" in transcription.lower():
                locations, date = extract_locations_and_date(transcription)
                query_type = "single"
                if len(locations) > 1:
                    query_type = "comparison"

                forecast_data = {}
                if date:
                    forecast_date = datetime.strptime(date, "%Y-%m-%d").date()
                    today = datetime.today().date()
                    if forecast_date == today:
                        for location in locations:
                            forecast_data[location] = get_weather_forecast(location)
                    elif forecast_date > today:
                        for location in locations:
                            forecast_data[location] = get_weather_forecast_for_date(location, date)
                    else:
                        for location in locations:
                            forecast_data[location] = get_historical_weather(location, date)
                else:
                    for location in locations:
                        forecast_data[location] = get_weather_forecast(location)

                response = generate_response(forecast_data, query_type)
                print(response)
                engine.say(response)
                engine.runAndWait()
            elif "hourly forecast" in transcription.lower():
                locations, _ = extract_locations_and_date(transcription)
                for location in locations:
                    response = get_hourly_forecast(location)
                    print(response)
                    engine.say(response)
                    engine.runAndWait()
            elif "7-day forecast" in transcription.lower():
                locations, _ = extract_locations_and_date(transcription)
                for location in locations:
                    response = get_7_day_forecast(location)
                    print(response)
                    engine.say(response)
                    engine.runAndWait()
            elif "weather alerts" in transcription.lower():
                locations, _ = extract_locations_and_date(transcription)
                for location in locations:
                    response = get_weather_alerts(location)
                    print(response)
                    engine.say(response)
                    engine.runAndWait()
            elif "air quality" in transcription.lower():
                locations, _ = extract_locations_and_date(transcription)
                for location in locations:
                    response = get_air_quality(location)
                    print(response)
                    engine.say(response)
                    engine.runAndWait()
            elif "uv index" in transcription.lower():
                locations, _ = extract_locations_and_date(transcription)
                for location in locations:
                    response = get_uv_index(location)
                    print(response)
                    engine.say(response)
                    engine.runAndWait()
            elif "sunrise" in transcription.lower() or "sunset" in transcription.lower():
                locations, _ = extract_locations_and_date(transcription)
                for location in locations:
                    response = get_sunrise_sunset(location)
                    print(response)
                    engine.say(response)
                    engine.runAndWait()
            elif "current time" in transcription.lower():
                current_times = get_current_time_in_ist()
                response = "The current time in IST for the given cities:\n"
                for city, time in current_times.items():
                    response += f"{city}: {time}\n"
                print(response)
                engine.say(response)
                engine.runAndWait()
            elif "units of measurement" in transcription.lower():
                units = display_units_of_measurement()
                response = "The standard units of measurement used in India are:\n"
                for measure, unit in units.items():
                    response += f"{measure.capitalize()}: {unit}\n"
                print(response)
                engine.say(response)
                engine.runAndWait()
            elif "compare temperatures" in transcription.lower() or "highest temperature recorded today" in transcription.lower():
                locations, _ = extract_locations_and_date(transcription)
                if len(locations) == 2:
                    city1, city2 = locations
                    request_type = "max"
                    response = compare_temperatures(city1, city2, request_type)
                else:
                    response = "Please provide exactly two cities to compare temperatures."

                print(response)
                engine.say(response)
                engine.runAndWait()
            else:
                response = chat_session.send_message(f"Limit to 100 words \n {transcription}")
                engine.say(response.text)
                engine.runAndWait()
        else:
            print("No valid transcription. Exiting...")
            break

if __name__ == "__main__":
    main()
