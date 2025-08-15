import streamlit as st
import joblib

# Load model and preprocessing tools
model = joblib.load("air line passenge.pkl")  # fix filename if needed
scaler = joblib.load("scaler.pkl")
le = joblib.load("le.pkl")       # Gender encoder
le1 = joblib.load("le1.pkl")     # Customer Type encoder
le2 = joblib.load("le2.pkl")     # Type of Travel encoder
le3 = joblib.load("le3.pkl")     # Class encoder

st.title("✈️ Airline Passenger Satisfaction Prediction")

# Inputs
gender = st.selectbox("Gender", le.classes_.tolist())
customer_type = st.selectbox("Customer Type", le1.classes_.tolist())
travel_type = st.selectbox("Type of Travel", le2.classes_.tolist())
travel_class = st.selectbox("Class", le3.classes_.tolist())

age = st.number_input("Age", 0, 100)
flight_distance = st.number_input("Flight Distance", 0)

# Service ratings (scale: 0–5)
wifi = st.slider("Inflight Wifi Service", 0, 5)
departure_arrival = st.slider("Departure/Arrival Time Convenience", 0, 5)
ease_online = st.slider("Ease of Online Booking", 0, 5)
gate_location = st.slider("Gate Location", 0, 5)
food = st.slider("Food and Drink", 0, 5)
online_boarding = st.slider("Online Boarding", 0, 5)
seat_comfort = st.slider("Seat Comfort", 0, 5)
inflight_entertainment = st.slider("Inflight Entertainment", 0, 5)
onboard_service = st.slider("On-board Service", 0, 5)
leg_room = st.slider("Leg Room Service", 0, 5)
baggage = st.slider("Baggage Handling", 0, 5)
checkin_service = st.slider("Check-in Service", 0, 5)
cleanliness = st.slider("Cleanliness", 0, 5)

# Delay related features
departure_delay = st.number_input("Departure Delay (minutes)", 0)
arrival_delay = st.number_input("Arrival Delay (minutes)", 0)

# Additional feature if applicable (if you don't have this, set 0)
# Replace 'booking_changes' with actual feature name or comment out if not used
booking_changes = st.number_input("Booking Changes", 0, 10, 0)

# Encode categorical variables
gender_encoded = le.transform([gender])[0]
customer_type_encoded = le1.transform([customer_type])[0]
travel_type_encoded = le2.transform([travel_type])[0]
class_encoded = le3.transform([travel_class])[0]

# Feature vector in correct order (22 features)
features = [[
    gender_encoded,
    customer_type_encoded,
    travel_type_encoded,
    class_encoded,
    age,
    flight_distance,
    wifi,
    departure_arrival,
    ease_online,
    gate_location,
    food,
    online_boarding,
    seat_comfort,
    inflight_entertainment,
    onboard_service,
    leg_room,
    baggage,
    checkin_service,
    cleanliness,
    departure_delay,
    arrival_delay,
    booking_changes
]]

# Scale features
scaled_input = scaler.transform(features)

# Predict
if st.button("Predict Satisfaction"):
    result = model.predict(scaled_input)[0]
    if result == 1:
        st.success("Passenger is Satisfied")
    else:
        st.error(" Passenger is Not Satisfied")
