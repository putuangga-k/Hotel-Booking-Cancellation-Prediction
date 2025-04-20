import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

st.set_page_config(page_title="Hotel Cancellation Predictor", layout="centered")

# --- Load trained pipeline ---
@st.cache_resource
def load_model(path: str = 'best_xgb_model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

model_pipeline = load_model()

# Page layout
st.title("Hotel Booking Cancellation Predictor")
st.markdown(
    """
    Fill in the booking details below and click on **Predict** to see 
    the probability of cancellation.
    """
)

# User inputs 
no_of_adults = st.number_input("Number of adults", min_value=0, max_value=10, value=2)
no_of_children = st.number_input("Number of children", min_value=0, max_value=10, value=0)

no_of_weekend_nights = st.number_input("Weekend nights", min_value=0, max_value=7, value=1)
no_of_week_nights = st.number_input("Week nights", min_value=0, max_value=30, value=2)

type_of_meal_plan = st.selectbox(
    "Meal plan",
    ["Meal Plan 1", "Not Selected", "Meal Plan 2", "Meal Plan 3"]
)

required_car_parking_space = st.number_input(
    "Required parking spaces", min_value=0, max_value=5, value=0
)

room_type_reserved = st.selectbox(
    "Room type reserved",
    ["Room_Type 1", "Room_Type 2", "Room_Type 3",
     "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"]
)

lead_time = st.number_input("Lead time (days)", min_value=0, max_value=500, value=100)

arrival_date = st.date_input("Arrival date", value=datetime.today())

market_segment_type = st.selectbox(
    "Market segment",
    ["Online", "Offline", "Corporate", "Complementary", "Aviation"]
)

repeated_guest = st.checkbox("Repeated guest")

no_of_previous_cancellations = st.number_input(
    "Previous cancellations", min_value=0, max_value=50, value=0
)
no_of_previous_bookings_not_canceled = st.number_input(
    "Previous bookings not canceled", min_value=0, max_value=50, value=1
)

avg_price_per_room = st.number_input(
    "Average price per room",
    min_value=0.0, max_value=10000.0, value=100.0, format="%.2f"
)

no_of_special_requests = st.number_input(
    "Number of special requests", min_value=0, max_value=10, value=0
)

# --- Prediction ---
if st.button("Predict"):
    # Feature engineering
    total_nights = no_of_weekend_nights + no_of_week_nights
    total_guests = no_of_adults + no_of_children
    arrival_month = arrival_date.month
    arrival_weekday = arrival_date.weekday()

    # Build DataFrame for model
    input_df = pd.DataFrame([{
        'no_of_adults': no_of_adults,
        'no_of_children': no_of_children,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'type_of_meal_plan': type_of_meal_plan,
        'required_car_parking_space': required_car_parking_space,
        'room_type_reserved': room_type_reserved,
        'lead_time': lead_time,
        'arrival_month': arrival_month,
        'market_segment_type': market_segment_type,
        'repeated_guest': int(repeated_guest),
        'no_of_previous_cancellations': no_of_previous_cancellations,
        'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
        'avg_price_per_room': avg_price_per_room,
        'no_of_special_requests': no_of_special_requests,
        'total_nights': total_nights,
        'total_guests': total_guests,
        'arrival_weekday': arrival_weekday
    }])

    # Generate prediction
    proba = model_pipeline.predict_proba(input_df)[0, 1]
    pred = model_pipeline.predict(input_df)[0]

    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Cancellation Probability:** {proba:.2%}")
    if pred == 1:
        st.error("The booking is likely to be **Canceled**.")
    else:
        st.success("The booking is likely **Not Canceled**.")