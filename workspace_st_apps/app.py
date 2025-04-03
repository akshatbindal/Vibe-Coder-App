import streamlit as st
import datetime

st.set_page_config(page_title="Flight Booking", layout="centered")

st.title("✈️ Flight Booking")

# --- Input Fields ---
col1, col2 = st.columns(2)
with col1:
    origin = st.text_input("From", placeholder="e.g., New York")
with col2:
    destination = st.text_input("To", placeholder="e.g., London")

departure_date = st.date_input(
    "Departure Date",
    min_value=datetime.date.today()
)

# --- Search Button ---
if st.button("Search Flights"):
    if origin and destination and departure_date:
        st.success(f"Searching for flights from {origin} to {destination} on {departure_date.strftime('%Y-%m-%d')}...")
        # In a real app, you would query an API here
        st.write("---")
        st.subheader("Available Flights (Example)")
        st.info("Flight XYZ123 - 08:00 AM - $350")
        st.info("Flight ABC789 - 01:30 PM - $420")
        st.info("Flight QWE456 - 06:00 PM - $390")
        st.write("*(These are sample results)*")
    else:
        st.warning("Please fill in all fields (Origin, Destination, Date).")

st.caption("A basic flight booking simulation.")
