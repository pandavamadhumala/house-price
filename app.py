import pandas as pd

# Load everything
model = joblib.load("house_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🏠 House Price Prediction")

# Take ALL inputs properly
area = st.number_input("Area")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")
stories = st.number_input("Stories")
parking = st.number_input("Parking")

# Example categorical
mainroad = st.selectbox("Main Road", ["yes", "no"])
guestroom = st.selectbox("Guest Room", ["yes", "no"])

if st.button("Predict"):

    input_dict = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "mainroad": mainroad,
        "guestroom": guestroom
    }

    input_df = pd.DataFrame([input_dict])

    # Apply get_dummies
    input_df = pd.get_dummies(input_df)

    # Match training columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    st.success(f"Estimated Price: ₹ {prediction[0]:,.2f}")
