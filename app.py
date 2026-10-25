
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

st.set_page_config(
    page_title="Airline Satisfaction",
    page_icon="✈️"
)


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        "airline_ann_model1.keras"
    )

    preprocessor = joblib.load(
        "airline_preprocessor1.pkl"
    )

    return model, preprocessor


model, preprocessor = load_artifacts()

st.title("✈️ Airline Passenger Satisfaction")
st.write("Enter passenger details to predict satisfaction.")

if not hasattr(preprocessor, "feature_names_in_"):
    st.error(
        "Original feature names not found in the preprocessor. "
        "Check your training notebook."
    )
    st.stop()

feature_names = list(preprocessor.feature_names_in_)

categories = {
    "Gender": ["Male", "Female"],
    "Customer Type": [
        "Loyal Customer",
        "disloyal Customer"
    ],
    "Type of Travel": [
        "Personal Travel",
        "Business travel"
    ],
    "Class": ["Eco", "Eco Plus", "Business"]
}

rating_words = [
    "Convenience",
    "Booking",
    "Service",
    "Boarding",
    "Gate Location",
    "Comfort",
    "Leg Room",
    "Cleanliness",
    "Food and Drink",
    "Entertainment",
    "Wifi",
    "Baggage Handling"
]

with st.form("passenger_form"):
    values = {}

    for col in feature_names:

        if col in categories:
            values[col] = st.selectbox(
                col, categories[col]
            )

        elif col == "Age":
            values[col] = st.number_input(
                col,
                min_value=1,
                max_value=110,
                value=30
            )

        elif col == "Flight Distance":
            values[col] = st.number_input(
                col,
                min_value=0,
                value=500
            )

        elif "Delay" in col:
            values[col] = st.number_input(
                col,
                min_value=0.0,
                value=0.0
            )

        elif any(
            word.lower() in col.lower()
            for word in rating_words
        ):
            values[col] = st.slider(
                col,
                min_value=0,
                max_value=5,
                value=3
            )

        else:
            values[col] = st.number_input(
                col,
                value=0.0
            )

    submitted = st.form_submit_button(
        "Predict Satisfaction"
    )


if submitted:
    try:
        input_df = pd.DataFrame(
            [values],
            columns=feature_names
        )

        processed = preprocessor.transform(input_df)

        if hasattr(processed, "toarray"):
            processed = processed.toarray()

        processed = np.asarray(
            processed,
            dtype=np.float32
        )

        prediction = model.predict(
            processed,
            verbose=0
        )

        score = float(
            np.asarray(prediction).reshape(-1)[0]
        )

        st.subheader("Prediction Result")
        st.metric(
            "Satisfaction score",
            f"{score:.2%}"
        )

        if score >= 0.5:
            st.success("Predicted: Satisfied")
        else:
            st.warning(
                "Predicted: Neutral or Dissatisfied"
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
