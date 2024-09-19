import streamlit as st
import joblib
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names, but.*")

# home page
st.set_page_config(
    page_title="Smart Crop",
    page_icon="logo.webp",
    layout="centered",
)

# our model
model = None

def load_model():
    global model
    with open("model.pkl", "rb") as f:
        model = joblib.load(f)


def main():

    html_text = """
    <style>
    p {font-size :17px;}
    .block-container {padding: 2rem 1rem 3rem;}
    #MainMenu {visibility: hidden;}
    </style>
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:center;"> SmartCrop: Crop Recommendation</h1>
    </div>
    """
    
    st.markdown(html_text, unsafe_allow_html=True)
   
    st.subheader("Find out the most suitable crop to grow in your Farm")

    # feature that we want from user
    N = st.number_input("Nitrogen", 1, 10000)
    P = st.number_input("Phosphorus", 1, 10000)
    K = st.number_input("Potassium", 1, 10000)
    temp = st.number_input("Temperature", 0.0, 100000.0)
    humidity = st.number_input("Humidity", 0.0, 100000.0)
    ph = st.number_input("Ph", 0.0, 100000.0)
    rainfall = st.number_input("Rainfall in mm", 0.0, 100000.0)

    feature = np.array([[N, P, K, temp, humidity, ph, rainfall]]).reshape(1, -1)

    # predication and display ans
    if st.button("Predict"):
        if model:
            crop = model.predict(feature)
            st.write("## Result")
            st.success(f"{crop[0]} will be  Best For Your Farm This Time.")

            # to download your information
            st.subheader("Input Features:")
            st.write(
                pd.DataFrame(
                    {
                        "Freature": [
                            "Nitrogen",
                            "Phosphorus",
                            "Potassium",
                            "Temperature",
                            "Humidity",
                            "PH",
                            "Rainfall",
                        ],
                        "Entered Value": feature[0],
                    }
                )
            )


if __name__ == "__main__":
    load_model()
    main()
