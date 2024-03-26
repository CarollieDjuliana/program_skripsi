import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv("databaseUkt2023_preprocessed2.csv")
    return data

# Train the Random Forest model
def train_model(data):
    X = data.drop(["no_test","ukt"], axis=1)
    y = data["ukt"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier()
    model.fit(X_scaled, y)
    return model, scaler

# Main function
def main():
    st.title("Random Forest Classifier")
    st.write("Aplikasi ini memprediksi variabel target menggunakan Random Forest.")

    # Load the data
    data = load_data()

    # Train the model
    model, scaler = train_model(data)

    # Get user input for prediction
    no_test = st.number_input("No Test")
    fakultas = st.number_input("Fakultas")
    pekerjaan_ayah = st.number_input("Pekerjaan Ayah")
    pekerjaan_ibu = st.number_input("Pekerjaan Ibu")
    penghasilan_ayah = st.number_input("Penghasilan Ayah")
    penghasilan_ibu = st.number_input("Penghasilan Ibu")
    total_pendapatan = st.number_input("Total Pendapatan")
    pendapatan_class = st.number_input("Pendapatan Class")
    jumlah_tanggungan = st.number_input("Jumlah Tanggungan")
    kepemilikan_rumah = st.number_input("Kepemilikan Rumah")
    kendaraan = st.number_input("Kendaraan")
    sekolah = st.number_input("Sekolah")
    listrik = st.number_input("Listrik")
    penghasilan = st.number_input("Penghasilan")
    pajak_mobil = st.number_input("Pajak Mobil")
    pajak_motor = st.number_input("Pajak Motor")

    # Normalize user input
    user_input = scaler.transform([[fakultas, pekerjaan_ayah, pekerjaan_ibu, penghasilan_ayah, penghasilan_ibu, total_pendapatan, pendapatan_class, jumlah_tanggungan, kepemilikan_rumah, kendaraan, sekolah, listrik, penghasilan, pajak_mobil, pajak_motor]])

    # Make prediction
    prediction = model.predict(user_input)

    # Display the prediction
    st.write("Prediksi:", prediction[0])

if __name__ == "__main__":
    main()
