import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
# from preprocessing import Preprocessor
from preparation import prepraration_data
from model import train_random_forest, preprocessing
from prediction import prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.preprocessing import PolynomialFeatures
from PIL import Image
from io import BytesIO


def main():

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_page_config(page_title='Random Forest Model Tester', page_icon='🌳')
    st.title('Klasifikasi Golongan Uang Kuliah Tunggal Menggunakan Random Forest')
    st.sidebar.image('logo_unsri.png', width=100)

    status = "Utama"
    if st.sidebar.button("Halaman Utama"):
        status = "Utama"

    if st.sidebar.button("Informasi Mengenai Dataset"):
        status = "Informasi Mengenai Dataset"

    if status == "Utama":
        st.write("Anda bisa melakukan pengujian model di sini.")
        st.write(
            "Silakan unggah data dan atur parameter model menggunakan slider di bawah.")
        input_file = st.file_uploader(
            "Upload CSV (data verifikasi ukt 2023).csv)", type=['csv'])

        if input_file is not None:
            n_estimators_options = [100, 150, 200, 250, 290, 300, 350, 400]
            default_n_estimators = 300
            n_estimators = st.select_slider(
                "Jumlah n_estimators:", options=n_estimators_options, value=default_n_estimators)

            max_depth_options = [None, 10, 20, 30, 35, 40, 45, 50, 60, 70, 80]
            default_max_depth = None
            max_depth = st.select_slider(
                "Jumlah max_depth:", options=max_depth_options, value=default_max_depth)

            test_size_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            default_test_size = 0.2
            test_size = st.select_slider(
                "Jumlah test_size:", options=test_size_options, value=default_test_size)
            data = pd.read_csv(input_file, delimiter=',', encoding='latin-1')
            st.write("Data Awal sebelum Preprocessing:")
            st.write(data.head())
            if st.button("Process"):
                prediction(data, n_estimators, max_depth, test_size)

    elif status == "Informasi Mengenai Dataset":
        st.write("Anda akan melihat informasi terkait data training di sini.")
        # Load data
        input_file = pd.read_csv(
            'data_mentah.csv', delimiter=',', encoding='latin-1')
        # Show other information
        st.write("Informasi Data:")
        st.write("- Jumlah sampel:", len(input_file))
        # Exclude target column
        st.write("- Jumlah atribut:", len(input_file.columns) - 1)
        st.write("- Distribusi Kelas:")
        st.write(input_file['ukt_rev'].value_counts())

        st.write(input_file.head())

        st.write("data setelah preparation")
        data = prepraration_data(input_file)
        st.write(data.head())

        # Show correlation heatmap
        st.write("Correlation Heatmap:")
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        st.pyplot()


if __name__ == "__main__":
    main()
