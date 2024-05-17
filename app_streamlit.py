import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preparation import preparation
from prediction import prediction
import time
import tabulate


class app:
    def main():

        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.set_page_config(
            page_title='Random Forest Model Tester')
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image('logo_unsri.png', width=65,
                     use_column_width='always', output_format='auto')
        with col2:
            st.markdown(
                "<h2 style='text-align: center;'>Klasifikasi Golongan Uang Kuliah Tunggal Menggunakan Random Forest</h2>", unsafe_allow_html=True)
        status = "Utama"
        if status == "Utama":
            st.write("")
            st.write("Anda bisa melakukan pengujian model di sini.")
            st.write(
                "Silakan unggah data dan atur parameter model menggunakan slider di bawah.")
            input_file = st.file_uploader(
                "Upload CSV (data verifikasi ukt 2023).csv)", type=['csv'])

            if input_file is not None:
                n_estimators_options = [5, 10, 15, 20, 25, 30, 35, 40, 50]
                default_n_estimators = 30
                n_estimators = st.select_slider(
                    "Jumlah n_estimators (pohon):", options=n_estimators_options, value=default_n_estimators)

                max_depth_options = [None, 5, 10, 15, 20, 25, 30, 35, 40, 50]
                default_max_depth = 5
                max_depth = st.select_slider(
                    "Jumlah max_depth (kedalaman):", options=max_depth_options, value=default_max_depth)

                test_size_options = [0.1, 0.2, 0.3,
                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                default_test_size = 0.2
                test_size = st.select_slider(
                    "Jumlah test_size:", options=test_size_options, value=default_test_size)
                data = pd.read_csv(
                    input_file, delimiter=',', encoding='latin-1')
                st.write("Preview data :")
                data.index += 1
                st.write(data.head(5), index=False)
                if st.button("Process"):
                    prediction.prediction(
                        data, n_estimators, max_depth, test_size)

    if __name__ == "__main__":
        main()
