import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
# from preprocessing import Preprocessor
from preparation import prepraration_data
from model import train_random_forest
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
        st.write("Silakan unggah data dan atur parameter model menggunakan slider di bawah.")
        input_file = st.file_uploader("Upload CSV (data verifikasi ukt 2023).csv)", type=['csv'])
        
        if input_file is not None:
            n_estimators_options = [100, 150, 200, 250,290, 300, 350, 400]
            default_n_estimators = 300
            n_estimators = st.select_slider("Jumlah n_estimators:", options=n_estimators_options, value=default_n_estimators)

            max_depth_options = [None, 10, 20, 30, 35, 40, 45, 50, 60, 70, 80]
            default_max_depth = None
            max_depth = st.select_slider("Jumlah max_depth:", options=max_depth_options, value=default_max_depth)

            test_size_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            default_test_size = 0.2
            test_size = st.select_slider("Jumlah test_size:", options=test_size_options, value=default_test_size)
            data = pd.read_csv(input_file, delimiter=',', encoding='latin-1')
            st.write("Data Awal sebelum Preprocessing:")
            st.write(data.head())
            if st.button("Process"):
                preprocessed_data = prepraration_data(data)
                # Print preview of the preprocessed data    
                X = preprocessed_data.drop(columns=['ukt_rev', 'no_test']).values
                y = preprocessed_data['ukt_rev'].values
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=33)

                # Train the model
                model = train_random_forest(X_train, y_train, n_estimators, max_depth)
                
                # Feature Engineering dengan Polynomial Features 
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                X_test_poly = poly.fit_transform(X_test)

                # Evaluate the model on test set
                y_pred_test = model.predict(X_test_poly)
                accuracy_test = accuracy_score(y_test, y_pred_test)
                confusion_mat_test = confusion_matrix(y_test, y_pred_test)
                recall_test = recall_score(y_test, y_pred_test, average='weighted')
                precision_test = precision_score(y_test, y_pred_test, average='weighted')
                f1_test = f1_score(y_test, y_pred_test, average='weighted')

                fig, ax = plt.subplots(figsize=(10,10))
                im = ax.imshow(confusion_mat_test, cmap='Oranges')
                # Menambahkan label untuk setiap sel pada confusion matrix
                for i in range(confusion_mat_test.shape[0]):
                    for j in range(confusion_mat_test.shape[1]):
                        text = ax.text(j, i, confusion_mat_test[i, j], ha='center', va='center')

                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                ax.set_title('Confusion Matrix')
                cbar = ax.figure.colorbar(im, ax=ax)
                # st.pyplot(fig)
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=80, facecolor='#FFEC9E')
                st.image(buf)

                st.write("Recall on Test Set:", recall_test)
                st.write("Accuracy on Test Set:", accuracy_test)
                st.write("Precision on Test Set:", precision_test)
                st.write("F1-score on Test Set:", f1_test)


    elif status == "Informasi Mengenai Dataset":
        st.write("Anda akan melihat informasi terkait data training di sini.")
        # Load data
        input_file = pd.read_csv('data_mentah.csv', delimiter=',', encoding='latin-1')
        # Show other information
        st.write("Informasi Data:")
        st.write("- Jumlah sampel:", len(input_file))
        st.write("- Jumlah atribut:", len(input_file.columns) - 1)  # Exclude target column
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
