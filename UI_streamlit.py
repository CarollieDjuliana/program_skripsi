import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
# from preprocessing import Preprocessor
from preprocessing import preprocessing_data
from preprocessing import preprocessing_data
from model import train_random_forest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.preprocessing import PolynomialFeatures
def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_page_config(page_title='Random Forest Model Tester', page_icon='🌳')
    st.title('Klasifikasi Menggunakan Random Forest')

    # Layout
    st.write("### Judul di atas kiri")

    # Sidebar
    status = "Utama"
    if st.sidebar.button("Halaman Utama"):
        status = "Utama"

    if st.sidebar.button("Informasi Data Training"):
        status = "Informasi Data Training"

    if status == "Utama":
        st.write("Selamat datang di halaman utama.")
        st.write("Anda bisa melakukan pengujian model di sini.")
        st.write("Silakan unggah data dan tekan tombol 'Process' untuk melanjutkan.")

        # Upload data
        input_file = st.file_uploader("Upload CSV (databaseUkt23_preprocessing.csv)", type=['csv'])

        if input_file is not None:
            preprocessed_data = preprocessing_data(input_file)
            # Print preview of the preprocessed data
            print("Preview data yang telah diproses:")
            print(preprocessed_data.head())

            X = preprocessed_data.drop(columns=['ukt_rev', 'no_test']).values
            y = preprocessed_data['ukt_rev'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            n_estimators = 300
            max_depth = None
            model = train_random_forest(X_train, y_train, n_estimators, max_depth)
            
            # 2. Feature Engineering dengan Polynomial Features dan Transformasi Fitur Non-linear
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            X_test = poly.fit_transform(X_test)

            # Evaluate the model on test set
            y_pred_test = model.predict(X_test)
            accuracy_test = accuracy_score(y_test, y_pred_test)
            confusion_mat_test = confusion_matrix(y_test, y_pred_test)
            recall_test = recall_score(y_test, y_pred_test, average='weighted')
            precision_test = precision_score(y_test, y_pred_test, average='weighted')
            f1_test = f1_score(y_test, y_pred_test, average='weighted')

           # Print evaluation result
            st.write("Confusion Matrix on Test Set:")
            st.write(confusion_mat_test)
            st.write("Accuracy on Test Set:", accuracy_test)
            st.write("Recall on Test Set:", recall_test)
            st.write("Precision on Test Set:", precision_test)
            st.write("F1-score on Test Set:", f1_test)

    elif status == "Informasi Data Training":
        st.write("Anda akan melihat informasi terkait data training di sini.")

        # Load data
        input_file = 'data_mentah.csv'
        data = preprocessing_data(input_file)

        # Show other information
        st.write("Informasi Data:")
        st.write("- Jumlah sampel:", len(data))
        st.write("- Jumlah atribut:", len(data.columns) - 1)  # Exclude target column
        st.write("- Distribusi Kelas:")
        st.write(data['ukt_rev'].value_counts())

        # Show correlation heatmap
        st.write("Correlation Heatmap:")
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        st.pyplot()

        # Visualisasi distribusi atribut numerik
        numerical_attributes = data.select_dtypes(include=['int64', 'float64']).columns
        for attribute in numerical_attributes:
            plt.figure(figsize=(8, 6))
            sns.histplot(data[attribute], bins=20, kde=True)
            plt.title(f'Distribusi {attribute}')
            plt.xlabel(attribute)
            plt.ylabel('Frekuensi')
            st.pyplot()

if __name__ == "__main__":
    main()
