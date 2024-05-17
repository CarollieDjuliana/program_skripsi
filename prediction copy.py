import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from preparation import preparation
from model import model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from io import BytesIO


class prediction:
    def prediction(data, n_estimators, max_depth, test_size):
        preprocessed_data = preparation.prepraration_data(data)
        # Print preview of the preprocessed data
        X = preprocessed_data.drop(columns=['ukt']).values
        y = preprocessed_data['ukt'].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=34)

        # Train the model
        model_train = model.train_random_forest(
            X_train, y_train, n_estimators, max_depth)

        # Evaluate the model on test set
        y_pred_test = model_train.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        confusion_mat_test = confusion_matrix(y_test, y_pred_test)
        recall_test = recall_score(y_test, y_pred_test, average='weighted')
        precision_test = precision_score(
            y_test, y_pred_test, average='weighted')
        f1_test = f1_score(y_test, y_pred_test, average='weighted')

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(confusion_mat_test, cmap='Oranges')
        # Menambahkan label untuk setiap sel pada confusion matrix
        for i in range(confusion_mat_test.shape[0]):
            for j in range(confusion_mat_test.shape[1]):
                text = ax.text(
                    j, i, confusion_mat_test[i, j], ha='center', va='center')

        ax.set_xticklabels(['', 'G.UKT 1', 'G.UKT 2', 'G.UKT 3',
                           'G.UKT 4', 'G.UKT 5', 'G.UKT 6', 'G.UKT 7', 'G.UKT 8'])
        ax.set_yticklabels(['', 'G.UKT 1', 'G.UKT 2', 'G.UKT 3',
                           'G.UKT 4', 'G.UKT 5', 'G.UKT 6', 'G.UKT 7', 'G.UKT 8'])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        cbar = ax.figure.colorbar(im, ax=ax)
        # st.pyplot(fig)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=80, facecolor='#FFEC9E')
        st.image(buf)
        print(confusion_mat_test)
        st.write("Accuracy on Test Set:", accuracy_test)
        st.write("Precision on Test Set:", precision_test)
        st.write("Recall on Test Set:", recall_test)
        st.write("F1-score on Test Set:", f1_test)
