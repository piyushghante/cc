# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, Lasso, Ridge
# from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
# from sklearn.mixture import GaussianMixture
# from streamlit.logger import get_logger
# from sklearn.metrics import confusion_matrix


# LOGGER = get_logger(__name__)

# def perform_regression(X_train, y_train, model_name):
#     # Function to perform regression
#     if model_name == "Linear Regression":
#         model = LinearRegression()
#     elif model_name == "Lasso Regression":
#         model = Lasso()
#     elif model_name == "Ridge Regression":
#         model = Ridge()
#     elif model_name == "DT Regression":
#         model = DecisionTreeRegressor()
#     else:
#         return None

#     y_train = y_train.values.ravel()

#     try:
#         model.fit(X_train, y_train)
#         return model
#     except Exception as e:
#         st.error(f"Error during model training: {e}")
#         return None

# def perform_classification(X_train, y_train, model_name):
#     # Function to perform classification
#     if model_name == "Naive Bayes":
#         model = GaussianNB()
#     elif model_name == "Decision Tree":
#         model = DecisionTreeClassifier()
#     elif model_name == "SVM":
#         model = SVC()
#     elif model_name == "Random Forest":
#         model = RandomForestClassifier()
#     elif model_name == "K-Nearest Neighbors":
#         model = KNeighborsClassifier()
#     else:
#         return None

#     model.fit(X_train, y_train)
#     return model

# def perform_clustering(X, model_name):
#     # Function to perform clustering
#     if model_name == "K-Means":
#         model = KMeans()
#     elif model_name == "DBSCAN":
#         model = DBSCAN()
#     elif model_name == "Hierarchical":
#         model = AgglomerativeClustering()
#     elif model_name == "Gaussian Mixture":
#         model = GaussianMixture()
#     else:
#         return None

#     model.fit(X)
#     return model

# def visualize_regression_results(model, X_test, y_test):
#     # Function to visualize regression results
#     predictions = model.predict(X_test)

#     plt.scatter(y_test, predictions)
#     plt.xlabel("Actual Values")
#     plt.ylabel("Predicted Values")
#     plt.title("Regression Model: Actual vs. Predicted Values")
#     st.pyplot()

#     residuals = y_test - predictions
#     plt.scatter(predictions, residuals)
#     plt.xlabel("Predicted Values")
#     plt.ylabel("Residuals")
#     plt.title("Regression Model: Residuals Plot")
#     st.pyplot()

# def visualize_classification_results(model, X_test, y_test):
#     # Function to visualize classification results
#     cm = confusion_matrix(y_test, model.predict(X_test))
#     st.write("Confusion Matrix:")
#     st.write(cm)

#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.title("Classification Model: Confusion Matrix")
#     st.pyplot()

# def visualize_clustering_results(model, X):
#     # Function to visualize clustering results
#     labels = model.labels_
#     plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis')
#     plt.xlabel("Feature 1")
#     plt.ylabel("Feature 2")
#     plt.title("Clustering Model: Scatter Plot of Clusters")
#     st.pyplot()

# def run():
#     st.set_page_config(
#         page_title="Hello",
#         page_icon="👋",
#     )
#     st.title("Machine Learning Model Selector")

#     uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

#     if uploaded_file is not None:
#         st.sidebar.header("Choose a Model:")
#         model_type = st.sidebar.selectbox("Select a model type", ["Regression", "Classification", "Clustering"])

#         df = pd.read_csv(uploaded_file)
#         st.dataframe(df.head())

#         features = st.multiselect("Select features", df.columns)
#         target_variable = st.selectbox("Select target variable", df.columns)

#         X = df[features]
#         y = df[target_variable]

#         if model_type == "Regression":
#             model_name = st.sidebar.selectbox("Select a regression model", ["Linear Regression", "Lasso Regression", "Ridge Regression", "DT Regression"])
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#             model = perform_regression(X_train, y_train, model_name)
#             if model is not None:
#                 st.write("Regression Model Results:")
#                 st.write("Model Score:", model.score(X_test, y_test))
#                 visualize_regression_results(model, X_test, y_test)

#         elif model_type == "Classification":
#             model_name = st.sidebar.selectbox("Select a classification model", ["Naive Bayes", "Decision Tree", "SVM", "Random Forest", "K-Nearest Neighbors"])
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#             model = perform_classification(X_train, y_train, model_name)
#             if model is not None:
#                 st.write("Classification Model Results:")
#                 st.write("Model Score:", model.score(X_test, y_test))
#                 visualize_classification_results(model, X_test, y_test)

#         elif model_type == "Clustering":
#             model_name = st.sidebar.selectbox("Select a clustering model", ["K-Means", "DBSCAN", "Hierarchical", "Gaussian Mixture"])
#             model = perform_clustering(X, model_name)
#             if model is not None:
#                 st.write("Clustering Model Results:")
#                 visualize_clustering_results(model, X)

# if __name__ == "__main__":
#     run()


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from streamlit.logger import get_logger
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score


LOGGER = get_logger(__name__)

# Apply deprecation warning suppression
st.set_option('deprecation.showPyplotGlobalUse', False)

def perform_regression(X_train, y_train, model_name):
    # Function to perform regression
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Lasso Regression":
        model = Lasso()
    elif model_name == "Ridge Regression":
        model = Ridge()
    elif model_name == "DT Regression":
        model = DecisionTreeRegressor()
    else:
        return None

    y_train = y_train.values.ravel()

    try:
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        st.error(f"Error during model training: {e}")
        return None

def perform_classification(X_train, y_train, model_name):
    # Function to perform classification
    if model_name == "Naive Bayes":
        model = GaussianNB()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "SVM":
        model = SVC()
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "K-Nearest Neighbors":
        model = KNeighborsClassifier()
    else:
        return None

    model.fit(X_train, y_train)
    return model

def perform_clustering(X, model_name):
    # Function to perform clustering
    if model_name == "K-Means":
        model = KMeans()
    elif model_name == "DBSCAN":
        model = DBSCAN()
    elif model_name == "Hierarchical":
        model = AgglomerativeClustering()
    elif model_name == "Gaussian Mixture":
        model = GaussianMixture()
    else:
        return None

    model.fit(X)
    return model

def visualize_regression_results(model, X_test, y_test):
    # Function to visualize regression results
    predictions = model.predict(X_test)

    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Regression Model: Actual vs. Predicted Values")
    st.pyplot()

    residuals = y_test - predictions
    plt.scatter(predictions, residuals)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Regression Model: Residuals Plot")
    st.pyplot()

    # Additional: Calculate and display mean squared error
    mse = mean_squared_error(y_test, predictions)
    st.write("Mean Squared Error:", mse)

def visualize_classification_results(model, X_test, y_test):
    # Function to visualize classification results
    predictions = model.predict(X_test)

    st.write("Accuracy:", accuracy_score(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)
    st.write("Confusion Matrix:")
    st.write(cm)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Classification Model: Confusion Matrix")
    st.pyplot()

    # Additional: Calculate and display precision, recall, and F1-score
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    st.write("Precision:", precision)
    st.write("Recall:", recall)
    st.write("F1 Score:", f1)

def visualize_clustering_results(model, X):
    # Function to visualize clustering results
    labels = model.labels_
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Clustering Model: Scatter Plot of Clusters")
    st.pyplot()

def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )
    st.title("Machine Learning Model Selector")

    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

    if uploaded_file is not None:
        st.sidebar.header("Choose a Model:")
        model_type = st.sidebar.selectbox("Select a model type", ["Regression", "Classification", "Clustering"])

        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        features = st.multiselect("Select features", df.columns)
        target_variable = st.selectbox("Select target variable", df.columns)

        X = df[features]
        y = df[target_variable]

        if model_type == "Regression":
            model_name = st.sidebar.selectbox("Select a regression model", ["Linear Regression", "Lasso Regression", "Ridge Regression", "DT Regression"])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = perform_regression(X_train, y_train, model_name)
            if model is not None:
                st.write("Regression Model Results:")
                st.write("Model Score:", model.score(X_test, y_test))
                visualize_regression_results(model, X_test, y_test)

        elif model_type == "Classification":
            model_name = st.sidebar.selectbox("Select a classification model", ["Naive Bayes", "Decision Tree", "SVM", "Random Forest", "K-Nearest Neighbors"])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = perform_classification(X_train, y_train, model_name)
            if model is not None:
                st.write("Classification Model Results:")
                st.write("Model Score:", model.score(X_test, y_test))
                visualize_classification_results(model, X_test, y_test)

        elif model_type == "Clustering":
            model_name = st.sidebar.selectbox("Select a clustering model", ["K-Means", "DBSCAN", "Hierarchical", "Gaussian Mixture"])
            model = perform_clustering(X, model_name)
            if model is not None:
                st.write("Clustering Model Results:")
                visualize_clustering_results(model, X)

if __name__ == "__main__":
    run()
