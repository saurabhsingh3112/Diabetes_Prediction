# Diabetes Prediction using Support Vector Machine (SVM)

![Diabetes Prediction](link-to-image)

## Overview
This project aims to predict diabetes in individuals using a Support Vector Machine (SVM) classifier. The dataset contains various health-related features for individuals, along with their diabetes outcome (0 for non-diabetic and 1 for diabetic). The SVM model is trained on this dataset to predict the likelihood of an individual being diabetic based on the provided features.

## Project Setup
1. Install the required libraries: Make sure you have the necessary Python libraries like NumPy, Pandas, scikit-learn installed. You can install them using pip: `pip install numpy pandas scikit-learn`

2. Load the Dataset: Download the diabetes dataset, which should be in CSV format, and load it into a pandas DataFrame.
   DATASET LINK :- https://www.dropbox.com/s/uh7o7uyeghqkhoy/diabetes.csv?dl=0
3. Data Exploration: Explore the dataset by printing the first few rows, checking the shape, and getting statistical measures of the data.

4. Data Preprocessing: Separate the data into features (X) and labels (Y) and perform standardization using the `StandardScaler` from scikit-learn.

5. Model Training: Split the data into training and testing sets. Train the SVM classifier using a linear kernel on the training data.

6. Model Evaluation: Calculate the accuracy score of the trained model on both the training and testing data.

7. Making Predictions: Provide new input data to the model and predict whether the individual is diabetic or not.

## Usage
1. Import Required Libraries: Ensure you have the necessary libraries imported at the beginning of your code.

2. Load the Diabetes Dataset: Load the diabetes dataset from a CSV file into a pandas DataFrame.

3. Data Preprocessing: Separate the data into features (X) and labels (Y), and perform standardization on the features.

4. Model Training and Evaluation: Split the data into training and testing sets. Train the SVM classifier using a linear kernel and evaluate its accuracy on both the training and testing data.

5. Diabetes Prediction: To predict whether an individual is diabetic or not, provide new input data as a tuple, and the model will make the prediction.

## Example Code
python
# ... (Include the code for loading the diabetes dataset and data preprocessing as provided)

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Training the SVM classifier with a linear kernel
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Accuracy score on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score on the training data: ', training_data_accuracy)

# Accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on the test data: ', test_data_accuracy)

# New input data for prediction
input_data = (2, 197, 70, 45, 543, 30.5, 0.158, 53)

# Standardize the input data and make the prediction
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')


*Note:* Before running the project, ensure that you have the diabetes dataset in CSV format and modify the file path accordingly. The project assumes that you have already installed the necessary libraries and have Python installed on your system.

## Getting Started
To get started with this Diabetes Prediction project, follow these steps:

1. Clone the repository to your local machine.

2. Install the required libraries by running `pip install -r requirements.txt`.

3. Download the diabetes dataset in CSV format and load it into a pandas DataFrame.

4. Run the provided code to preprocess the data, train the SVM classifier, and make predictions.

Feel free to explore the code, modify it to suit your preferences, and have fun predicting diabetes probabilities!

## Acknowledgments
Special thanks to the creators of the diabetes dataset for providing the data used in this project, and to the scikit-learn team for enabling machine learning capabilities in Python.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
If you have any questions or suggestions regarding this project, feel free to contact me at [saurabhsinghcse3112@gmail.com]

Stay healthy and proactive! 
