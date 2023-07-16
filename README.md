# Application-of-ML-based-Diabetes-Prediction-model

# INTRO 

Diabetes is a chronic metabolic disorder affecting millions of people worldwide.
Early detection and accurate prediction of diabetes can play a crucial role in
decision-making for handling and prevention of complications. This project aims
to develop a machine learning model for diabetes prediction that can accurately
predict the likelihood of a person having diabetes based on various clinical
features.
In this project, the dataset is collected, from healthcare databases, surveys, and medical
records, on considering the relevance of patients having diabetes or not. Data
cleaning techniques are playing an important role in cleaning and normalizing the
collected data, by removing any inconsistencies or missing values.
Next, feature extraction methods are utilized to identify the most significant
predictors of diabetes from the available dataset.
Machine learning algorithms are trained and evaluated to build the system.

# Block Diagram
<img width="640" alt="blockdiagram" src="https://github.com/Amarjeet2909/Application-of-ML-based-Diabetes-Prediction-model/assets/78557124/350004d5-c696-49fa-80e7-2ed2f6c325ab">

# Objectives 

The following are the objectives of our project:
• The goal of this project is to develop a machine-learning model that can accurately predict the
likelihood of a person having diabetes based on various clinical features.
• The main objective of this project is to provide accurate predictions which can further be used
• The proposed solution for ML-based diabetes prediction presents cost-effective, reliable and
technically simple solution
• This approach believes that by using Machine Learning Humans can be benefited and their
life can be saved.

# Proposed Mechanism

Studied various machine learning algorithms and to train the model and then finding the
best model suitable for predicting the diabetic nature in a patient by using the healthcare data of
particular patient.
Following are few points highlighting the methodology:
• Dataset has been taken from Kaggle containing the values like Glucose level, Pregnancies,
Blood Pressure, Skin Thickness, Insulin, BMI and age.
• Preprocessing of data before going through the model has been done.
• 80% data taken for training and 20% data taken for testing.
• Training model based on concept like SVM, random forest.
• Various machine learning algorithms will be compared, we have trained model based on
SVM and Logistic regression like that.
• Evaluating the model whether it is right or not.

# Algorithm Used

There are four different machine learning algorithms is used in this project and they are support
vector machine, Decision Tree, Random Forest and Naïve Bayes. These four machine learning
classifiers is used and compared based on different parameters to select best one to predict
diabetes.
• The SVM model utilized was configured with a linear kernel. A linear kernel creates a
linear decision boundary between classes, assuming the data is linearly separable. Then
we divided the dataset into training and testing sets. The training data were used to train
the SVM model, while the testing data is used to evaluate its performance. Prior to
training the SVM model, we applied feature scaling to the input data using the
StandardScaler from the sklearn.preprocessing module. Scaling the data helps in
normalizing the features and ensuring that they have similar ranges, which can improve
the performance of the SVM algorithm.
• We utilized the DecisionTreeClassifier from the sklearn.tree module. This classifier is
specifically designed for classification tasks and uses the Decision Tree algorithm to
create a tree-like model of decisions and their possible consequences. The Decision Tree
algorithm recursively partitions the data based on the selected features, aiming to create
homogeneous subsets with respect to the target variable (Outcome in this case). The
splits are determined based on certain criteria, such as Gini impurity or entropy, to
maximize the separation of classes.
• We utilized the RandomForestClassifier from the sklearn.ensemble module. This
classifier is specifically designed for classification tasks and combines the predictions
of multiple decision trees to make the final prediction. Random Forest builds an
ensemble of decision trees by using bootstrap sampling to create multiple subsets of the
training data. Each decision tree is trained on a different subset of the data, and the final
prediction is made by aggregating the predictions of all the trees.
• We utilized the GaussianNB class from the sklearn.naive_bayes module. This class
implements the Gaussian Naive Bayes algorithm, which assumes that the features follow
a Gaussian (normal) distribution. Naive Bayes calculates the probability of each class
based on the training data. It assumes that the features are conditionally independent
given the class label. The probabilities are estimated using the Gaussian distribution
parameters (mean and variance) for each feature.

# Dataset Used

The dataset used in this project is taken from Kaggle which is Pima Indians Diabetes Database.
It is very famous among the researchers who are working on ML based healthcare application.
Dataset contains total of 8 (Eight) features and 2000 rows. The features are Glucose level,
Pregnancies, Blood Pressure, Skin Thickness, Insulin, BMI, age and Diabetes Pedigree
Function. 

<img width="453" alt="dataset" src="https://github.com/Amarjeet2909/Application-of-ML-based-Diabetes-Prediction-model/assets/78557124/a4674ff9-05be-44d0-92a3-6bb4396ae077">

# Implementation details

Following are the different steps of Implementation:
• Implementation starts with importing necessary libraries of Python
• Reads a CSV file named using pandas and Checks if there are any missing values in the
Data frame
• Performs outlier removal using the Interquartile Range (IQR) method. It creates a new
Data Frame called that excludes rows containing outliers.
• Creates a scatter matrix plot using seaborn's to visualize the relationships between
variables in the Data Frame.
• Extracts the features (X) and target (y) from the outlier-removed Data Frame.
• Performs feature scaling by creating a StandardScaler object, fitting it to the feature data
(X), and transforming the feature data using transform.
• Splits the dataset into training and testing sets using the sklearn, with a test size of 20%
 And training size of 80%
• Trains and evaluates a Support Vector Machine (SVM) model using the linear kernel.
• Trains and evaluates a Random Forest Classifier model.
• Trains and evaluates a Gaussian Naive Bayes model.
• Makes a prediction on the standardized input data using the trained classifier and give
the prediction result based on the predicted class label.

# Results & Analysis

Support Vector Machine 78%
Random Forest 97%
Naïve Bayes 73%
Decision Tree 79%

# Future Scope

Further work that can be done is to consider more life-threatening disease like diabetes. In this
project we have only considered diabetes but there are many other diseases like cancer, Tumor,
Obesity, Kidney disease etc which can be considered for prediction and classification purpose.
Health Insurance policy Suggestion can be the application of this Diabetes prediction. The
policy suggestion is directly related to the most accurate model. For Health policy suggestion,
It can be classified it into three categories which is:
1. Type A Health Insurance Policy
2. Type B Health Insurance Policy
3. Type C Health Insurance Policy
Type A policy can be suggested to the patients with High diabetes, Type B Policy can be
suggested to the patients with medium level of diabetes and last Type C Policy can be suggested
to patients with Low diabetes and for the people with no diabetes no health Insurance policy is
suggested.
