# Airbnb Ratings Prediction Project

## Overview

This project aims to predict the ratings of Airbnb listings in Dublin using various machine learning models. By analyzing multiple features and services of the listings, the project seeks to create an accurate predictive model to assist hosts in improving their offerings and provide renters with reliable information. 

## Objective

The primary goal of this project is to forecast possible ratings for Airbnb listings in Dublin based on features such as cleanliness, communication, overall experience, location, value, and accuracy. These predictions are crucial for helping hosts enhance their services and for providing potential renters with dependable information for their stay.

## Data Sources

Data for this project was sourced from the Inside Airbnb website. The datasets used include:
- **Listings Dataset:** Contains 75 feature columns with a total of 7,345 records.
- **Reviews Dataset:** Contains 6 feature columns with a total of 230,065 reviews.

## Methodology

### Data Preprocessing
1. **Listings Dataset:** Cleaned by excluding irrelevant columns and handling missing values. Amenities were categorized into broader groups for better usability.
2. **Reviews Dataset:** Processed by filtering out non-English entries, removing special characters and emojis, and retaining substantial comments for analysis.

### Feature Engineering
- The listings and reviews data were merged, and features were selected based on their correlation with the target variables.
- The review text data was transformed using the Term Frequency-Inverse Document Frequency (TF-IDF) methodology to extract important tokens as features.

### Machine Learning Models
Three machine learning models were employed and evaluated:
1. **Logistic Regression:** Used for its simplicity and effectiveness in binary and multiclass classification problems. Hyperparameters were tuned to optimize performance.
2. **k-Nearest Neighbors (k-NN):** Utilized to classify data points based on the majority vote of their neighbors. The optimal value of k was determined through cross-validation.
3. **Decision Tree Classifier:** Applied to create a model that splits data based on feature values to make predictions. The depth of the tree was tuned to balance model complexity and accuracy.

## Results

- **Logistic Regression** consistently outperformed the other models in predicting Airbnb ratings. It showed high accuracy and robustness across various target variables.
- **k-NN** provided reasonable predictions but was less effective compared to Logistic Regression, especially for higher-dimensional data.
- **Decision Tree Classifier** showed good performance with appropriate tree depth but was generally less accurate than Logistic Regression.

## Conclusion

This project successfully demonstrates the application of machine learning to predict Airbnb ratings, offering valuable insights for both hosts and platform developers. By identifying key features influencing ratings, hosts can improve their services, and the platform can enhance user experience through personalized recommendations.


## Contact

For any questions or feedback, please contact:
- **Xin Wang**
- Email: wangx33@tcd.ie
