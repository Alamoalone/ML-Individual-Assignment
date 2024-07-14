import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
from sklearn.dummy import DummyRegressor, DummyClassifier
from yellowbrick.classifier import ROCAUC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, auc
from sklearn.metrics import f1_score, classification_report, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.model_selection import cross_val_score


#  Data Loading - Listings and Reviews

listings = pd.read_csv("listings.csv")
reviews = pd.read_csv("reviews_final.csv")

#  Splitting the amenities and logically clustering them into 9 different categories

amenities = listings['amenities'].str.split(',', expand=True)
amenities = amenities.loc[amenities[77].notnull()]
amenities_list = amenities.iloc[0]
amenities_list = amenities_list.to_list()


listings.loc[listings['amenities'].str.contains('Hot Water|Shower gel|Hair dryer|Bathtub|Shampoo|Essentials|Bidet|Conditioner|Body soap|Baby bath'), 
             'bath-products'] = 1
listings.loc[listings['amenities'].str.contains('Bluetooth sound system|Ethernet connection|Heating|Pocket wifi|Cable TV|Wifi'), 
             'electric-system'] = 1
listings.loc[listings['amenities'].str.contains('Breakfast'), 
             'food-services'] = 1
listings.loc[listings['amenities'].str.contains('Outdoor furniture|Dining table|Hangers|High chair|Crib|Clothing storage: wardrobe|Dedicated workspace|Drying rack for clothing|Bed linens|Extra pillows and blankets'), 
             'house-furniture'] = 1
listings.loc[listings['amenities'].str.contains('Cleaning before checkout|Luggage dropoff allowed|Long term stays allowed'), 
             'house-rules'] = 1
listings.loc[listings['amenities'].str.contains('Oven|Hot water kettle|Kitchen|Cooking basics|Microwave|Fire pit|Dishes and silverware|Barbecue utensils|Cleaning products|Baking sheet|Free washer|Free dryer|Iron|Dishwasher|Freezer|Coffee maker|Refrigerator|Toaster|dinnerware|BBQ grill|Stove|Wine glasses'), 
                          'kitchen-appliances'] = 1
listings.loc[listings['amenities'].str.contains('Free parking on premises|Free street parking'), 
             'parking'] = 1
listings.loc[listings['amenities'].str.contains('Board games|Indoor fireplace|Bikes|Shared patio or balcony|Private fenced garden or backyard|crib|books and toys|Outdoor dining area|Private gym in building|Piano|HDTV with Netflix|premium cable|standard cable'), 
             'recreation'] = 1
listings.loc[listings['amenities'].str.contains('Fire extinguisher|Carbon monoxide alarm|Window guards|Fireplace guards|First aid kit|Baby monitor|Private entrance|Lockbox|Smoke alarm|Room-darkening shades|Baby safety gates'), 
             'safety'] = 1


#   Splitting the host contact details into 3 categories namely host_email, host_phone, host_work and host_work_email
host_verification = listings['host_verifications'].str.split(',', expand=True)

listings.loc[listings['host_verifications'].str.contains('email'), 
             'host_email'] = 1
listings.loc[listings['host_verifications'].str.contains('phone'), 
             'host_phone'] = 1
listings.loc[listings['host_verifications'].str.contains('work_email'), 
             'host_work_email'] = 1


#   Handling of NaN values of New Feature Columns Created
new_feature_cols = listings.iloc[:,75:].columns
listings[new_feature_cols] = listings[new_feature_cols].fillna(0)


#   Merging the Listings data with the Cleansed Reviews Dataset
listings = listings.merge(reviews, how='inner', left_on='id', right_on='listing_id')


#   Dropping the columns which are not required while building a Machine Learning Model
#   Plot %NaN Values
def plot_nas(df: pd.DataFrame):
    if df.isnull().sum().sum() != 0:
        na_df = (df.isnull().sum() / len(df)) * 100      
        na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'NaN %' :na_df})
        missing_data.plot(kind = "bar")
        plt.show()
    else:
        print('No NAs found')
plot_nas(listings)


#   Column Drop List
not_needed_columns = [
    'id','listing_url', 'scrape_id', 'last_scraped', 'source', 'name', 
    'picture_url', 'host_id', 'host_url', 'host_name', 'host_location',
    'host_about', 'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood',
    'neighbourhood', 'neighbourhood_group_cleansed', 
    'calendar_updated', 'first_review', 'last_review', 'license', 
    'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes', 
    'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms',
    'description', 'neighborhood_overview', 'host_verifications','host_since', 
    'bathrooms', 'bathrooms_text', 'amenities', 'availability_30', 
    'availability_60', 'availability_90', 'availability_365', 'calendar_last_scraped', 
    'number_of_reviews_ltm', 'number_of_reviews_l30d', 'host_has_profile_pic', 'property_type',
    'minimum_minimum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm',
    'minimum_maximum_nights', 'maximum_minimum_nights', 'maximum_nights_avg_ntm'
    ]

listings.drop(not_needed_columns, axis = 1, inplace = True)
listings = listings.dropna()


#   Missing Data Imputation
imputation_cols = ['bedrooms', 'beds']
for i in imputation_cols:
    listings.loc[listings.loc[:,i].isnull(),i] = listings.loc[:,i].median()


#   Pre-Processing of Features 'price', 'host_response_rate' and 'host_acceptance_rate'
listings['price'] = listings['price'].str.replace('$', '')
listings['price'] = listings['price'].str.replace(',', '')
listings['price'] = pd.to_numeric(listings['price'])
listings['host_response_rate'] = listings["host_response_rate"].str.replace("%","")
listings['host_response_rate'] = pd.to_numeric(listings['host_response_rate'])
listings['host_acceptance_rate'] = listings["host_acceptance_rate"].str.replace("%","")
listings['host_acceptance_rate'] = pd.to_numeric(listings['host_acceptance_rate'])

listings[['price']].plot(kind='box', title='Price BoxPlot')
plt.ylim(0,1750)


#   Outlier Removal for the Feature 'price'
listings = listings[listings.price > 50]
listings = listings[listings.price <= 300]
listings[['price']].plot(kind='box', title='Price BoxPlot')


#   Neighbourhood Analysis
listings.neighbourhood_cleansed.unique()
listings.groupby('neighbourhood_cleansed').host_response_time.count()
neighbourhood_DF=listings.groupby('neighbourhood_cleansed').host_response_time.count()
neighbourhood_DF=neighbourhood_DF.reset_index()
neighbourhood_DF=neighbourhood_DF.rename(columns={'host_response_time':'Number_Of_Listings'})
neighbourhood_DF.plot(kind='bar', 
           x='neighbourhood_cleansed',
           y='Number_Of_Listings',
           figsize =(18,8), 
           title = 'Dublin Neighborhood Frequency', 
           legend = False)


#   Statistical Analysis
print(listings.review_scores_accuracy.mean())
print(listings.review_scores_checkin.mean())
print(listings.review_scores_cleanliness.mean())
print(listings.review_scores_communication.mean())
print(listings.review_scores_location.mean())
print(listings.review_scores_rating.mean())
print(listings.review_scores_value.mean())

print(listings.review_scores_accuracy.median())
print(listings.review_scores_checkin.median())
print(listings.review_scores_cleanliness.median())
print(listings.review_scores_communication.median())
print(listings.review_scores_location.median())
print(listings.review_scores_rating.median())
print(listings.review_scores_value.median())

print(listings.review_scores_accuracy.mode())
print(listings.review_scores_checkin.mode())
print(listings.review_scores_cleanliness.mode())
print(listings.review_scores_communication.mode())
print(listings.review_scores_location.mode())
print(listings.review_scores_rating.mode())
print(listings.review_scores_value.mode())

print(listings.review_scores_accuracy.min())
print(listings.review_scores_checkin.min())
print(listings.review_scores_cleanliness.min())
print(listings.review_scores_communication.min())
print(listings.review_scores_location.min())
print(listings.review_scores_rating.min())
print(listings.review_scores_value.min())

print(listings.review_scores_accuracy.max())
print(listings.review_scores_checkin.max())
print(listings.review_scores_cleanliness.max())
print(listings.review_scores_communication.max())
print(listings.review_scores_location.max())
print(listings.review_scores_rating.max())
print(listings.review_scores_value.max())


#   Density Plots for all 7 Target Variables (Review Scores)
plt.figure(figsize=(18, 18))
plt.subplot(3, 3, 1)
review_scores_accuracy_density = sns.kdeplot(listings.review_scores_accuracy, color="b", 
                                             label='review_scores_accuracy - density')
x_review_scores_accuracy = review_scores_accuracy_density.lines[0].get_xdata()
y_review_scores_accuracy = review_scores_accuracy_density.lines[0].get_ydata()
mean_review_scores_accuracy_density = listings.review_scores_accuracy.mean()
height_review_scores_accuracy = np.interp(mean_review_scores_accuracy_density, x_review_scores_accuracy, 
                                                   y_review_scores_accuracy)
review_scores_accuracy_density.vlines(mean_review_scores_accuracy_density, 0, height_review_scores_accuracy, 
                                      color='b', ls=':')
review_scores_accuracy_density.fill_between(x_review_scores_accuracy, 0, y_review_scores_accuracy, 
                                            facecolor='b', alpha=0.2)
plt.legend()


plt.subplot(3, 3, 2)
review_scores_checkin_density = sns.kdeplot(listings.review_scores_checkin, color="b", 
                                             label='review_scores_checkin - density')
x_review_scores_checkin = review_scores_checkin_density.lines[0].get_xdata()
y_review_scores_checkin = review_scores_checkin_density.lines[0].get_ydata()
mean_review_scores_checkin_density = listings.review_scores_checkin.mean()
height_review_scores_checkin = np.interp(mean_review_scores_checkin_density, x_review_scores_checkin, 
                                                   y_review_scores_checkin)
review_scores_checkin_density.vlines(mean_review_scores_checkin_density, 0, height_review_scores_checkin, 
                                      color='b', ls=':')
review_scores_checkin_density.fill_between(x_review_scores_checkin, 0, y_review_scores_checkin, 
                                            facecolor='b', alpha=0.2)
plt.legend()



plt.subplot(3, 3, 3)
review_scores_cleanliness_density = sns.kdeplot(listings.review_scores_cleanliness, color="b", 
                                             label='review_scores_cleanliness - density')
x_review_scores_cleanliness = review_scores_cleanliness_density.lines[0].get_xdata()
y_review_scores_cleanliness = review_scores_cleanliness_density.lines[0].get_ydata()
mean_review_scores_cleanliness_density = listings.review_scores_cleanliness.mean()
height_review_scores_cleanliness = np.interp(mean_review_scores_cleanliness_density, x_review_scores_cleanliness, 
                                                   y_review_scores_cleanliness)
review_scores_cleanliness_density.vlines(mean_review_scores_cleanliness_density, 0, height_review_scores_cleanliness, 
                                      color='b', ls=':')
review_scores_cleanliness_density.fill_between(x_review_scores_cleanliness, 0, y_review_scores_cleanliness, 
                                            facecolor='b', alpha=0.2)
plt.legend()


plt.subplot(3, 3, 4)
review_scores_communication_density = sns.kdeplot(listings.review_scores_communication, color="b", 
                                             label='review_scores_communication - density')
x_review_scores_communication = review_scores_communication_density.lines[0].get_xdata()
y_review_scores_communication = review_scores_communication_density.lines[0].get_ydata()
mean_review_scores_communication_density = listings.review_scores_communication.mean()
height_review_scores_communication = np.interp(mean_review_scores_communication_density, x_review_scores_communication, 
                                                   y_review_scores_communication)
review_scores_communication_density.vlines(mean_review_scores_communication_density, 0, height_review_scores_communication, 
                                      color='b', ls=':')
review_scores_communication_density.fill_between(x_review_scores_communication, 0, y_review_scores_communication, 
                                            facecolor='b', alpha=0.2)
plt.legend()

plt.subplot(3, 3, 5)
review_scores_rating_density = sns.kdeplot(listings.review_scores_rating, color="b", 
                                             label='review_scores_rating - density')
x_review_scores_rating = review_scores_rating_density.lines[0].get_xdata()
y_review_scores_rating = review_scores_rating_density.lines[0].get_ydata()
mean_review_scores_rating_density = listings.review_scores_rating.mean()
height_review_scores_rating = np.interp(mean_review_scores_rating_density, x_review_scores_rating, 
                                                   y_review_scores_rating)
review_scores_rating_density.vlines(mean_review_scores_rating_density, 0, height_review_scores_rating, 
                                      color='b', ls=':')
review_scores_rating_density.fill_between(x_review_scores_rating, 0, y_review_scores_rating, 
                                            facecolor='b', alpha=0.2)
plt.legend()


plt.subplot(3, 3, 6)
review_scores_location_density = sns.kdeplot(listings.review_scores_location, color="b", 
                                             label='review_scores_location - density')
x_review_scores_location = review_scores_location_density.lines[0].get_xdata()
y_review_scores_location = review_scores_location_density.lines[0].get_ydata()
mean_review_scores_location_density = listings.review_scores_location.mean()
height_review_scores_location = np.interp(mean_review_scores_location_density, x_review_scores_location, 
                                                   y_review_scores_location)
review_scores_location_density.vlines(mean_review_scores_location_density, 0, height_review_scores_location, 
                                      color='b', ls=':')
review_scores_location_density.fill_between(x_review_scores_location, 0, y_review_scores_location, 
                                            facecolor='b', alpha=0.2)
plt.legend()

plt.subplot(3, 3, 8)
review_scores_value_density = sns.kdeplot(listings.review_scores_value, color="b", 
                                             label='review_scores_value - density')
x_review_scores_value = review_scores_value_density.lines[0].get_xdata()
y_review_scores_value = review_scores_value_density.lines[0].get_ydata()
mean_review_scores_value_density = listings.review_scores_value.mean()
height_review_scores_value = np.interp(mean_review_scores_value_density, x_review_scores_value, 
                                                   y_review_scores_value)
review_scores_value_density.vlines(mean_review_scores_value_density, 0, height_review_scores_value, 
                                      color='b', ls=':')
review_scores_value_density.fill_between(x_review_scores_value, 0, y_review_scores_value, 
                                            facecolor='b', alpha=0.2)
plt.legend()

plt.show()


#   Data Scaling
#   Min Max Scaling
def minmax(X):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (X.max() - X.min()) + X.min()
    return X_scaled


#   List of Columns to be Scaled
scaling_data = ['host_response_rate', 'host_acceptance_rate', 'bedrooms', 'beds',
                'host_listings_count', 'host_total_listings_count',
                'latitude', 'longitude', 'accommodates', 'price', 
                'minimum_nights', 'maximum_nights', 'number_of_reviews', 'num_cmt', 'avg_senti',
                'review_scores_rating', 'review_scores_accuracy', 
                'review_scores_cleanliness', 'review_scores_checkin', 
                'review_scores_communication', 'review_scores_location', 
                'review_scores_value', 'reviews_per_month','bath-products','electric-system',
                'food-services','house-furniture','house-rules',
                'kitchen-appliances','parking','recreation','safety',
                'host_email','host_work_email']

for i in scaling_data:
  listings[i] = minmax(listings[i])


#   Label Encoding
listings.dropna(axis = 1, inplace = True)
label_encoder = preprocessing.LabelEncoder()
listings.host_response_time     = label_encoder.fit_transform(listings.host_response_time)
listings.host_is_superhost      = label_encoder.fit_transform(listings.host_is_superhost)
listings.host_identity_verified = label_encoder.fit_transform(listings.host_identity_verified)
listings.instant_bookable       = label_encoder.fit_transform(listings.instant_bookable)
listings.room_type              = label_encoder.fit_transform(listings.room_type)
listings.neighbourhood_cleansed = label_encoder.fit_transform(listings.neighbourhood_cleansed)
listings.has_availability       = label_encoder.fit_transform(listings.has_availability)


#   Correlation Matrix
test_corr = listings.corr()
test_corr.to_csv("test_corr.csv")

#   Metrics Print Function
def print_metrics(y_test, y_pred):
    print("-"*10+"CONFUSION-MATRIX"+"-"*10)
    print(confusion_matrix(y_test, y_pred))

    print("-"*10+"CLASSIFICATION-REPORT"+"-"*10)
    print(classification_report(y_test, y_pred))

#   Review Scores Accuracy
#   Logistic Regression
#    Defining the Input Variables
X = listings[
                ['host_response_time', 'host_response_rate', 'host_acceptance_rate',
                 'bedrooms', 'beds','neighbourhood_cleansed',
                 'host_is_superhost', 'host_listings_count', 'host_total_listings_count',
                 'host_identity_verified', 'room_type',
                 'accommodates','price', 'minimum_nights', 'maximum_nights',
                 'bath-products','electric-system',
                 'food-services','house-furniture','house-rules',
                 'kitchen-appliances','parking','recreation','safety',
                 'host_email','host_work_email'] + list(reviews.columns[2:])
]


#    Defining the Quantile Bins for the Target Variables
y = listings[['review_scores_accuracy']]
y = (y/y.max())*100

y = y.assign(
    rating_bin_ep = pd.qcut(
        y['review_scores_accuracy'],
        q=3,
        duplicates='drop',
        labels=[0,1,2]
    )
)


#    Min Max of Each Bin
y.groupby('rating_bin_ep').min()
y.groupby('rating_bin_ep').max()
y = y['rating_bin_ep']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


#    Number of Records in Each Bin
cnt_plt = sns.countplot(y)
cnt_plt.bar_label(cnt_plt.containers[0])
plt.show()


#    Logistic Regression - Varied C Range, using 'newton-cg' solver and multi_class='multinomial'
c_range = [0.001, 0.1, 1, 10, 100, 1000]
mean_error = []
std_error = []
for c in sorted(c_range):
    logit = LogisticRegression(C=c, random_state=0, solver='newton-cg',multi_class='multinomial')
    logit.fit(X_train, y_train)
    y_pred = logit.predict(X_test)
    print("C = ",c)
    print('Train accuracy score:',logit.score(X_train, y_train))
    print('Test accuracy score:',logit.score(X_test, y_test))
    print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
    scores = cross_val_score(logit, X_test, y_test, cv=5, scoring='accuracy')
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
plt.errorbar(c_range, mean_error, yerr = std_error, linewidth=3)
plt.xlabel('C', fontsize=25)
plt.ylabel('Accuracy Score', fontsize=25)
title_cv = "Logistic Regression - Cross Validation \nFor Degree = 1"
plt.title(title_cv, fontsize=25)
plt.show()


#    ROC-AUC Curve for all three categories
visualizer = ROCAUC(logit, classes=["0", "1", "2"], macro=False, micro=False)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)        
visualizer.show() 

#    Feature Importance
from matplotlib import pyplot
cols = X.columns
cols = np.asarray(cols)

plt.figure(figsize=(30,15))
feature_importance = abs(logit.coef_[0])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
top_features = pd.DataFrame({'feature_imp': feature_importance, 
                             'features': cols}, columns=['feature_imp', 'features'])
top_features = top_features.sort_values(by='feature_imp', ascending=False).head(20)
plt.bar(top_features.features, top_features.feature_imp)
plt.xlabel('Importance', fontsize=35, fontweight='bold')
plt.ylabel('Feature', fontsize=35, fontweight='bold')
plt.xticks(fontsize=30, rotation = 90)
plt.yticks(fontsize=30)
plt.show()

plt.figure(figsize=(30,15))
feature_importance = abs(logit.coef_[1])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
top_features = pd.DataFrame({'feature_imp': feature_importance, 
                             'features': cols}, columns=['feature_imp', 'features'])
top_features = top_features.sort_values(by='feature_imp', ascending=False).head(20)
plt.bar(top_features.features, top_features.feature_imp)
plt.xlabel('Importance', fontsize=35, fontweight='bold')
plt.ylabel('Feature', fontsize=35, fontweight='bold')
plt.xticks(fontsize=30, rotation = 90)
plt.yticks(fontsize=30)
plt.show()

plt.figure(figsize=(30,15))
feature_importance = abs(logit.coef_[2])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
top_features = pd.DataFrame({'feature_imp': feature_importance, 
                             'features': cols}, columns=['feature_imp', 'features'])
top_features = top_features.sort_values(by='feature_imp', ascending=False).head(20)
plt.bar(top_features.features, top_features.feature_imp)
plt.xlabel('Importance', fontsize=35, fontweight='bold')
plt.ylabel('Feature', fontsize=35, fontweight='bold')
plt.xticks(fontsize=30, rotation = 90)
plt.yticks(fontsize=30)
plt.show()


#    Comparison With Baseline Classifier
dmfr = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
dmun = DummyClassifier(strategy='uniform').fit(X_train, y_train)

print("\n\nDUMMY CLASSIFIER - frequent")
print(dmfr.score(X_test, y_test))
y_pred = dmfr.predict(X_test)
print_metrics(y_test, y_pred)

print("\n\nDUMMY CLASSIFIER - Uniform")
print(dmun.score(X_test, y_test))
y_pred = dmun.predict(X_test)
print_metrics(y_test, y_pred)


#    Density Plot for Term 'Home' for each Review Bin
test = pd.concat([X, y], axis=1)
g = sns.kdeplot(test.loc[test['rating_bin_ep'] == 0, 'term_home'], color="r", label='term_home_intensity, bin=0')
g.set(ylim=(0, 19))
g.set(xlim=(-0.25, 1))
plt.legend()

h = sns.kdeplot(test.loc[test['rating_bin_ep'] == 1,'term_home'], color="g", label='term_home_intensity, bin=1')
h.set(ylim=(0, 19))
h.set(xlim=(-0.25, 1))
plt.legend()

i = sns.kdeplot(test.loc[test['rating_bin_ep'] == 2, 'term_home'], color="b", label='term_home_intensity, bin=2')
i.set(ylim=(0, 19))
i.set(xlim=(-0.25, 1))
plt.legend()

plt.show()


#    Density Plots for Number of Comments against each Review Bin
test = pd.concat([X, y], axis=1)

g = sns.kdeplot(test.loc[test['rating_bin_ep'] == 0, 'num_cmt'], color="r", label='num_cmt, bin=0')
g.set(ylim=(0, 0.03))
plt.legend()

h = sns.kdeplot(test.loc[test['rating_bin_ep'] == 1,'num_cmt'], color="g", label='num_cmt, bin=1')
h.set(ylim=(0, 0.03))
plt.legend()

i = sns.kdeplot(test.loc[test['rating_bin_ep'] == 2, 'num_cmt'], color="b", label='num_cmt, bin=2')
i.set(ylim=(0, 0.03))
plt.legend()

plt.show()


#    Number of Records in each Review Bin for the type of Host (Superhost)
test.host_is_superhost.value_counts()
test1 = test.groupby(['host_is_superhost', 'rating_bin_ep']).size()

plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
for j in (0,1,2):
    plt.bar(j, test1[0][j], label = str(j))
plt.title("Host is a Super Host")    
plt.legend()
plt.xlabel('Review Bin', fontweight ='bold', fontsize = 15)
plt.ylabel('# Records', fontweight ='bold', fontsize = 15)
plt.subplot(1, 2, 2)
for j in (0,1,2):
    plt.bar(j, test1[1][j], label = str(j))
plt.title("Host is not a Super Host")
plt.legend()
plt.xlabel('Review Bin', fontweight ='bold', fontsize = 15)
plt.ylabel('# Records', fontweight ='bold', fontsize = 15)
plt.show()


#    Neighbourhood Type Analysis
test.neighbourhood_cleansed.value_counts()
test2 = test.groupby(['neighbourhood_cleansed', 'rating_bin_ep']).size()

plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
for j in (0,1,2):
    plt.bar(j, test2[0][j], label = str(j))
plt.title("Neighbourhood 0",fontsize=25)    
plt.legend()
plt.xlabel('Review Bin', fontweight ='bold', fontsize = 15)
plt.ylabel('# Records', fontweight ='bold', fontsize = 15)
plt.subplot(1, 2, 2)
for j in (0,1,2):
    plt.bar(j, test2[1][j], label = str(j))
plt.title("Neighbourhood 1",fontsize=25)
plt.legend()
plt.xlabel('Review Bin', fontweight ='bold', fontsize = 15)
plt.ylabel('# Records', fontweight ='bold', fontsize = 15)
plt.show()

plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
for j in (0,1,2):
    plt.bar(j, test2[2][j], label = str(j))
plt.title("Neighbourhood 2",fontsize=25)
plt.legend()
plt.xlabel('Review Bin', fontweight ='bold', fontsize = 15)
plt.ylabel('# Records', fontweight ='bold', fontsize = 15)
plt.subplot(1, 2, 2)
for j in (0,1,2):
    plt.bar(j, test2[3][j], label = str(j))
plt.title("Neighbourhood 3",fontsize=25)
plt.legend()
plt.xlabel('Review Bin', fontweight ='bold', fontsize = 15)
plt.ylabel('# Records', fontweight ='bold', fontsize = 15)
plt.show()


#    Polynomial Degree and Error Plots
c_range = [0.001, 0.1, 1, 10, 100, 1000]
degree_range = [2]

for i in degree_range:
    trans = PolynomialFeatures(degree = i)
    x_poly = trans.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size = 0.2, random_state=(1))
    mean_error = []
    std_error = []
    for c in c_range:
        log_reg = LogisticRegression(C = c, random_state=0, solver='newton-cg',multi_class='multinomial')
        log_reg.fit(x_train, y_train)
        y_pred = log_reg.predict(x_test)
        
        cnf_mtx = metrics.confusion_matrix(y_test, y_pred)
        f1_score = (2*cnf_mtx[1][1])/((2*cnf_mtx[1][1]) + cnf_mtx[0][1] + cnf_mtx[1][0])
        
        scores = cross_val_score(log_reg, x_test, y_test, cv=5, scoring='accuracy')
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
        
        
        print(" Logistic Regression")
        print(" For Degree = ", i)
        print(" For C = ", c)
        print(" Confusion Matrix - \n", cnf_mtx)
        print(' Train accuracy score: ', log_reg.score(x_train, y_train))
        print(' Test accuracy score: ', log_reg.score(x_test, y_test))
        print(" F1 Score = ", f1_score)
        print(" Classification Report\n", classification_report(y_test, y_pred))
        print("\n")
    
    plt.errorbar(c_range, mean_error, yerr = std_error, linewidth=3)
    plt.xlabel('C', fontsize=25)
    plt.ylabel('Accuracy Score', fontsize=25)
    title_cv = f"Logistic Regression - Cross Validation \nFor Degree = {i}"
    plt.title(title_cv, fontsize=25)
    plt.show()


#   k-NN Classifier
nn_range = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
x_train_nn, x_test_nn, y_train_nn, y_test_nn = train_test_split(X, y, test_size = 0.2, random_state=(1))
merr = []
serr = []

for nn in nn_range:
    knn_model = KNeighborsClassifier(n_neighbors=nn, weights='uniform')
    knn_model.fit(x_train_nn, y_train_nn)
    y_pred_nn = knn_model.predict(x_test_nn)
    print("NN = ", nn)
    print('Train accuracy score:',knn_model.score(x_train_nn, y_train_nn))
    print('Test accuracy score:',knn_model.score(x_test_nn, y_test_nn))
    
    scores_knn = cross_val_score(knn_model, x_test_nn, y_test_nn, cv=5, scoring='accuracy')
    merr.append(np.array(scores_knn).mean())
    serr.append(np.array(scores_knn).std())

plt.errorbar(nn_range, merr, yerr = serr, linewidth=3)
plt.xlabel('NN', fontsize=25)
plt.ylabel('Accuracy Score', fontsize=25)
title_cv = f"k-NN - Cross Validation"
plt.title(title_cv, fontsize=25)
plt.show()


#    ROC-AUC Curve
visualizer = ROCAUC(knn_model, classes=["0", "1", "2"], macro=False, micro=False)

visualizer.fit(x_train_nn, y_train_nn)
visualizer.score(x_test_nn, y_test_nn)        
visualizer.show() 


#   Review Scores Checkin
X = listings[
                ['host_response_time', 'host_response_rate', 'host_acceptance_rate',
                 'bedrooms', 'beds','neighbourhood_cleansed',
                 'host_is_superhost', 'host_listings_count', 'host_total_listings_count',
                 'host_identity_verified', 'room_type',
                 'accommodates','price', 'minimum_nights', 'maximum_nights',
                 'bath-products','electric-system',
                 'food-services','house-furniture','house-rules',
                 'kitchen-appliances','parking','recreation','safety',
                 'host_email','host_work_email'] + list(reviews.columns[2:])
]

y = listings[['review_scores_checkin']]
y = (y/y.max())*100

y = y.assign(
    rating_bin_ep = pd.qcut(
        y['review_scores_checkin'],
        q=2,
        duplicates='drop',
        labels=[0,1]
    )
)


#    Min Max of Each Bin
y.groupby('rating_bin_ep').min()
y.groupby('rating_bin_ep').max()


#    Splitting Data in 75-25 Ratio
y = y['rating_bin_ep']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


#    Number of Records in Each Bin
cnt_plt = sns.countplot(y)
cnt_plt.bar_label(cnt_plt.containers[0])
plt.show()


#    Logistic Regression - Varied C Range, using 'newton-cg' solver and multi_class='multinomial'
c_range = [0.001, 0.1, 1, 10, 100, 1000]
mean_error = []
std_error = []
for c in sorted(c_range):
    logit = LogisticRegression(C=c, random_state=0, solver='newton-cg',multi_class='multinomial')
    logit.fit(X_train, y_train)
    y_pred = logit.predict(X_test)
    print("C = ",c)
    print('Train accuracy score:',logit.score(X_train, y_train))
    print('Test accuracy score:',logit.score(X_test, y_test))
    print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
    scores = cross_val_score(logit, X_test, y_test, cv=5, scoring='accuracy')
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
plt.errorbar(c_range, mean_error, yerr = std_error, linewidth=3)
plt.xlabel('C', fontsize=25)
plt.ylabel('Accuracy Score', fontsize=25)
title_cv = "Logistic Regression - Cross Validation \nFor Degree = 1"
plt.title(title_cv, fontsize=25)
plt.show()


#    Feature Importance
cols = X.columns
cols = np.asarray(cols)

plt.figure(figsize=(30,15))
feature_importance = abs(logit.coef_[0])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
top_features = pd.DataFrame({'feature_imp': feature_importance, 
                             'features': cols}, columns=['feature_imp', 'features'])
top_features = top_features.sort_values(by='feature_imp', ascending=False).head(20)
plt.bar(top_features.features, top_features.feature_imp)
plt.xlabel('Importance', fontsize=35, fontweight='bold')
plt.ylabel('Feature', fontsize=35, fontweight='bold')
plt.xticks(fontsize=30, rotation = 90)
plt.yticks(fontsize=30)
plt.show()


#    Polynomial Degree and Error Plots
c_range = [0.001, 0.1, 1, 10, 100, 1000]
degree_range = [2]

for i in degree_range:
    trans = PolynomialFeatures(degree = i)
    x_poly = trans.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size = 0.2, random_state=(1))
    mean_error = []
    std_error = []
    for c in c_range:
        log_reg = LogisticRegression(C = c, random_state=0, solver='newton-cg',multi_class='multinomial')
        log_reg.fit(x_train, y_train)
        y_pred = log_reg.predict(x_test)
        
        cnf_mtx = metrics.confusion_matrix(y_test, y_pred)
        f1_score = (2*cnf_mtx[1][1])/((2*cnf_mtx[1][1]) + cnf_mtx[0][1] + cnf_mtx[1][0])
        
        scores = cross_val_score(log_reg, x_test, y_test, cv=5, scoring='accuracy')
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
        
        
        print(" Logistic Regression")
        print(" For Degree = ", i)
        print(" For C = ", c)
        print(" Confusion Matrix - \n", cnf_mtx)
        print(' Train accuracy score: ', log_reg.score(x_train, y_train))
        print(' Test accuracy score: ', log_reg.score(x_test, y_test))
        print(" F1 Score = ", f1_score)
        print(" Classification Report\n", classification_report(y_test, y_pred))
        print("\n")
    
    plt.errorbar(c_range, mean_error, yerr = std_error, linewidth=3)
    plt.xlabel('C', fontsize=25)
    plt.ylabel('Accuracy Score', fontsize=25)
    title_cv = f"Logistic Regression - Cross Validation \nFor Degree = {i}"
    plt.title(title_cv, fontsize=25)
    plt.show()


#   k-NN Classifier
nn_range = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
x_train_nn, x_test_nn, y_train_nn, y_test_nn = train_test_split(X, y, test_size = 0.2, random_state=(1))
merr = []
serr = []

for nn in nn_range:
    knn_model = KNeighborsClassifier(n_neighbors=nn, weights='uniform')
    knn_model.fit(x_train_nn, y_train_nn)
    y_pred_nn = knn_model.predict(x_test_nn)
    print("NN = ", nn)
    print('Train accuracy score:',knn_model.score(x_train_nn, y_train_nn))
    print('Test accuracy score:',knn_model.score(x_test_nn, y_test_nn))
    
    scores_knn = cross_val_score(knn_model, x_test_nn, y_test_nn, cv=5, scoring='accuracy')
    merr.append(np.array(scores_knn).mean())
    serr.append(np.array(scores_knn).std())

plt.errorbar(nn_range, merr, yerr = serr, linewidth=3)
plt.xlabel('NN', fontsize=25)
plt.ylabel('Accuracy Score', fontsize=25)
title_cv = f"k-NN - Cross Validation"
plt.title(title_cv, fontsize=25)
plt.show()


#   Review Scores Cleanliness
X = listings[
                ['host_response_time', 'host_response_rate', 'host_acceptance_rate',
                 'bedrooms', 'beds','neighbourhood_cleansed',
                 'host_is_superhost', 'host_listings_count', 'host_total_listings_count',
                 'host_identity_verified', 'room_type',
                 'accommodates','price', 'minimum_nights', 'maximum_nights',
                 'bath-products','electric-system',
                 'food-services','house-furniture','house-rules',
                 'kitchen-appliances','parking','recreation','safety',
                 'host_email','host_work_email'] + list(reviews.columns[2:])
]

y = listings[['review_scores_cleanliness']]
y = (y/y.max())*100

y = y.assign(
    rating_bin_ep = pd.qcut(
        y['review_scores_cleanliness'],
        q=3,
        duplicates='drop',
        labels=[0,1,2]
    )
)


#    Min Max of Each Bin
y.groupby('rating_bin_ep').min()
y.groupby('rating_bin_ep').max()


#    Splitting Data in 75-25 Ratio
y = y['rating_bin_ep']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


#    Number of Records in Each Bin
cnt_plt = sns.countplot(y)
cnt_plt.bar_label(cnt_plt.containers[0])
plt.show()


#    Logistic Regression - Varied C Range, using 'newton-cg' solver and multi_class='multinomial'
c_range = [0.001, 0.1, 1, 10, 100, 1000]
mean_error = []
std_error = []
for c in sorted(c_range):
    logit = LogisticRegression(C=c, random_state=0, solver='newton-cg',multi_class='multinomial')
    logit.fit(X_train, y_train)
    y_pred = logit.predict(X_test)
    print("C = ",c)
    print('Train accuracy score:',logit.score(X_train, y_train))
    print('Test accuracy score:',logit.score(X_test, y_test))
    print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
    scores = cross_val_score(logit, X_test, y_test, cv=5, scoring='accuracy')
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
plt.errorbar(c_range, mean_error, yerr = std_error, linewidth=3)
plt.xlabel('C', fontsize=25)
plt.ylabel('Accuracy Score', fontsize=25)
title_cv = "Logistic Regression - Cross Validation \nFor Degree = 1"
plt.title(title_cv, fontsize=25)
plt.show()


#    Feature Importance
cols = X.columns
cols = np.asarray(cols)

plt.figure(figsize=(30,15))
feature_importance = abs(logit.coef_[0])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
top_features = pd.DataFrame({'feature_imp': feature_importance, 
                             'features': cols}, columns=['feature_imp', 'features'])
top_features = top_features.sort_values(by='feature_imp', ascending=False).head(20)
plt.bar(top_features.features, top_features.feature_imp)
plt.xlabel('Importance', fontsize=35, fontweight='bold')
plt.ylabel('Feature', fontsize=35, fontweight='bold')
plt.xticks(fontsize=30, rotation = 90)
plt.yticks(fontsize=30)
plt.show()

plt.figure(figsize=(30,15))
feature_importance = abs(logit.coef_[1])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
top_features = pd.DataFrame({'feature_imp': feature_importance, 
                             'features': cols}, columns=['feature_imp', 'features'])
top_features = top_features.sort_values(by='feature_imp', ascending=False).head(20)
plt.bar(top_features.features, top_features.feature_imp)
plt.xlabel('Importance', fontsize=35, fontweight='bold')
plt.ylabel('Feature', fontsize=35, fontweight='bold')
plt.xticks(fontsize=30, rotation = 90)
plt.yticks(fontsize=30)
plt.show()

plt.figure(figsize=(30,15))
feature_importance = abs(logit.coef_[2])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
top_features = pd.DataFrame({'feature_imp': feature_importance, 
                             'features': cols}, columns=['feature_imp', 'features'])
top_features = top_features.sort_values(by='feature_imp', ascending=False).head(20)
plt.bar(top_features.features, top_features.feature_imp)
plt.xlabel('Importance', fontsize=35, fontweight='bold')
plt.ylabel('Feature', fontsize=35, fontweight='bold')
plt.xticks(fontsize=30, rotation = 90)
plt.yticks(fontsize=30)
plt.show()


#    ROC-AUC Curve for all three categories
visualizer = ROCAUC(logit, classes=["0", "1", "2"], macro=False, micro=False)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)        
visualizer.show() 


#    Polynomial Degree and Error Plots
c_range = [0.001, 0.1, 1, 10, 100, 1000]
degree_range = [2]

for i in degree_range:
    trans = PolynomialFeatures(degree = i)
    x_poly = trans.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size = 0.2, random_state=(1))
    mean_error = []
    std_error = []
    for c in c_range:
        log_reg = LogisticRegression(C = c, random_state=0, solver='newton-cg',multi_class='multinomial')
        log_reg.fit(x_train, y_train)
        y_pred = log_reg.predict(x_test)
        
        cnf_mtx = metrics.confusion_matrix(y_test, y_pred)
        f1_score = (2*cnf_mtx[1][1])/((2*cnf_mtx[1][1]) + cnf_mtx[0][1] + cnf_mtx[1][0])
        
        scores = cross_val_score(log_reg, x_test, y_test, cv=5, scoring='accuracy')
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
        
        
        print(" Logistic Regression")
        print(" For Degree = ", i)
        print(" For C = ", c)
        print(" Confusion Matrix - \n", cnf_mtx)
        print(' Train accuracy score: ', log_reg.score(x_train, y_train))
        print(' Test accuracy score: ', log_reg.score(x_test, y_test))
        print(" F1 Score = ", f1_score)
        print(" Classification Report\n", classification_report(y_test, y_pred))
        print("\n")
    
    plt.errorbar(c_range, mean_error, yerr = std_error, linewidth=3)
    plt.xlabel('C', fontsize=25)
    plt.ylabel('Accuracy Score', fontsize=25)
    title_cv = f"Logistic Regression - Cross Validation \nFor Degree = {i}"
    plt.title(title_cv, fontsize=25)
    plt.show()


#    k-NN Classifier
nn_range = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
x_train_nn, x_test_nn, y_train_nn, y_test_nn = train_test_split(X, y, test_size = 0.2, random_state=(1))
merr = []
serr = []

for nn in nn_range:
    knn_model = KNeighborsClassifier(n_neighbors=nn, weights='uniform')
    knn_model.fit(x_train_nn, y_train_nn)
    y_pred_nn = knn_model.predict(x_test_nn)
    print("NN = ", nn)
    print('Train accuracy score:',knn_model.score(x_train_nn, y_train_nn))
    print('Test accuracy score:',knn_model.score(x_test_nn, y_test_nn))
    
    scores_knn = cross_val_score(knn_model, x_test_nn, y_test_nn, cv=5, scoring='accuracy')
    merr.append(np.array(scores_knn).mean())
    serr.append(np.array(scores_knn).std())

plt.errorbar(nn_range, merr, yerr = serr, linewidth=3)
plt.xlabel('NN', fontsize=25)
plt.ylabel('Accuracy Score', fontsize=25)
title_cv = f"k-NN - Cross Validation"
plt.title(title_cv, fontsize=25)
plt.show()


#    ROC-AUC Curve
visualizer = ROCAUC(knn_model, classes=["0", "1", "2"], macro=False, micro=False)

visualizer.fit(x_train_nn, y_train_nn)
visualizer.score(x_test_nn, y_test_nn)        
visualizer.show() 


#   Review Scores Communication
X = listings[
                ['host_response_time', 'host_response_rate', 'host_acceptance_rate',
                 'bedrooms', 'beds','neighbourhood_cleansed',
                 'host_is_superhost', 'host_listings_count', 'host_total_listings_count',
                 'host_identity_verified', 'room_type',
                 'accommodates','price', 'minimum_nights', 'maximum_nights',
                 'bath-products','electric-system',
                 'food-services','house-furniture','house-rules',
                 'kitchen-appliances','parking','recreation','safety',
                 'host_email','host_work_email'] + list(reviews.columns[2:])
]

y = listings[['review_scores_communication']]
y = (y/y.max())*100

y = y.assign(
    rating_bin_ep = pd.qcut(
        y['review_scores_communication'],
        q=2,
        duplicates='drop',
        labels=[0,1]
    )
)


#    Min Max of Each Bin
y.groupby('rating_bin_ep').min()
y.groupby('rating_bin_ep').max()


#    Splitting Data in 75-25 Ratio
y = y['rating_bin_ep']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


#    Number of Records in Each Bin
cnt_plt = sns.countplot(y)
cnt_plt.bar_label(cnt_plt.containers[0])
plt.show()


#    Logistic Regression - Varied C Range, using 'newton-cg' solver and multi_class='multinomial'
c_range = [0.001, 0.1, 1, 10, 100, 1000]
mean_error = []
std_error = []
for c in sorted(c_range):
    logit = LogisticRegression(C=c, random_state=0, solver='newton-cg',multi_class='multinomial')
    logit.fit(X_train, y_train)
    y_pred = logit.predict(X_test)
    print("C = ",c)
    print('Train accuracy score:',logit.score(X_train, y_train))
    print('Test accuracy score:',logit.score(X_test, y_test))
    print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
    scores = cross_val_score(logit, X_test, y_test, cv=5, scoring='accuracy')
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
plt.errorbar(c_range, mean_error, yerr = std_error, linewidth=3)
plt.xlabel('C', fontsize=25)
plt.ylabel('Accuracy Score', fontsize=25)
title_cv = "Logistic Regression - Cross Validation \nFor Degree = 1"
plt.title(title_cv, fontsize=25)
plt.show()


#    Feature Importance
cols = X.columns
cols = np.asarray(cols)

plt.figure(figsize=(30,15))
feature_importance = abs(logit.coef_[0])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
top_features = pd.DataFrame({'feature_imp': feature_importance, 
                             'features': cols}, columns=['feature_imp', 'features'])
top_features = top_features.sort_values(by='feature_imp', ascending=False).head(20)
plt.bar(top_features.features, top_features.feature_imp)
plt.xlabel('Importance', fontsize=35, fontweight='bold')
plt.ylabel('Feature', fontsize=35, fontweight='bold')
plt.xticks(fontsize=30, rotation = 90)
plt.yticks(fontsize=30)
plt.show()


#    Polynomial Degree and Error Plots
c_range = [0.001, 0.1, 1, 10, 100, 1000]
degree_range = [2]

for i in degree_range:
    trans = PolynomialFeatures(degree = i)
    x_poly = trans.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size = 0.2, random_state=(1))
    mean_error = []
    std_error = []
    for c in c_range:
        log_reg = LogisticRegression(C = c, random_state=0, solver='newton-cg',multi_class='multinomial')
        log_reg.fit(x_train, y_train)
        y_pred = log_reg.predict(x_test)
        
        cnf_mtx = metrics.confusion_matrix(y_test, y_pred)
        f1_score = (2*cnf_mtx[1][1])/((2*cnf_mtx[1][1]) + cnf_mtx[0][1] + cnf_mtx[1][0])
        
        scores = cross_val_score(log_reg, x_test, y_test, cv=5, scoring='accuracy')
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
        
        
        print(" Logistic Regression")
        print(" For Degree = ", i)
        print(" For C = ", c)
        print(" Confusion Matrix - \n", cnf_mtx)
        print(' Train accuracy score: ', log_reg.score(x_train, y_train))
        print(' Test accuracy score: ', log_reg.score(x_test, y_test))
        print(" F1 Score = ", f1_score)
        print(" Classification Report\n", classification_report(y_test, y_pred))
        print("\n")
    
    plt.errorbar(c_range, mean_error, yerr = std_error, linewidth=3)
    plt.xlabel('C', fontsize=25)
    plt.ylabel('Accuracy Score', fontsize=25)
    title_cv = f"Logistic Regression - Cross Validation \nFor Degree = {i}"
    plt.title(title_cv, fontsize=25)
    plt.show()


#   k-NN Classifier
nn_range = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
x_train_nn, x_test_nn, y_train_nn, y_test_nn = train_test_split(X, y, test_size = 0.2, random_state=(1))
merr = []
serr = []

for nn in nn_range:
    knn_model = KNeighborsClassifier(n_neighbors=nn, weights='uniform')
    knn_model.fit(x_train_nn, y_train_nn)
    y_pred_nn = knn_model.predict(x_test_nn)
    print("NN = ", nn)
    print('Train accuracy score:',knn_model.score(x_train_nn, y_train_nn))
    print('Test accuracy score:',knn_model.score(x_test_nn, y_test_nn))
    
    scores_knn = cross_val_score(knn_model, x_test_nn, y_test_nn, cv=5, scoring='accuracy')
    merr.append(np.array(scores_knn).mean())
    serr.append(np.array(scores_knn).std())

plt.errorbar(nn_range, merr, yerr = serr, linewidth=3)
plt.xlabel('NN', fontsize=25)
plt.ylabel('Accuracy Score', fontsize=25)
title_cv = f"k-NN - Cross Validation"
plt.title(title_cv, fontsize=25)
plt.show()


#   Review Scores Location
X = listings[
                ['host_response_time', 'host_response_rate', 'host_acceptance_rate',
                 'bedrooms', 'beds','neighbourhood_cleansed',
                 'host_is_superhost', 'host_listings_count', 'host_total_listings_count',
                 'host_identity_verified', 'room_type',
                 'accommodates','price', 'minimum_nights', 'maximum_nights',
                 'bath-products','electric-system',
                 'food-services','house-furniture','house-rules',
                 'kitchen-appliances','parking','recreation','safety',
                 'host_email','host_work_email'] + list(reviews.columns[2:])
]

y = listings[['review_scores_location']]
y = (y/y.max())*100

y = y.assign(
    rating_bin_ep = pd.qcut(
        y['review_scores_location'],
        q=3,
        duplicates='drop',
        labels=[0,1,2]
    )
)


#    Min Max of Each Bin
y.groupby('rating_bin_ep').min()
y.groupby('rating_bin_ep').max()


#    Splitting Data in 75-25 Ratio
y = y['rating_bin_ep']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


#    Number of Records in Each Bin
cnt_plt = sns.countplot(y)
cnt_plt.bar_label(cnt_plt.containers[0])
plt.show()


#    Logistic Regression - Varied C Range, using 'newton-cg' solver and multi_class='multinomial'
c_range = [0.001, 0.1, 1, 10, 100, 1000]
mean_error = []
std_error = []
for c in sorted(c_range):
    logit = LogisticRegression(C=c, random_state=0, solver='newton-cg',multi_class='multinomial')
    logit.fit(X_train, y_train)
    y_pred = logit.predict(X_test)
    print("C = ",c)
    print('Train accuracy score:',logit.score(X_train, y_train))
    print('Test accuracy score:',logit.score(X_test, y_test))
    print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
    scores = cross_val_score(logit, X_test, y_test, cv=5, scoring='accuracy')
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
plt.errorbar(c_range, mean_error, yerr = std_error, linewidth=3)
plt.xlabel('C', fontsize=25)
plt.ylabel('Accuracy Score', fontsize=25)
title_cv = "Logistic Regression - Cross Validation \nFor Degree = 1"
plt.title(title_cv, fontsize=25)
plt.show()


#    Feature Importance
cols = X.columns
cols = np.asarray(cols)

plt.figure(figsize=(30,15))
feature_importance = abs(logit.coef_[0])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
top_features = pd.DataFrame({'feature_imp': feature_importance, 
                             'features': cols}, columns=['feature_imp', 'features'])
top_features = top_features.sort_values(by='feature_imp', ascending=False).head(20)
plt.bar(top_features.features, top_features.feature_imp)
plt.xlabel('Importance', fontsize=35, fontweight='bold')
plt.ylabel('Feature', fontsize=35, fontweight='bold')
plt.xticks(fontsize=30, rotation = 90)
plt.yticks(fontsize=30)
plt.show()

plt.figure(figsize=(30,15))
feature_importance = abs(logit.coef_[1])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
top_features = pd.DataFrame({'feature_imp': feature_importance, 
                             'features': cols}, columns=['feature_imp', 'features'])
top_features = top_features.sort_values(by='feature_imp', ascending=False).head(20)
plt.bar(top_features.features, top_features.feature_imp)
plt.xlabel('Importance', fontsize=35, fontweight='bold')
plt.ylabel('Feature', fontsize=35, fontweight='bold')
plt.xticks(fontsize=30, rotation = 90)
plt.yticks(fontsize=30)
plt.show()

plt.figure(figsize=(30,15))
feature_importance = abs(logit.coef_[2])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
top_features = pd.DataFrame({'feature_imp': feature_importance, 
                             'features': cols}, columns=['feature_imp', 'features'])
top_features = top_features.sort_values(by='feature_imp', ascending=False).head(20)
plt.bar(top_features.features, top_features.feature_imp)
plt.xlabel('Importance', fontsize=35, fontweight='bold')
plt.ylabel('Feature', fontsize=35, fontweight='bold')
plt.xticks(fontsize=30, rotation = 90)
plt.yticks(fontsize=30)
plt.show()


#    ROC-AUC Curve for all three categories
visualizer = ROCAUC(logit, classes=["0", "1", "2"], macro=False, micro=False)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)        
visualizer.show() 

#    Polynomial Degree and Error Plots
c_range = [0.001, 0.1, 1, 10, 100, 1000]
degree_range = [2]

for i in degree_range:
    trans = PolynomialFeatures(degree = i)
    x_poly = trans.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size = 0.2, random_state=(1))
    mean_error = []
    std_error = []
    for c in c_range:
        log_reg = LogisticRegression(C = c, random_state=0, solver='newton-cg',multi_class='multinomial')
        log_reg.fit(x_train, y_train)
        y_pred = log_reg.predict(x_test)
        
        cnf_mtx = metrics.confusion_matrix(y_test, y_pred)
        f1_score = (2*cnf_mtx[1][1])/((2*cnf_mtx[1][1]) + cnf_mtx[0][1] + cnf_mtx[1][0])
        
        scores = cross_val_score(log_reg, x_test, y_test, cv=5, scoring='accuracy')
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
        
        
        print(" Logistic Regression")
        print(" For Degree = ", i)
        print(" For C = ", c)
        print(" Confusion Matrix - \n", cnf_mtx)
        print(' Train accuracy score: ', log_reg.score(x_train, y_train))
        print(' Test accuracy score: ', log_reg.score(x_test, y_test))
        print(" F1 Score = ", f1_score)
        print(" Classification Report\n", classification_report(y_test, y_pred))
        print("\n")
    
    plt.errorbar(c_range, mean_error, yerr = std_error, linewidth=3)
    plt.xlabel('C', fontsize=25)
    plt.ylabel('Accuracy Score', fontsize=25)
    title_cv = f"Logistic Regression - Cross Validation \nFor Degree = {i}"
    plt.title(title_cv, fontsize=25)
    plt.show()


#    k-NN Classifier
nn_range = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
x_train_nn, x_test_nn, y_train_nn, y_test_nn = train_test_split(X, y, test_size = 0.2, random_state=(1))
merr = []
serr = []

for nn in nn_range:
    knn_model = KNeighborsClassifier(n_neighbors=nn, weights='uniform')
    knn_model.fit(x_train_nn, y_train_nn)
    y_pred_nn = knn_model.predict(x_test_nn)
    print("NN = ", nn)
    print('Train accuracy score:',knn_model.score(x_train_nn, y_train_nn))
    print('Test accuracy score:',knn_model.score(x_test_nn, y_test_nn))
    
    scores_knn = cross_val_score(knn_model, x_test_nn, y_test_nn, cv=5, scoring='accuracy')
    merr.append(np.array(scores_knn).mean())
    serr.append(np.array(scores_knn).std())

plt.errorbar(nn_range, merr, yerr = serr, linewidth=3)
plt.xlabel('NN', fontsize=25)
plt.ylabel('Accuracy Score', fontsize=25)
title_cv = f"k-NN - Cross Validation"
plt.title(title_cv, fontsize=25)
plt.show()


#    ROC-AUC Curve
visualizer = ROCAUC(knn_model, classes=["0", "1", "2"], macro=False, micro=False)

visualizer.fit(x_train_nn, y_train_nn)
visualizer.score(x_test_nn, y_test_nn)        
visualizer.show() 


#   Review Scores Rating
X = listings[
                ['host_response_time', 'host_response_rate', 'host_acceptance_rate',
                 'bedrooms', 'beds','neighbourhood_cleansed',
                 'host_is_superhost', 'host_listings_count', 'host_total_listings_count',
                 'host_identity_verified', 'room_type',
                 'accommodates','price', 'minimum_nights', 'maximum_nights',
                 'bath-products','electric-system',
                 'food-services','house-furniture','house-rules',
                 'kitchen-appliances','parking','recreation','safety',
                 'host_email','host_work_email'] + list(reviews.columns[2:])
]

y = listings[['review_scores_rating']]
y = (y/y.max())*100

y = y.assign(
    rating_bin_ep = pd.qcut(
        y['review_scores_rating'],
        q=3,
        duplicates='drop',
        labels=[0,1,2]
    )
)


#    Min Max of Each Bin
y.groupby('rating_bin_ep').min()
y.groupby('rating_bin_ep').max()


#    Splitting Data in 75-25 Ratio
y = y['rating_bin_ep']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


#    Number of Records in Each Bin
cnt_plt = sns.countplot(y)
cnt_plt.bar_label(cnt_plt.containers[0])
plt.show()


#    Logistic Regression - Varied C Range, using 'newton-cg' solver and multi_class='multinomial'
c_range = [0.001, 0.1, 1, 10, 100, 1000]
mean_error = []
std_error = []
for c in sorted(c_range):
    logit = LogisticRegression(C=c, random_state=0, solver='newton-cg',multi_class='multinomial')
    logit.fit(X_train, y_train)
    y_pred = logit.predict(X_test)
    print("C = ",c)
    print('Train accuracy score:',logit.score(X_train, y_train))
    print('Test accuracy score:',logit.score(X_test, y_test))
    print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
    scores = cross_val_score(logit, X_test, y_test, cv=5, scoring='accuracy')
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
plt.errorbar(c_range, mean_error, yerr = std_error, linewidth=3)
plt.xlabel('C', fontsize=25)
plt.ylabel('Accuracy Score', fontsize=25)
title_cv = "Logistic Regression - Cross Validation \nFor Degree = 1"
plt.title(title_cv, fontsize=25)
plt.show()


#    Feature Importance
cols = X.columns
cols = np.asarray(cols)

plt.figure(figsize=(30,15))
feature_importance = abs(logit.coef_[0])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
top_features = pd.DataFrame({'feature_imp': feature_importance, 
                             'features': cols}, columns=['feature_imp', 'features'])
top_features = top_features.sort_values(by='feature_imp', ascending=False).head(20)
plt.bar(top_features.features, top_features.feature_imp)
plt.xlabel('Importance', fontsize=35, fontweight='bold')
plt.ylabel('Feature', fontsize=35, fontweight='bold')
plt.xticks(fontsize=30, rotation = 90)
plt.yticks(fontsize=30)
plt.show()

plt.figure(figsize=(30,15))
feature_importance = abs(logit.coef_[1])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
top_features = pd.DataFrame({'feature_imp': feature_importance, 
                             'features': cols}, columns=['feature_imp', 'features'])
top_features = top_features.sort_values(by='feature_imp', ascending=False).head(20)
plt.bar(top_features.features, top_features.feature_imp)
plt.xlabel('Importance', fontsize=35, fontweight='bold')
plt.ylabel('Feature', fontsize=35, fontweight='bold')
plt.xticks(fontsize=30, rotation = 90)
plt.yticks(fontsize=30)
plt.show()

plt.figure(figsize=(30,15))
feature_importance = abs(logit.coef_[2])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
top_features = pd.DataFrame({'feature_imp': feature_importance, 
                             'features': cols}, columns=['feature_imp', 'features'])
top_features = top_features.sort_values(by='feature_imp', ascending=False).head(20)
plt.bar(top_features.features, top_features.feature_imp)
plt.xlabel('Importance', fontsize=35, fontweight='bold')
plt.ylabel('Feature', fontsize=35, fontweight='bold')
plt.xticks(fontsize=30, rotation = 90)
plt.yticks(fontsize=30)
plt.show()


#    ROC-AUC Curve for all three categories
visualizer = ROCAUC(logit, classes=["0", "1", "2"], macro=False, micro=False)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)        
visualizer.show() 


#    Polynomial Degree and Error Plots
c_range = [0.001, 0.1, 1, 10, 100, 1000]
degree_range = [2]

for i in degree_range:
    trans = PolynomialFeatures(degree = i)
    x_poly = trans.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size = 0.2, random_state=(1))
    mean_error = []
    std_error = []
    for c in c_range:
        log_reg = LogisticRegression(C = c, random_state=0, solver='newton-cg',multi_class='multinomial')
        log_reg.fit(x_train, y_train)
        y_pred = log_reg.predict(x_test)
        
        cnf_mtx = metrics.confusion_matrix(y_test, y_pred)
        f1_score = (2*cnf_mtx[1][1])/((2*cnf_mtx[1][1]) + cnf_mtx[0][1] + cnf_mtx[1][0])
        
        scores = cross_val_score(log_reg, x_test, y_test, cv=5, scoring='accuracy')
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
        
        
        print(" Logistic Regression")
        print(" For Degree = ", i)
        print(" For C = ", c)
        print(" Confusion Matrix - \n", cnf_mtx)
        print(' Train accuracy score: ', log_reg.score(x_train, y_train))
        print(' Test accuracy score: ', log_reg.score(x_test, y_test))
        print(" F1 Score = ", f1_score)
        print(" Classification Report\n", classification_report(y_test, y_pred))
        print("\n")
    
    plt.errorbar(c_range, mean_error, yerr = std_error, linewidth=3)
    plt.xlabel('C', fontsize=25)
    plt.ylabel('Accuracy Score', fontsize=25)
    title_cv = f"Logistic Regression - Cross Validation \nFor Degree = {i}"
    plt.title(title_cv, fontsize=25)
    plt.show()


#    k-NN Classifier
nn_range = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
x_train_nn, x_test_nn, y_train_nn, y_test_nn = train_test_split(X, y, test_size = 0.2, random_state=(1))
merr = []
serr = []

for nn in nn_range:
    knn_model = KNeighborsClassifier(n_neighbors=nn, weights='uniform')
    knn_model.fit(x_train_nn, y_train_nn)
    y_pred_nn = knn_model.predict(x_test_nn)
    print("NN = ", nn)
    print('Train accuracy score:',knn_model.score(x_train_nn, y_train_nn))
    print('Test accuracy score:',knn_model.score(x_test_nn, y_test_nn))
    
    scores_knn = cross_val_score(knn_model, x_test_nn, y_test_nn, cv=5, scoring='accuracy')
    merr.append(np.array(scores_knn).mean())
    serr.append(np.array(scores_knn).std())

plt.errorbar(nn_range, merr, yerr = serr, linewidth=3)
plt.xlabel('NN', fontsize=25)
plt.ylabel('Accuracy Score', fontsize=25)
title_cv = f"k-NN - Cross Validation"
plt.title(title_cv, fontsize=25)
plt.show()


#    ROC-AUC Curve
visualizer = ROCAUC(knn_model, classes=["0", "1", "2"], macro=False, micro=False)

visualizer.fit(x_train_nn, y_train_nn)
visualizer.score(x_test_nn, y_test_nn)        
visualizer.show() 


#   Review Scores Value
X = listings[
                ['host_response_time', 'host_response_rate', 'host_acceptance_rate',
                 'bedrooms', 'beds','neighbourhood_cleansed',
                 'host_is_superhost', 'host_listings_count', 'host_total_listings_count',
                 'host_identity_verified', 'room_type',
                 'accommodates','price', 'minimum_nights', 'maximum_nights',
                 'bath-products','electric-system',
                 'food-services','house-furniture','house-rules',
                 'kitchen-appliances','parking','recreation','safety',
                 'host_email','host_work_email'] + list(reviews.columns[2:])
]

y = listings[['review_scores_value']]
y = (y/y.max())*100

y = y.assign(
    rating_bin_ep = pd.qcut(
        y['review_scores_value'],
        q=3,
        duplicates='drop',
        labels=[0,1,2]
    )
)


#    Min Max of Each Bin
y.groupby('rating_bin_ep').min()
y.groupby('rating_bin_ep').max()


#    Splitting Data in 75-25 Ratio
y = y['rating_bin_ep']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


#    Number of Records in Each Bin
cnt_plt = sns.countplot(y)
cnt_plt.bar_label(cnt_plt.containers[0])
plt.show()


#    Logistic Regression - Varied C Range, using 'newton-cg' solver and multi_class='multinomial'
c_range = [0.001, 0.1, 1, 10, 100, 1000]
mean_error = []
std_error = []
for c in sorted(c_range):
    logit = LogisticRegression(C=c, random_state=0, solver='newton-cg',multi_class='multinomial')
    logit.fit(X_train, y_train)
    y_pred = logit.predict(X_test)
    print("C = ",c)
    print('Train accuracy score:',logit.score(X_train, y_train))
    print('Test accuracy score:',logit.score(X_test, y_test))
    print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
    scores = cross_val_score(logit, X_test, y_test, cv=5, scoring='accuracy')
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
plt.errorbar(c_range, mean_error, yerr = std_error, linewidth=3)
plt.xlabel('C', fontsize=25)
plt.ylabel('Accuracy Score', fontsize=25)
title_cv = "Logistic Regression - Cross Validation \nFor Degree = 1"
plt.title(title_cv, fontsize=25)
plt.show()


#    Feature Importance
cols = X.columns
cols = np.asarray(cols)

plt.figure(figsize=(30,15))
feature_importance = abs(logit.coef_[0])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
top_features = pd.DataFrame({'feature_imp': feature_importance, 
                             'features': cols}, columns=['feature_imp', 'features'])
top_features = top_features.sort_values(by='feature_imp', ascending=False).head(20)
plt.bar(top_features.features, top_features.feature_imp)
plt.xlabel('Importance', fontsize=35, fontweight='bold')
plt.ylabel('Feature', fontsize=35, fontweight='bold')
plt.xticks(fontsize=30, rotation = 90)
plt.yticks(fontsize=30)
plt.show()

plt.figure(figsize=(30,15))
feature_importance = abs(logit.coef_[1])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
top_features = pd.DataFrame({'feature_imp': feature_importance, 
                             'features': cols}, columns=['feature_imp', 'features'])
top_features = top_features.sort_values(by='feature_imp', ascending=False).head(20)
plt.bar(top_features.features, top_features.feature_imp)
plt.xlabel('Importance', fontsize=35, fontweight='bold')
plt.ylabel('Feature', fontsize=35, fontweight='bold')
plt.xticks(fontsize=30, rotation = 90)
plt.yticks(fontsize=30)
plt.show()

plt.figure(figsize=(30,15))
feature_importance = abs(logit.coef_[2])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
top_features = pd.DataFrame({'feature_imp': feature_importance, 
                             'features': cols}, columns=['feature_imp', 'features'])
top_features = top_features.sort_values(by='feature_imp', ascending=False).head(20)
plt.bar(top_features.features, top_features.feature_imp)
plt.xlabel('Importance', fontsize=35, fontweight='bold')
plt.ylabel('Feature', fontsize=35, fontweight='bold')
plt.xticks(fontsize=30, rotation = 90)
plt.yticks(fontsize=30)
plt.show()


#    ROC-AUC Curve for all three categories
visualizer = ROCAUC(logit, classes=["0", "1", "2"], macro=False, micro=False)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)        
visualizer.show() 

#    Polynomial Degree and Error Plots
c_range = [0.001, 0.1, 1, 10, 100, 1000]
degree_range = [2]

for i in degree_range:
    trans = PolynomialFeatures(degree = i)
    x_poly = trans.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size = 0.2, random_state=(1))
    mean_error = []
    std_error = []
    for c in c_range:
        log_reg = LogisticRegression(C = c, random_state=0, solver='newton-cg',multi_class='multinomial')
        log_reg.fit(x_train, y_train)
        y_pred = log_reg.predict(x_test)
        
        cnf_mtx = metrics.confusion_matrix(y_test, y_pred)
        f1_score = (2*cnf_mtx[1][1])/((2*cnf_mtx[1][1]) + cnf_mtx[0][1] + cnf_mtx[1][0])
        
        scores = cross_val_score(log_reg, x_test, y_test, cv=5, scoring='accuracy')
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
        
        
        print(" Logistic Regression")
        print(" For Degree = ", i)
        print(" For C = ", c)
        print(" Confusion Matrix - \n", cnf_mtx)
        print(' Train accuracy score: ', log_reg.score(x_train, y_train))
        print(' Test accuracy score: ', log_reg.score(x_test, y_test))
        print(" F1 Score = ", f1_score)
        print(" Classification Report\n", classification_report(y_test, y_pred))
        print("\n")
    
    plt.errorbar(c_range, mean_error, yerr = std_error, linewidth=3)
    plt.xlabel('C', fontsize=25)
    plt.ylabel('Accuracy Score', fontsize=25)
    title_cv = f"Logistic Regression - Cross Validation \nFor Degree = {i}"
    plt.title(title_cv, fontsize=25)
    plt.show()


#    k-NN Classifier
nn_range = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
x_train_nn, x_test_nn, y_train_nn, y_test_nn = train_test_split(X, y, test_size = 0.2, random_state=(1))
merr = []
serr = []

for nn in nn_range:
    knn_model = KNeighborsClassifier(n_neighbors=nn, weights='uniform')
    knn_model.fit(x_train_nn, y_train_nn)
    y_pred_nn = knn_model.predict(x_test_nn)
    print("NN = ", nn)
    print('Train accuracy score:',knn_model.score(x_train_nn, y_train_nn))
    print('Test accuracy score:',knn_model.score(x_test_nn, y_test_nn))
    
    scores_knn = cross_val_score(knn_model, x_test_nn, y_test_nn, cv=5, scoring='accuracy')
    merr.append(np.array(scores_knn).mean())
    serr.append(np.array(scores_knn).std())

plt.errorbar(nn_range, merr, yerr = serr, linewidth=3)
plt.xlabel('NN', fontsize=25)
plt.ylabel('Accuracy Score', fontsize=25)
title_cv = f"k-NN - Cross Validation"
plt.title(title_cv, fontsize=25)
plt.show()


#    ROC-AUC Curve
visualizer = ROCAUC(knn_model, classes=["0", "1", "2"], macro=False, micro=False)
visualizer.fit(x_train_nn, y_train_nn)
visualizer.score(x_test_nn, y_test_nn)        
visualizer.show() 
