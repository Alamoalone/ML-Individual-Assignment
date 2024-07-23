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
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.model_selection import cross_val_score

def processing_and_merge(listings, reviews):

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

  host_verification = listings['host_verifications'].str.split(',', expand=True)

  listings.loc[listings['host_verifications'].str.contains('email'),
              'host_email'] = 1
  listings.loc[listings['host_verifications'].str.contains('phone'),
              'host_phone'] = 1
  listings.loc[listings['host_verifications'].str.contains('work_email'),
              'host_work_email'] = 1

  new_feature_cols = listings.iloc[:,75:].columns
  listings[new_feature_cols] = listings[new_feature_cols].fillna(0)
  listings = listings.merge(reviews, how='inner', left_on='id', right_on='listing_id')

  return listings

def plot_nas(df: pd.DataFrame):
    if df.isnull().sum().sum() != 0:
        na_df = (df.isnull().sum() / len(df)) * 100
        na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'NaN %' :na_df})
        missing_data.plot(kind = "bar")
        plt.show()
    else:
        print('No NAs found')

def drop_useless_cols(listings):
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

  imputation_cols = ['bedrooms', 'beds']
  for i in imputation_cols:
      listings.loc[listings.loc[:,i].isnull(),i] = listings.loc[:,i].median()

  return listings

def plot_price_box(listings):
  # Step 1: Clean the `price` column
  listings['price'] = listings['price'].astype(str).str.replace('$', '').str.replace(',', '')
  listings['price'] = pd.to_numeric(listings['price'])

  # Step 2: Clean the `host_response_rate` and `host_acceptance_rate` columns
  listings['host_response_rate'] = listings["host_response_rate"].str.replace("%","")
  listings['host_response_rate'] = pd.to_numeric(listings['host_response_rate'])
  listings['host_acceptance_rate'] = listings["host_acceptance_rate"].str.replace("%","")
  listings['host_acceptance_rate'] = pd.to_numeric(listings['host_acceptance_rate'])

  # Step 3: Plot the initial price boxplot
  plt.figure(figsize=(10, 5))
  listings[['price']].plot(kind='box', title='Price BoxPlot')
  plt.ylim(0, 1600)
  plt.show()

  # Step 4: Filter the `price` column data
  listings_filtered = listings[(listings.price > 50) & (listings.price <= 300)]

  # Step 5: Plot the filtered price boxplot
  plt.figure(figsize=(10, 5))
  listings_filtered[['price']].plot(kind='box', title='Filtered Price BoxPlot')
  plt.show()

  listings_filtered.head()

  # Plot the distribution of host_response_rate
  plt.figure(figsize=(10, 6))
  listings['host_response_rate'].plot(kind='hist', bins=20, edgecolor='black', alpha=0.7)
  plt.title('Distribution of Host Response Rate')
  plt.xlabel('Host Response Rate (%)')
  plt.ylabel('Frequency')
  plt.grid(axis='y', linestyle='--', alpha=0.7)
  plt.show()

  return listings

def plot_review_cols(listings):
  # Convert review scores to numeric and fill NaNs with the mean value of each column
  review_score_columns = [
    'review_scores_accuracy',
    'review_scores_checkin',
    'review_scores_cleanliness',
    'review_scores_communication',
    'review_scores_location',
    'review_scores_rating',
    'review_scores_value'
  ]

  # Ensure all columns are numeric and fill NaNs with the mean value
  for col in review_score_columns:
      listings[col] = pd.to_numeric(listings[col], errors='coerce')
      listings[col].fillna(listings[col].mean(), inplace=True)

  # Plot the distribution of each review score column
  fig, axes = plt.subplots(3, 3, figsize=(18, 18))

  # Define positions to leave a gap in the grid
  positions = [1, 2, 3, 4, 5, 6, 8]  # Subplot positions in a 3x3 grid

  for i, (col, pos) in enumerate(zip(review_score_columns, positions)):
      ax = plt.subplot(3, 3, pos)
      listings[col].plot(kind='hist', bins=20, edgecolor='black', alpha=0.7, ax=ax)
      ax.set_title(f'Distribution of {col}')
      ax.set_xlabel(f'{col}')
      ax.set_ylabel('Frequency')
      ax.grid(axis='y', linestyle='--', alpha=0.7)

  plt.tight_layout()
  plt.show()

def plot_neighborhood(listings):
  # Calculate the number of listings in each neighborhood
  neighbourhood_DF = listings.groupby('neighbourhood_cleansed').host_response_time.count().reset_index()
  neighbourhood_DF = neighbourhood_DF.rename(columns={'host_response_time': 'Number_Of_Listings'})

  # Sort by the number of listings
  neighbourhood_DF = neighbourhood_DF.sort_values(by='Number_Of_Listings', ascending=False)

  # Plot the bar chart
  plt.figure(figsize=(18, 8))
  plt.bar(neighbourhood_DF['neighbourhood_cleansed'], neighbourhood_DF['Number_Of_Listings'])
  plt.title('Dublin Neighborhood Frequency')
  plt.xlabel('Neighborhood')
  plt.ylabel('Number of Listings')
  plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
  plt.show()

def statics_review_score(listings):
  # Define the list of review score columns
  review_score_columns = [
      'review_scores_accuracy',
      'review_scores_checkin',
      'review_scores_cleanliness',
      'review_scores_communication',
      'review_scores_location',
      'review_scores_rating',
      'review_scores_value'
  ]

  # Calculate and print statistics for each review score column
  for col in review_score_columns:
      print(f"{col} statistics:")
      print(f"Mean: {listings[col].mean()}")
      print(f"Median: {listings[col].median()}")
      print(f"Mode: {listings[col].mode()[0]}")
      print(f"Min: {listings[col].min()}")
      print(f"Max: {listings[col].max()}")
      print("\n")

def plot_Kernel_Density_Estimate(listings):
  plt.figure(figsize=(18, 18))

  # Define the list of review score columns and their subplot positions
  review_score_columns = [
      'review_scores_accuracy',
      'review_scores_checkin',
      'review_scores_cleanliness',
      'review_scores_communication',
      'review_scores_location',
      'review_scores_rating',
      'review_scores_value'
  ]

  positions = [1, 2, 3, 4, 5, 6, 8]  # Subplot positions in a 3x3 grid

  for col, pos in zip(review_score_columns, positions):
      plt.subplot(3, 3, pos)
      density = sns.kdeplot(listings[col], color="b", label=f'{col} - density')
      x = density.lines[0].get_xdata()
      y = density.lines[0].get_ydata()
      mean = listings[col].mean()
      height = np.interp(mean, x, y)
      density.vlines(mean, 0, height, color='b', ls=':')
      density.fill_between(x, 0, y, facecolor='b', alpha=0.2)
      plt.legend()

  plt.tight_layout()
  plt.show()

def minmax(X):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (X.max() - X.min()) + X.min()
    return X_scaled

def scaling_data(listings):
  scaling_data = ['host_response_rate', 'host_acceptance_rate', 'bedrooms', 'beds',
                'host_listings_count', 'host_total_listings_count',
                'latitude', 'longitude', 'accommodates', 'price',
                'minimum_nights', 'maximum_nights', 'number_of_reviews', 'num_cmt', #'avg_senti',
                'review_scores_rating', 'review_scores_accuracy',
                'review_scores_cleanliness', 'review_scores_checkin',
                'review_scores_communication', 'review_scores_location',
                'review_scores_value', 'reviews_per_month','bath-products','electric-system',
                'food-services','house-furniture','house-rules',
                'kitchen-appliances','parking','recreation','safety',
                'host_email','host_work_email']

  for i in scaling_data:
    listings[i] = minmax(listings[i])

  listings.dropna(axis = 1, inplace = True)
  label_encoder = preprocessing.LabelEncoder()
  listings.host_response_time     = label_encoder.fit_transform(listings.host_response_time)
  listings.host_is_superhost      = label_encoder.fit_transform(listings.host_is_superhost)
  listings.host_identity_verified = label_encoder.fit_transform(listings.host_identity_verified)
  listings.instant_bookable       = label_encoder.fit_transform(listings.instant_bookable)
  listings.room_type              = label_encoder.fit_transform(listings.room_type)
  listings.neighbourhood_cleansed = label_encoder.fit_transform(listings.neighbourhood_cleansed)
  listings.has_availability       = label_encoder.fit_transform(listings.has_availability)

  test_corr = listings.corr()
  test_corr.to_csv("test_corr.csv")

  return listings

def plot_confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

def ROCAUC_visualizer(model, bin_count, X, y):
    y = y['rating_bin_ep'].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    visualizer = ROCAUC(model, classes=list(range(bin_count)), macro=False, micro=False)
    visualizer.fit(x_train, y_train)
    visualizer.score(x_test, y_test)
    visualizer.show()

def bin_column(listings, col, n_bins):
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

  y = listings[[col]]
  y = (y/y.max())*100

  y = y.assign(
      rating_bin_ep = pd.qcut(
          y[col],
          q=n_bins,
          duplicates='drop',
          labels=list(range(n_bins))
      )
  )

  y.groupby('rating_bin_ep').min()
  y.groupby('rating_bin_ep').max()

  return X , y

def select_important_features(X, y, threshold=0.05):

    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    selected_features = correlations[correlations > threshold].index

    plt.figure(figsize=(10, 6))
    sns.barplot(x=selected_features, y=correlations[selected_features], palette="viridis")
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.xlabel('Features')
    plt.ylabel('Correlation with target')
    plt.title('Selected Features and Their Correlations')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

    return selected_features, correlations

def check_bins(y):
  y1 = y['rating_bin_ep']
  cnt_plt = sns.countplot(y1)
  cnt_plt.bar_label(cnt_plt.containers[0])
  plt.show()

def evaluate_logistic_regression(X, y, c_range, degree_range=[1], bin_count=2):

    y1 = y['rating_bin_ep']
    for i in degree_range:
        trans = PolynomialFeatures(degree=i)
        x_poly = trans.fit_transform(X)
        x_train, x_test, y_train, y_test = train_test_split(x_poly, y1, test_size=0.2, random_state=1)
        mean_error = []
        std_error = []
        for c in c_range:
            log_reg = LogisticRegression(C=c, random_state=0, solver='newton-cg', multi_class='multinomial')
            log_reg.fit(x_train, y_train)
            y_pred = log_reg.predict(x_test)

            cnf_mtx = confusion_matrix(y_test, y_pred)
            f1_score = metrics.f1_score(y_test, y_pred, average='weighted')

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
            print(" Mean Squared Error = ", mean_squared_error(y_test, y_pred))
            print(" Classification Report\n", classification_report(y_test, y_pred))
            print("\n")

            title = f"Logistic Regression Confusion Matrix\nFor Degree = {i} and C = {c}"
            plot_confusion_matrix(y_test, y_pred, title)

        plt.errorbar(c_range, mean_error, yerr=std_error, linewidth=3)
        plt.xlabel('C', fontsize=25)
        plt.ylabel('Accuracy Score', fontsize=25)
        title_cv = f"Logistic Regression - Cross Validation \nFor Degree = {i}"
        plt.title(title_cv, fontsize=25)
        plt.show()

        ROCAUC_visualizer(log_reg, bin_count, x_poly, y)

def evaluate_knn(X, y, nn_range, bin_count=2):
    y1 = y['rating_bin_ep']
    x_train, x_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=1)
    merr = []
    serr = []

    for nn in nn_range:
        knn_model = KNeighborsClassifier(n_neighbors=nn, weights='uniform')
        knn_model.fit(x_train, y_train)
        y_pred_nn = knn_model.predict(x_test)

        cnf_mtx = confusion_matrix(y_test, y_pred_nn)
        f1_score = metrics.f1_score(y_test, y_pred_nn, average='weighted')

        scores_knn = cross_val_score(knn_model, x_test, y_test, cv=5, scoring='accuracy')
        merr.append(np.array(scores_knn).mean())
        serr.append(np.array(scores_knn).std())

        print(" K Neighbors Classifier")
        print(" For NN = ", nn)
        print(" Confusion Matrix - \n", cnf_mtx)
        print(' Train accuracy score: ', knn_model.score(x_train, y_train))
        print(' Test accuracy score: ', knn_model.score(x_test, y_test))
        print(" F1 Score = ", f1_score)
        print(" Mean Squared Error = ", mean_squared_error(y_test, y_pred_nn))
        print(" Classification Report\n", classification_report(y_test, y_pred_nn))
        print("\n")

        title = f"k-NN Confusion Matrix\nFor NN = {nn}"
        plot_confusion_matrix(y_test, y_pred_nn, title)

    plt.errorbar(nn_range, merr, yerr=serr, linewidth=3)
    plt.xlabel('NN', fontsize=25)
    plt.ylabel('Accuracy Score', fontsize=25)
    title_cv = f"k-NN - Cross Validation"
    plt.title(title_cv, fontsize=25)
    plt.show()

    ROCAUC_visualizer(knn_model, bin_count, X, y)

def evaluate_decision_tree(X, y, depth_range, bin_count=2):
    y1 = y['rating_bin_ep']
    x_train, x_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=1)
    merr_dt = []
    serr_dt = []

    for depth in depth_range:
        dt_model = DecisionTreeClassifier(max_depth=depth, random_state=1)
        dt_model.fit(x_train, y_train)
        y_pred_dt = dt_model.predict(x_test)

        cnf_mtx = confusion_matrix(y_test, y_pred_dt)
        f1_score = metrics.f1_score(y_test, y_pred_dt, average='weighted')

        scores_dt = cross_val_score(dt_model, x_test, y_test, cv=5, scoring='accuracy')
        merr_dt.append(np.array(scores_dt).mean())
        serr_dt.append(np.array(scores_dt).std())

        print(" Decision Tree Classifier")
        print(" For Depth = ", depth)
        print(" Confusion Matrix - \n", cnf_mtx)
        print(' Train accuracy score: ', dt_model.score(x_train, y_train))
        print(' Test accuracy score: ', dt_model.score(x_test, y_test))
        print(" F1 Score = ", f1_score)
        print(" Mean Squared Error = ", mean_squared_error(y_test, y_pred_dt))
        print(" Classification Report\n", classification_report(y_test, y_pred_dt))
        print("\n")

        title = f"Decision Tree Confusion Matrix\nFor Depth = {depth}"
        plot_confusion_matrix(y_test, y_pred_dt, title)

    plt.errorbar(depth_range, merr_dt, yerr=serr_dt, linewidth=3)
    plt.xlabel('Depth', fontsize=25)
    plt.ylabel('Accuracy Score', fontsize=25)
    title_cv = f"Decision Tree - Cross Validation"
    plt.title(title_cv, fontsize=25)
    plt.show()

    ROCAUC_visualizer(dt_model, bin_count, X, y)

def plot_kde_by_bin(X, y, feature, bin_column, bins=3, ylim=(0, 19), xlim=(-0.25, 1)):
    test = pd.concat([X, y], axis=1)
    colors = ['r', 'g', 'b']
    labels = [f'{feature}, bin={i}' for i in range(bins)]

    for i in range(bins):
        sns.kdeplot(test.loc[test[bin_column] == i, feature], color=colors[i], label=labels[i]).set(ylim=ylim, xlim=xlim)
        plt.legend()

    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.title(f'KDE of {feature} by {bin_column}')
    plt.show()

def plot_super_host(X, y):
  test = pd.concat([X, y], axis=1)
  test = test.groupby(['host_is_superhost', 'rating_bin_ep']).size().unstack()

  plt.figure(figsize=(18, 6))

  plt.subplot(1, 2, 1)
  for j in range(3):
      plt.bar(j, test.loc[1, j], label=f'Bin {j}', alpha=0.7)
  plt.title("Host is a Super Host")
  plt.legend()
  plt.xlabel('Review Bin', fontweight='bold', fontsize=15)
  plt.ylabel('# Records', fontweight='bold', fontsize=15)

  plt.subplot(1, 2, 2)
  for j in range(3):
      plt.bar(j, test.loc[0, j], label=f'Bin {j}', alpha=0.7)
  plt.title("Host is not a Super Host")
  plt.legend()
  plt.xlabel('Review Bin', fontweight='bold', fontsize=15)
  plt.ylabel('# Records', fontweight='bold', fontsize=15)

  plt.show()

# load csv
listings = pd.read_csv("listings.csv")
reviews = pd.read_csv("reviews_final.csv")

# listings pre-processing
listings = processing_and_merge(listings, reviews)
plot_nas(listings)
listings = drop_useless_cols(listings)
listings = plot_price_box(listings)
plot_review_cols(listings)
plot_neighborhood(listings)
statics_review_score(listings)
plot_Kernel_Density_Estimate(listings)
listings = scaling_data(listings)

# model parmater
c_range = [0.001, 0.1, 1, 10, 100, 1000]
degree_range = [1, 2]
nn_range = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
depth_range = [3, 5, 7, 9, 11, 13, 15, 17, 19]

# review_scores_checkin evaluating
X,y = bin_column(listings, 'review_scores_checkin', 2)
check_bins(y)
yy = listings['review_scores_checkin']
select_important_features(X, yy, threshold=0.01)
evaluate_logistic_regression(X, y, c_range, degree_range, 2)
evaluate_knn(X, y, nn_range, 2)
evaluate_decision_tree(X, y, depth_range, 2)

# review_scores_communication evaluating
X,y = bin_column(listings, 'review_scores_communication', 2)
check_bins(y)
yy = listings['review_scores_communication']
select_important_features(X, yy, threshold=0.01)
evaluate_logistic_regression(X, y, c_range, degree_range, 2)
evaluate_knn(X, y, nn_range, 2)
evaluate_decision_tree(X, y, depth_range, 2)

# review_scores_cleanliness evaluating
X,y = bin_column(listings, 'review_scores_cleanliness', 3)
check_bins(y)
yy = listings['review_scores_cleanliness']
select_important_features(X, yy, threshold=0.01)
evaluate_logistic_regression(X, y, c_range, degree_range, 3)
evaluate_knn(X, y, nn_range, 3)
evaluate_decision_tree(X, y, depth_range, 3)

# review_scores_location evaluating
X,y = bin_column(listings, 'review_scores_location', 3)
check_bins(y)
yy = listings['review_scores_location']
select_important_features(X, yy, threshold=0.01)
evaluate_logistic_regression(X, y, c_range, degree_range, 3)
evaluate_knn(X, y, nn_range, 3)
evaluate_decision_tree(X, y, depth_range, 3)

# review_scores_rating evaluating
X,y = bin_column(listings, 'review_scores_rating', 3)
check_bins(y)
yy = listings['review_scores_rating']
select_important_features(X, yy, threshold=0.01)
evaluate_logistic_regression(X, y, c_range, degree_range, 3)
evaluate_knn(X, y, nn_range, 3)
evaluate_decision_tree(X, y, depth_range, 3)

# review_scores_value evaluating
X,y = bin_column(listings, 'review_scores_value', 3)
check_bins(y)
yy = listings['review_scores_value']
select_important_features(X, yy, threshold=0.01)
evaluate_logistic_regression(X, y, c_range, degree_range, 3)
evaluate_knn(X, y, nn_range, 3)
evaluate_decision_tree(X, y, depth_range, 3)

# review_scores_accuracy evaluating
X,y = bin_column(listings, 'review_scores_accuracy', 3)
check_bins(y)
yy = listings['review_scores_accuracy']
select_important_features(X, yy, threshold=0.01)
evaluate_logistic_regression(X, y, c_range, degree_range, 3)
evaluate_knn(X, y, nn_range, 3)
evaluate_decision_tree(X, y, depth_range, 3)

# kde term home and num cmt
plot_kde_by_bin(X, y, 'term_home', 'rating_bin_ep', bins=3, ylim=(0, 19), xlim=(-0.25, 1))
plot_kde_by_bin(X, y, 'num_cmt', 'rating_bin_ep', bins=3, ylim=(0, 0.03), xlim=(-200, 1600))

# superhost 
plot_super_host(X, y)