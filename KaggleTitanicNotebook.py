# %% [markdown]
## Part 1 - Introduction - Look at the Big Picture
#
#In this notebook I'm attempting my first Kaggle competition.
#
### 1.1 - Frame The Problem
#
#This problem can be classified as follows:
#
#- A **classfication** problem : we are being asked to predict whether an individual dies or not based on a range of continuous and categorical features.  Indeed to be more specific this is a **univariate classifcaiton** problem : ie we are being asked to predcict a single label for each individual;
#- **Supervised** learning : we are given a training data set with labeled examples;
#- **Batch** Learning : we will run the models as a one off as the data is static and will never change;
#
#Wider non-functional requirements indicate:
#
#- Low data volume - 
#- Non-sensitive data - 
#- No legal or ethical concerns - 
#
#The types of model that are likely to support this problem are therefore:
#
#- X
#- Y
#- Z
# 
#Key things I wanted to achieve:
#
#1. Use of Microsoft's DataPrep SDK for Python rather than more traditional data wrangling methods - taking advantage of some of the advanced features such as "column by example";
#2. Start with application of a simple linear regression model and then build more sophisticated models later on.
#3. Implement a data pipeline using 
#
### 1.2 - Select a Performance Measure
#
#The performance measure is driven by Kaggle: that is the overall **accuracy** of the model - defined as the percentage of people who are labeled correctly.
#It would also be interesting to see more detailed metrics - ie the confusion matirix.
#Also want to use advanced techniques for hyperparameter tuning - more on this later.
# 
### 1.3 - List and Validate The Assumptions
# 
#- Assumption 1 - 
# 
### Pre-requisites
#
#The major non-standard modules used are:
# 
#The 'azurelm.dataprep' SDK.  To install the Microsoft Azure ML Dataprep package:
#
#`pip install azureml-dataprep`
#
#The "XG Boost" machine learning package:
#
#`pip install xgboost`
#
#Documentation is available at the following web sites:
#

# %%
import azureml.dataprep as dprep
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import datetime

# %% [markdown]
#Input data files are available in the "../input/" directory.
#For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#%%
for dirname, _, filenames in os.walk('./kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# %% [markdown]
#***
## Part 2 - Data Ingestion

# %%
#Load the train and test files as Pandas dataframes...
df_train = pd.read_csv('./kaggle/input/train.csv')
df_train.shape
# %%
df_train.info()

# %%
df_test = pd.read_csv('./kaggle/input/test.csv')
df_test.shape

# %%
dataflow = dprep.auto_read_file('./kaggle/input/train.csv')
dataflow.head(10)

# %%
dataflow_test = dprep.auto_read_file('./kaggle/input/test.csv')
dataflow_test.head(10)

# %% [markdown]
# ***
## Part 3 - Feature Engineering I
# 
#Feature engineering necessary to support exploratory data analysis (EDA) below.
#
#In this section a range of feature extraction actions are performed using Microsoft's ML Dataprep SDK for Python:
#1. Append the test data set to training data set so that feature engineering can be addressed collectively;
#2. Filling in empty values in "Cabin" and "Fare" where simple rules can be applied;
#3. Pulling "Title" from the full name of each passenger;
#4. Counting cabin occupancy and passengers who also shared the same ticket;
#5. Extract - pulling the "Level" from the cabin (if indeed the passenger had a cabin).
#
#Then use Pandas to:
#- Address missing values in "Cabin" and "Level" (where people have no "Cabin" assigned);
#- Create the training data set;
#- Extract the target labels from the training data set;
#- Create the test data set.

# %% [markdown]
#***
### Combine Train and Test Data Sets
#
# Not best practice, but for good reason - so that I can apply dataprep functionality effectively.
#
#%%
dataflow = dataflow.append_rows([dataflow_test]) 

# %% [markdown]
#***
### Address Missing Values in "Fare" and "Cabin"
#
# %%
dataflow = dataflow.replace('Cabin', '', 'None')

#%%
dataflow = dataflow.fill_nulls('Fare', 0)

# %% [markdown]
#***
### Creation of "Title" feature.
#
#Used the following to find specific titles in the name column - this helps to select examples to train the "derive by column" approach:
#
#`df_title_examples[df_title_examples['Name'].str.contains("Countess")]`

# %%
builder_title = dataflow.builders.derive_column_by_example(source_columns = ['Name'], new_column_name = 'Title')
builder_title.add_example(source_data = {'Name': 'Braund, Mr. Owen Harris'}, example_value = 'Mr')
builder_title.add_example(source_data = {'Name': 'Nasser, Mrs. Nicholas (Adele Achem)'}, example_value = 'Mrs')
builder_title.add_example(source_data = {'Name': 'Rice, Master. Eugene'}, example_value = 'Master')

#%%
# This example is important as it gives the model an example of a "double barelled" surname to learn from.
df_train.iloc[38]

#%%
builder_title.add_example(source_data = {'Name': 'Vander Planke, Miss. Augusta Maria'}, example_value = 'Miss')

#%%
# This is an example of a Countess which again is a bit different.
df_train.iloc[759]

#%%
builder_title.add_example(source_data = {'Name': 'Rothes, the Countess. of'}, example_value = 'Countess')
builder_title.preview() # Preview top 10 rows.

#%%
dataflow = builder_title.to_dataflow()

# %%
profile = dataflow.get_profile()
profile.columns['Title'].value_counts

# %% [markdown]
# What about the last entry "Jonkheer" is that a title?  According to (Wikipedia)[https://en.wikipedia.org/wiki/Jonkheer] it is!

# %%
df_train[df_train['Name'].str.contains('Jonkheer')].iloc[0]['Name']

# %% [markdown]
#***
### Creation of "Cabin Occupancy" Feature

#%%
# Add in a CabinOccupancy using a count.
dataflow = dataflow.summarize(
        summary_columns=[
            dprep.SummaryColumnsValue(
                column_id='PassengerId',
                summary_column_name='CabinOccupancy', 
                summary_function=dprep.SummaryFunction.COUNT)],
        group_by_columns=['Cabin'],
        join_back=True)

# %% [markdown]
#***
### Creation of "Level" Feature

# %%
dataflow.head(100)

# %%
builder_level = dataflow.builders.derive_column_by_example(source_columns = ['Cabin'], new_column_name = 'Level')
builder_level.add_example(source_data = {'Cabin': 'C85'}, example_value = 'C')
builder_level.add_example(source_data = {'Cabin': 'E46'}, example_value = 'E')
builder_level.add_example(source_data = {'Cabin': 'None'}, example_value = 'None')
builder_level.preview() # Preview top 10 rows.

# %%
dataflow = builder_level.to_dataflow()

# %% [markdown]
#***
### Create "People On Ticket" Feature

# %%
dataflow = dataflow.summarize(
        summary_columns=[
            dprep.SummaryColumnsValue(
                column_id='PassengerId',
                summary_column_name='PeopleOnTicket', 
                summary_function=dprep.SummaryFunction.COUNT)],
        group_by_columns=['Ticket'],
        join_back=True)

# %% [markdown]
#***
### Inspect Results and Create Output
#Have a look at the dataprep data flow and publish the results to a Pandas dataframe for downstream analysis and processing.
# %%
# Print out the final dataflow to show all the elements that it contains.
dataflow
# %%
profile = dataflow.get_profile()
profile
# %%
profile.columns['Title'].value_counts
# %%
profile.columns['CabinOccupancy'].value_counts
# %%
profile.columns['Level'].value_counts
# %%
profile.columns['PeopleOnTicket'].value_counts

# %% [markdown]
#***
### Fix Issue With Cabin Occupancy
#Couldn't figure out a way of doing this inside the dataprep data flow.  So need to set Null cabin counts to zero for both training and 
# %%
df = dataflow.to_pandas_dataframe()
# %%
df['CabinOccupancy'].value_counts()
# %%
# Fix issue with cabin oocupancy count:
df['CabinOccupancy'].loc[df['CabinOccupancy'] == 1014] = 0

# %%
df['CabinOccupancy'].value_counts()

# %%
# Output data to file...
df.to_csv('./titanic_data.csv')

# %% [markdown]
#***
## Part 4 - Inital Exploratory Data Analysis (EDA)
#
#Now going to work through each column in the data set to get an insight into it.

# %%
df.info()

# %%
# Generate initial view of data using historgram plot.
df.hist(bins=20, figsize=(12,12))

# %%
# Compute the correlation matrix
correlation_matrix = df.corr()
correlation_matrix['Survived'].sort_values(ascending=False)

# %%
def generate_box_swarm_plot(df, x, y, hue):
    # Set up global parameters
    sns.set_style("ticks", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
    plt.rcParams.update({'axes.titlesize' : 18, 'lines.linewidth' : 1.5,\
    'axes.labelsize' : 14, 'xtick.labelsize' : 12, 'ytick.labelsize' : 12})

    # Dynamically create the palette
    unique_count = df.loc[:,[hue]].groupby(hue).count().shape[0]
    dynamic_palette = sns.color_palette("RdBu_r", unique_count)

    plt.figure(figsize=(12,10))
    # Create a box blot
    chart = sns.boxplot(x=x, y=y, \
        data=df, orient="v", fliersize =0, color="lightgrey")
    # Overlay with swarm plot.
    chart = sns.swarmplot(x=x, y=y, \
        data=df, orient="v", hue=hue, \
            alpha=0.7, palette=dynamic_palette, size=6)
    plt.draw()
    chart.set_yticklabels(chart.get_yticklabels(), rotation=90, verticalalignment="center")
    plt.show()

# %%
generate_box_swarm_plot(df, 'Sex', 'Age', 'Survived')

# %%
generate_box_swarm_plot(df, 'Pclass', 'Age', 'Survived')

# %%
generate_box_swarm_plot(df, 'Title', 'Age', 'Survived')

# %%
generate_box_swarm_plot(df, 'CabinOccupancy', 'Age', 'Survived')

# %%
generate_box_swarm_plot(df, 'Level', 'Fare', 'Survived')

# %%
generate_box_swarm_plot(df, 'PeopleOnTicket', 'Age', 'Survived')

# %% [markdown]
#***
## Part 5 - Feature Engineering II
#
#Here I create a Sci-Kit data pipeline to apply any steps required to get into shape for the models.
#
#2. Feature Engineering - 
#This will include:
#- Build a Sci_kit Learn pipeline to perform a series of transformations:
#    - Imputing missing "Age" using regressor to match based on values in other columns;
#    - Min max scaling numerical values;
#    - One-hot Encoding categorical values;
#    - Non-linear transformation of the "Fare" feature.
#
# At each stage we will apply feature engineering to both the train and test data sets.
# %% [markdown]
#***
### Apply Sci-Kit Learn Pipeline To Transform Data
#
# %%
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# %%
# First build lists of the various columns that need different operations performed on them in the pipeline:
features_to_one_hot_encode = ['Sex', 'Title', 'Level', 'Embarked']
features_to_impute_min_max_scale = ['Age']
features_to_min_max_scale = ['Parch', 'Pclass', 'SibSp', 'CabinOccupancy', 'PeopleOnTicket']
features_to_transform = ['Fare']

# %%
# Then instantiate pipelines
# This pipeline is set up to one hot encode categorical variables:
pipeline_one_hot_encode = Pipeline(steps=[('one_hot_encoder', OneHotEncoder())])
# NOTE - had problems getting the Normalize stage to work on the 'Age' column as I had it appearing in two pipelines, eventually figured out I needed to create a speciial pipeline that combined both steps for it.
# This pipeline is set up to impute missing values and then normalize numerical features:
pipeline_impute_min_max_scale = Pipeline(steps=[('imputer', IterativeImputer(max_iter=10, random_state=0, add_indicator=False)), ('min_max_scaler', MinMaxScaler())])
# ('normalizer', StandardScaler())
# This pipeline is set up to scale numerical features:
pipeline_min_max_scale = Pipeline(steps=[('min_max_scaler', MinMaxScaler())])
# This piplein is set up to perform a non-linear transformation on numerical features.
# Note : the "Box Cox" method will only work with strinctly positive values:
pipeline_transform = Pipeline(steps=[('transformer', PowerTransformer())])

# %%
# Then put the features list and the transformers together using the column transformer
column_transformer = ColumnTransformer(transformers=[ \
    ('column_transform_one_hot_encode', pipeline_one_hot_encode, features_to_one_hot_encode), \
        ('column_transform_impute_min_max_scale', pipeline_impute_min_max_scale, features_to_impute_min_max_scale), \
            ('column_transform_min_max_scale', pipeline_min_max_scale, features_to_min_max_scale), \
                ('column_transform_transform', pipeline_transform, features_to_transform) 
], remainder='passthrough', verbose=True, sparse_threshold=0)
# %%
# Now apply the end to end pipeline to the data:
transformed_data = column_transformer.fit_transform(df)

# %%
def get_transformer_feature_names(column_transformer, original_dataframe, transformed_dataframe):

    output_features = []

    original_column_names = original_dataframe.columns

    for name, pipe, features in column_transformer.transformers_:
        if name!='remainder':
            for i in pipe:
                trans_features = []
                if hasattr(i,'categories_'):
                    trans_features.extend(i.get_feature_names(features))
                else:
                    trans_features = features
            output_features.extend(trans_features)
        else:
            for j in features:
                output_features.append(original_column_names[j])
    
    transformed_dataframe = pd.DataFrame(transformed_dataframe, columns=output_features)
    return transformed_dataframe

# %%
transformed_data = get_transformer_feature_names(column_transformer, df, transformed_data)

# %%
transformed_data.head(20)

# %%
list(df_transformed)

# %%
dictionary_of_column_types = { \
    'Sex_female': int,
    'Sex_male': int,
    'Title_Capt': int,
    'Title_Col': int,
    'Title_Countess': int,
    'Title_Don': int,
    'Title_Dona': int,
    'Title_Dr': int,
    'Title_Jonkheer': int,
    'Title_Lady': int,
    'Title_Major': int,
    'Title_Master': int,
    'Title_Miss': int,
    'Title_Mlle': int,
    'Title_Mme': int,
    'Title_Mr': int,
    'Title_Mrs': int,
    'Title_Ms': int,
    'Title_Rev': int,
    'Title_Sir': int,
    'Level_A': int,
    'Level_B': int,
    'Level_C': int,
    'Level_D': int,
    'Level_E': int,
    'Level_F': int,
    'Level_G': int,
    'Level_None': int,
    'Level_T': int,
    'Embarked_': int,
    'Embarked_C': int,
    'Embarked_Q': int,
    'Embarked_S': int,
    'Age': float,
    'Parch': float,
    'Pclass': float,
    'SibSp': float,
    'CabinOccupancy': float,
    'PeopleOnTicket': float,
    'Fare': float,
    'PassengerId': int,
    'Survived': int
}
#df_transformed = df_transformed.astype(dictionary_of_column_types)
df_transformed.info()

# %%
df_transformed

# %% [markdown]
#***
### Create Training Data Set
#
# %%
list_of_columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
X = df_transformed.loc[df_transformed['Survived'] >= 0 ]
X = X.drop(list_of_columns_to_drop, axis=1)

# %%
X.info()

# %% [markdown]
#***
### Extract Target Label From Training Data Set
#
# %%
y = X[['Survived']]
X = X.drop(['Survived'], axis=1)

# %%
y.info()

# %%
y=y.astype('int')

# %% [markdown]
#***
### Create Test data set - finally generate a Pandas dataframe from the dataflow.
# %%
X_test = df_transformed.loc[pd.isnull(df['Survived'])]
df_passenger_IDs = pd.DataFrame(X_test['PassengerId']).reset_index(drop=True)
df_passenger_IDs

# %%
X_test = X_test.drop(list_of_columns_to_drop, axis=1)
X_test = X_test.drop(['Survived'], axis=1)
X_test.info()

# %% [markdown]
#***
## Part 6 - Run The Models
#
#Now the fun begins!  Can run the model against the data. 
#
# %% [markdown]
#***
## Random Forreset Classifier

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
model_rfc = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# %%
model_rfc.fit(X, y)

# %%
predictions = model_rfc.predict(X_test)
predictions

# %%
def save_predictions_to_csv(df_passenger_IDs, predictions, model_name):
    df_predictions = pd.DataFrame(predictions, columns=['Survived'])
    df_submission = df_passenger_IDs.join(df_predictions)
    datetime_string = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    df_submission.to_csv('./kaggle/output/{}_submission_{}.csv'.format(model_name, datetime_string), index=False)

# %%
save_predictions_to_csv(df_passenger_IDs, predictions, 'random_forrest_classifier')

# %% [markdown]
#***
## K Nearest Neighbours
#
# %%
from sklearn.neighbors import KNeighborsClassifier

# %%
model_knn = KNeighborsClassifier(n_neighbors=3)

# %%
model_knn.fit(X, y)

# %%
predictions = model_knn.predict(X_test)
predictions

# %%
save_predictions_to_csv(df_passenger_IDs, predictions, 'k_nearest_neighbours')


# %% [markdown]
#***
## XG Boost
#
# %%
import xgboost as xgb

# %%
model_gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)

# %%
model_gbm.fit(X, y)

# %%
predictions = model_gbm.predict(X_test)
predictions

# %%
save_predictions_to_csv(df_passenger_IDs, predictions, 'gradient_boosted_decision_tree')

# %%
