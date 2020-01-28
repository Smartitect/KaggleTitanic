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
#The major non-standard module used is the 'azurelm.dataprep' SDK.  To install the Microsoft Azure ML Dataprep package:
#
#`pip install azureml-dataprep`
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
from sklearn.ensemble import RandomForestClassifier

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
df_test = pd.read_csv('./kaggle/input/test.csv')
df_test.shape

# %%
dataflow = dprep.auto_read_file('./kaggle/input/train.csv')

# %% [markdown]
# ***
## Part 3 - Feature Engineering I
# 
#Feature engineering necessary to support exploratory data analysis (EDA) below.
#
#In this section a range of feature extraction actions are performed using Microsoft's ML Dataprep SDK for Python:
#1. Pulling "Title" from the full name of each passenger;
#3. Counting cabin occupancy and passengers who also shared the same ticket;
#4. Extract - pulling the "Level" from the cabin (if indeed the passenger had a cabin);
#
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
dataflow = dataflow.fill_nulls('Cabin', 'None')

# %%
builder_level = dataflow.builders.derive_column_by_example(source_columns = ['Cabin'], new_column_name = 'Level')
builder_level.add_example(source_data = {'Cabin': 'C85'}, example_value = 'C')
builder_level.add_example(source_data = {'Cabin': 'E46'}, example_value = 'E')
builder_level.add_example(source_data = {'Cabin': 'None'}, example_value = 'None')
# builder_level.add_example(source_data = {'Cabin': None}, example_value = 'None')
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
#Training data set - finally generate a Pandas dataframe from the dataflow.
# %%
df_train_1 = dataflow.to_pandas_dataframe()
# %%
df_train_1['CabinOccupancy'].value_counts()
# %%
# Fix issue with cabin oocupancy count:
df_train_1['CabinOccupancy'].loc[df_train_1['CabinOccupancy'] == 687] = 0
# %%
df_train_1['CabinOccupancy'].value_counts()
# %%
df_train_1['Level'].value_counts()
# %%
# Investigate level T as this does not exist in Titantic documentaiton:
df_train_1.loc[df_train_1['Level']=='T']

# %%
dataflow_test = dataflow.replace_datasource(dprep.LocalDataSource('./kaggle/input/test.csv'))
# %%
profile = dataflow_test.get_profile()
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
#Test data set - finally generate a Pandas dataframe from the dataflow.
# %%
df_test_1 = dataflow_test.to_pandas_dataframe()
# %%
df_test_1['CabinOccupancy'].value_counts()
# %%
# Fix issue with cabin oocupancy count:
df_test_1['CabinOccupancy'].loc[df_test_1['CabinOccupancy'] == 327] = 0
# %%
df_test_1['CabinOccupancy'].value_counts()


# %% [markdown]
#***
## Part 4 - Inital Exploratory Data Analysis (EDA)
#
#Now going to work through each column in the data set to get an insight into it.

# %%
# Generate initial view of data using historgram plot.
df_train_1.hist(bins=20, figsize=(12,12))

# %%
# Compute the correlation matrix
correlation_matrix = df_train_1.corr()
correlation_matrix['Survived'].sort_values(ascending=False)

# %%
correlation_matrix['Age'].sort_values(ascending=False)

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
generate_box_swarm_plot(df_train_1, 'Sex', 'Age', 'Survived')

# %%
generate_box_swarm_plot(df_train_1, 'Pclass', 'Age', 'Survived')

# %%
generate_box_swarm_plot(df_train_1, 'Title', 'Age', 'Survived')

# %%
generate_box_swarm_plot(df_train_1, 'CabinOccupancy', 'Age', 'Survived')

# %%
generate_box_swarm_plot(df_train_1, 'Level', 'Fare', 'Survived')

# %%
generate_box_swarm_plot(df_train_1, 'PeopleOnTicket', 'Age', 'Survived')

# %% [markdown]
#***
## Part 5 - Feature Engineering II
#
#Here I create a Sci-Kit data pipeline to apply any steps required to get into shape for the models.
#
#2. Feature Engineering - 
#This will include:
#- Address missing values in "Level" (where people have no "Cabin" assigned);
#- Imputing missing ages a passenger using regressor to match based on values in other columns;
#- Scaling numerical values;
#- One-hot Encoding categorical values;
#
# At each stage we will apply feature engineering to both the train and test data sets.

# %% [markdown]
#***
### Address Missing Values For "Level"
#
df_train_1[['Level']] = df_train_1[['Level']].fillna(value='No Cabin')

# %%
df_train_1.head(10)

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
#features_to_normalize = ['Age']
features_to_min_max_scale = ['Parch', 'Pclass', 'SibSp', 'CabinOccupancy', 'PeopleOnTicket']
features_to_transform = ['Fare']

# %%
# Then instantiate pipelines
# This pipeline is set up to one hot encode categorical variables:
pipeline_one_hot_encode = Pipeline(steps=[('one_hot_encoder', OneHotEncoder())])

# This pipeline is set up to scale numerical features:
#pipeline_normalize = Pipeline(steps=[('normalizer', StandardScaler())])

# This pipeline is set up to scale numerical features:
pipeline_min_max_scale = Pipeline(steps=[('min_max_scaler', MinMaxScaler())])

# This piplein is set up to perform a non-linear transformation on numerical features.
# Note : the "Box Cox" method will only work with strinctly positive values:
pipeline_transform = Pipeline(steps=[('transformer', PowerTransformer())])

# %%
# Then put the features list and the transformers together using the column transformer
column_transformer = ColumnTransformer(transformers=[ \
    ('column_transform_one_hot_encode', pipeline_one_hot_encode, features_to_one_hot_encode), \
        #('column_transform_normalize', pipeline_normalize, features_to_normalize), \
            ('column_transform_min_max_scale', pipeline_min_max_scale, features_to_min_max_scale), \
                ('column_transform_transform', pipeline_transform, features_to_transform) 
], remainder='passthrough', verbose=True, sparse_threshold=0)

# %%
# Now apply the end to end pipeline to the data:
numpy_array = column_transformer.fit_transform(df_train_1)

# %%
def get_transformer_feature_names(columnTransformer, originalColumnNames):

    output_features = []

    for name, pipe, features in columnTransformer.transformers_:
        if name!='remainder':
            for i in pipe:
                trans_features = []
                if hasattr(i,'categories_'):
                    trans_features.extend(i.get_feature_names(features))
                else:
                    trans_features = features
            output_features.extend(trans_features)
        else:
            print(features)
            print(originalColumnNames)
            for j in features:
                print(originalColumnNames[j])
                output_features.append(originalColumnNames[j])
    return output_features

feature_names = get_transformer_feature_names(column_transformer, list(df_train_1))

# %%
column_transformer.transformers_[3][2]

# %% 
feature_names

# %%
list(df_train_1)

# %%
df_train_2 = pd.DataFrame(numpy_array, columns=feature_names)

# %%
df_train_2.head(10)

# %%
df_train_1.info()

# %%
list(df_train_1)

# %%
df_train_2.describe()

# %% [markdown]
#***
### Imput Missing Values For "Age"

# %%
imputer = IterativeImputer(max_iter=10, random_state=0, add_indicator=True)
output = imputer.fit_transform(df_train_2)

# %%
df_output = pd.DataFrame(output)

# %%
df_output

# %%
df_output.loc[df_output[40] == 1]

# %%
imputer

# %%
imputer.get_params()






















# %%
numpy_array

#%%
df_train_1[['Fare']].hist(bins=50, figsize=(12,12))

#%%
df_train_1[['Fare']].describe()

#%%
df_train_1['Fare'].value_counts()

# %%
df_train_1['Fare'].sort_values(ascending=True)

#%%
df_train_2[['Fare']].hist(bins=50, figsize=(12,12))

# %%
df_train_2.head(20)

# %%
df_train_2.describe()

# %%
df_train_2

# %%
column_transformer.get_feature_names()

# %% [markdown]
#***
## Part 5 - Exploratory Data Analysis (EDA)
#
#Now going to work through each column in the data set to get an insight into it.

# %%
y = df["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")