# %%
# =============================================================================
# %% [markdown]
# # Part 1 - Introduction - Look at the Big Picture
#
# In this notebook I'm attempting my first Kaggle competition.
#
# ## 1.1 - Frame The Problem
#
# This problem can be classified as follows:
# - A **classfication** problem : we are being asked to predict whether an individual dies or not based on a range of continuous and categorical features.  Indeed to be more specific this is a **univariate classifcaiton** problem : ie we are being asked to predcict a single label for each individual;
# - **Supervised** learning : we are given a training data set with labeled examples;
# - **Batch** Learning : we will run the models as a one off as the data is static and will never change;
#
# Wider non-functional requirements indicate:
# - Low data volume - 
# - Non-sensitive data - 
# - No legal or ethical concerns - 
#
# The types of model that are likely to support this problem are therefore:
# - X
# - Y
# - Z
# 
# Key things I wanted to achieve:
# 1. Use of Microsoft's DataPrep SDK for Python rather than more traditional data wrangling methods - taking advantage of some of the advanced features such as "column by example";
# 2. Start with application of a simple linear regression model and then build more sophisticated models later on.
#
# ## 1.2 - Select a Performance Measure
#
# The performance measure is driven by Kaggle: that is the overall **accuracy** of the model - defined as the percentage of people who are labeled correctly.
# It would also be interesting to see more detailed metrics - ie the confusion matirix.
# Also want to use advanced techniques for hyperparameter tuning - more on this later.
# 
# ## 1.3 - List and Validate The Assumptions
# 
# - Assumption 1 - 
# 
# ## Pre-requisites
#
# The major non-standard module used is the 'azurelm.dataprep' SDK.  To install the Microsoft Azure ML Dataprep package:
#
# `pip install azureml-dataprep`
#
# Documentation is available at the following web sites:
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
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#%%
for dirname, _, filenames in os.walk('./kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# =============================================================================
# %% [markdown]
# # Part 2 - Data Ingestion

# %%
# Load the train and test files as Pandas dataframes...
df_train = pd.read_csv('./kaggle/input/train.csv')
df_train.shape

# %%
df_test = pd.read_csv('./kaggle/input/test.csv')
df_test.shape

# %%
dataflow_train = dprep.auto_read_file('./kaggle/input/train.csv')
dataflow_test = dprep.auto_read_file('./kaggle/input/test.csv')
dataflow = dataflow_train.append_rows([dataflow_test])
dataflow.get_profile()

# %% [markdown]
# =============================================================================
# # Part 3 - Feature Engineering I
# 
# Feature engineering necessary to support exploratory data analysis (EDA) below.
#
# In this section a range of actions are performed such as:
# 1. Feature Extraction - pulling "Title" from the full name of each passenger;
# 2. Feature Engineering - imputing missing ages a passenger based on the mean age for the group of passengers to which that passenger is aligned;
# 3. Feature Extraction - counting cabin occupancy and passengers who also shared the same ticket;
# 4. Feature Extract - pulling the "Level" from the cabin (if indeed the passenger had a cabin);
#
# =============================================================================
# %% [markdown]
# ## Creation of "Title" feature.

# %%
# First we neeed to build some examples to provide as examples to train dataprep
def create_examples(df, builder, column, search_example_pairs):
    example_list = []
    for key, value in search_example_pairs.items():
        example = df[df[column].str.contains(key)].iloc[0][column]
        builder.add_example(source_data = {column: example}, example_value = value)

# %%
df_train[df_train['Name'].str.contains('Jonkheer')].iloc[0]['Name']

#%%
df_train.iloc[38]

#%%
df_train.iloc[759]

# Use the following to find specific titles in the name column - this helps to select examples to train the "derive by column" approach.
#df_title_examples[df_title_examples['Name'].str.contains("Countess")]

# %%
builder_title = dataflow.builders.derive_column_by_example(source_columns = ['Name'], new_column_name = 'Title')
title_search_example_pairs = {
    'Mr': 'Mr',
    'Mrs': 'Mrs',
    'Miss': 'Miss',
    'Vander Planke, Miss': 'Miss', 
    'Rothes, the Countess. of': 'Countess',
    'Master': 'Master' }

create_examples(df_train, builder_title, 'Name', title_search_example_pairs)

dataflow = builder_title.to_dataflow()

# %%
profile = dataflow.get_profile()
profile.columns['Title'].value_counts

# =============================================================================
# %% [markdown]
# ## Creation of "Cabin Occupancy" Feature

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

# =============================================================================
# %% [markdown]
# ## Creation of "Level" Feature

# %%
dataflow = dataflow.fill_nulls('Cabin', 'None')

# %%
builder_level = dataflow.builders.derive_column_by_example(source_columns = ['Cabin'], new_column_name = 'Level')
# builder_level.add_example(source_data = df.iloc[0], example_value = 'None')
builder_level.add_example(source_data = {'Cabin': 'C85'}, example_value = 'C')
builder_level.add_example(source_data = {'Cabin': 'E46'}, example_value = 'E')
builder_level.add_example(source_data = {'Cabin': 'None'}, example_value = 'None')
builder_level.add_example(source_data = {'Cabin': None}, example_value = 'None')
builder_level.preview() # Preview top 10 rows.

# %%
dataflow = builder_level.to_dataflow()

# =============================================================================
# %% [markdown]
# ## Create "People On Ticket" Feature

dataflow = dataflow.summarize(
        summary_columns=[
            dprep.SummaryColumnsValue(
                column_id='PassengerId',
                summary_column_name='PeopleOnTicket', 
                summary_function=dprep.SummaryFunction.COUNT)],
        group_by_columns=['Ticket'],
        join_back=True)

# =============================================================================
# %% [markdown]
# ## Impute Missing Values "Age" Feature

# Configure the impute for the Age column using MEAN
impute_mean = dprep.ImputeColumnArguments(column_id='Age',
                                          impute_function=dprep.ReplaceValueFunction.MEAN)
# Apply the impute and define grouping
builder_impute_age = dataflow.builders.impute_missing_values(impute_columns=[impute_mean],
                                                   group_by_columns=['Title'])

# call learn() to learn a fixed program to impute missing values
builder_impute_age.learn()

# call to_dataflow() to get a dataflow with impute step added
dataflow = builder_impute_age.to_dataflow()

# %%
dataflow

# %%
dataflow.get_profile()

# %%
dataflow_test = dataflow.replace_datasource(dprep.LocalDataSource('./kaggle/input/test.csv'))

# %%
profile = dataflow_test.get_profile()
profile.columns['Title'].value_counts

# =============================================================================
# %% [markdown]
# # Part 4 - Exploratory Data Analysis (EDA)
#
# Now going to work through each column in the data set to get an insight into it.

# %%
dataflow.get_profile()

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
df = dataflow.to_pandas_dataframe()

# %%
generate_box_swarm_plot(df, 'Sex', 'Age', 'Survived')

# %%
generate_box_swarm_plot(df, 'Pclass', 'Age', 'Survived')

# %%
generate_box_swarm_plot(df, 'Title', 'Age', 'Survived')

# %%
generate_box_swarm_plot(df, 'SibSp', 'Age', 'Survived')

# %%
generate_box_swarm_plot(df, 'Parch', 'Age', 'Survived')

# %%
generate_box_swarm_plot(df, 'Embarked', 'Age', 'Survived')

# %%
generate_box_swarm_plot(df, 'Level', 'Age', 'Survived')

# %%
generate_box_swarm_plot(df, 'CabinOccupancy', 'Age', 'Survived')

# %%
generate_box_swarm_plot(df, 'Level', 'Fare', 'Survived')

# %%
generate_box_swarm_plot(df, 'PeopleOnTicket', 'Age', 'Survived')

# =============================================================================
# %% [markdown]
# # Part 5 - Feature Engineering II
#
# Here I apply any steps required to get into shape for the models.
#
# This will include:
# - Scaling
# - 
# - One-hot Encoding
#
# =============================================================================
# %% [markdown]
# # Part 5 - Exploratory Data Analysis (EDA)
#
# Now going to work through each column in the data set to get an insight into it.

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