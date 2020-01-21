# %% [markdown]
# # Introduction...
#
# In this notebook I'm attempting my first Kaggle competition.
#
# Key things I wanted to achieve:
# 1. Use of Microsoft's DataPrep SDK for Python rather than more traditional data wrangling methods - a much more scalable framework;
# 2. Application of a simple linear regression model initially;
# 3. Moving onto more sophisticated models later on.

# ## Pre-requisites
#
# To install the Microsoft Azure ML Dataprep package:
#
# pip install azureml-dataprep

# %%
import azureml.dataprep as dprep
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# %% [markdown]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#%%
for dirname, _, filenames in os.walk('./kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# %%
# =============================================================================
# %% [markdown]
# # Part 1 - Data Ingestion

# %%
# Load the train and test files as Pandas dataframes...
df_train = pd.read_csv('./kaggle/input/train.csv')
df_train.shape

# %%
df_test = pd.read_csv('./kaggle/input/test.csv')
df_test.shape

# %%
dataflow = dprep.auto_read_file('./kaggle/input/train.csv')
dataflow.get_profile()

# %% [markdown]
# %%
# =============================================================================
# # Part 2 - Feature Engineering

# %%
# =============================================================================
# %% [markdown]
# ## Creation of "Title" feature.

# %%
# First we neeed to build some examples to provide as examples to train dataprep
def create_examples(df, builder, column, search_example_pairs):
    example_list = []
    for key, value in search_example_pairs.items():
        example = df_train[df_train[column].str.contains(key)].iloc[0][column]
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

# %%
# =============================================================================
# %% [markdown]
# ## Creation of "Cabin Occupancy" Feature

# %%
dataflow_filtered = dataflow.filter((dataflow_train_add_title['Cabin'] != ''))
dataflow_filtered.get_profile()

#%%
# Add in a CabinOccupancy using a count.
dataflow_filtered = dataflow_filtered.summarize(
        summary_columns=[
            dprep.SummaryColumnsValue(
                column_id='PassengerId',
                summary_column_name='CabinOccupancy', 
                summary_function=dprep.SummaryFunction.COUNT)],
        group_by_columns=['Cabin'],
        join_back=True)
dataflow_filtered = dataflow_filtered.keep_columns(['PassengerId', 'CabinOccupancy'])

# %%
# Use a left outer join to add the new CabinOccupancy column back into the main data set.
dataflow = dprep.Dataflow.join(left_dataflow=dataflow
                                      right_dataflow=dataflow_filtered,
                                      join_key_pairs=[('PassengerId', 'PassengerId')],
                                      left_column_prefix=None,
                                      right_column_prefix='r_',
                                      join_type=dprep.JoinType.LEFTOUTER)

# %%
# Tidy up by filling nulls, dropping unwanted column and renaming new column.
dataflow = dataflow.fill_nulls('r_CabinOccupancy', 0)
dataflow = dataflow.drop_columns('r_PassengerId')
dataflow = dataflow.rename_columns({'r_CabinOccupancy': 'CabinOccupancy'})


# %%
# Use the following to view the result of the cell above based on a specific cabin.
#df_examples = dataflow_train_new_features.to_pandas_dataframe()
#df_examples[df_examples['Cabin'].str.contains("G6")]

# %%
# =============================================================================
# %% [markdown]
# ## Creation of "Level" Feature

# %%
df_level_examples = dataflow.to_pandas_dataframe()
df_level_examples.head(20)

# %%
df_level_examples.tail(20)

# %%
builder_level = dataflow.builders.derive_column_by_example(source_columns = ['Cabin'], new_column_name = 'Level')
# builder_level.add_example(source_data = df.iloc[0], example_value = 'None')
builder_level.add_example(source_data = df_level_examples.iloc[0], example_value = 'C')
builder_level.add_example(source_data = df_level_examples.iloc[2], example_value = 'E')
builder_level.add_example(source_data = df_level_examples.iloc[879], example_value = None)
builder_level.preview() # Preview top 10 rows.

# %%
dataflow = builder_level.to_dataflow()

# %%
dataflow = dataflow.fill_nulls('Level', 'None')

# %%
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
dataflow.get_profile()

# %%
# =============================================================================
# %% [markdown]
# ## Impute Missing Values "Age" Feature

# impute with MEAN
impute_mean = dprep.ImputeColumnArguments(column_id='Age',
                                          impute_function=dprep.ReplaceValueFunction.MEAN)
# get instance of ImputeMissingValuesBuilder
builder_impute_age = dataflow.builders.impute_missing_values(impute_columns=[impute_mean],
                                                   group_by_columns=['Title'])

# call learn() to learn a fixed program to impute missing values
builder_impute_age.learn()

# call to_dataflow() to get a dataflow with impute step added
dataflow = builder_impute_age.to_dataflow()


# %% [markdown]
# # Part 3 - Exploratory Data Analysis (EDA)
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

# %%
from sklearn.ensemble import RandomForestClassifier

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