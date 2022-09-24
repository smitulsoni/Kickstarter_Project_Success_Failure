# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 17:18:42 2021

@author: sonis
"""

# Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV



import warnings
warnings.filterwarnings("ignore")

# Import dataset
df = pd.read_excel("C:/Users/sonis/Desktop/Courses/Fall/INSY662 - Data Mining & Visualization/Indi Project/Kickstarter.xlsx")
df.shape

#######################################################################################################
################################## TRAINING THE CLASSIFICATION MODEL ##################################
#######################################################################################################

def data_preprocessing(kickstarter_df):
    # Since we only need to predict the states as "Successful"/"Failed", filtering the data
    kickstarter_df = kickstarter_df.loc[(kickstarter_df['state']=="successful") | (kickstarter_df['state']=="failed")]
    kickstarter_df.shape
    
    # Removing labelled columns irrelevant to any analaysis
    kickstarter_df = kickstarter_df.drop(columns = ['project_id','name'])
    
    # Removing dates as they are already divided further in other columns
    kickstarter_df = kickstarter_df.drop(columns = ['deadline','state_changed_at', 'created_at', 'launched_at'])
    
    # Removing columns related to state change as the state will be changed after the launch which is not related for our use case
    kickstarter_df = kickstarter_df.drop(columns = ['state_changed_at_weekday','state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr', 'state_changed_at_hr'])
    
    # Removing currency, static_usd_rate as it depends on the country. So only taking country suffices
    kickstarter_df = kickstarter_df.drop(columns = ['currency'])
    # Removing usd_pledged, pledged as it comes after launching on kickstarter
    kickstarter_df = kickstarter_df.drop(columns = ['pledged', 'usd_pledged'])
    
    # Removing spotlight, backers_count as the project cannot feature on kickstarter on the launch day
    kickstarter_df = kickstarter_df.drop(columns = ['spotlight', 'backers_count'])
    
    # Removing name_len & blurb_len since the clean columns for them are included
    kickstarter_df = kickstarter_df.drop(columns = ['name_len','blurb_len'])
    
    # Removing disable_communication as all the values are FALSE
    kickstarter_df = kickstarter_df.drop(columns = ['disable_communication'])
    
    # Checking null values
    kickstarter_df.isnull().sum()
    
    # Removing column launch_to_state_change_days as maximum values are null
    kickstarter_df = kickstarter_df.drop(columns = 'launch_to_state_change_days')
    
    # Removing rest of the NA values
    kickstarter_df = kickstarter_df.dropna()
    kickstarter_df.shape
    
    # Converting state to 0/1
    kickstarter_df["state"] = kickstarter_df["state"].map({'successful': 1, 'failed': 0}).astype(int)
    
    # Rearranging columns
    first_column = kickstarter_df.pop('state')
    kickstarter_df.insert(0, 'state', first_column)
    
    # Dummifying the categorical variables
    dummy_category=pd.get_dummies(kickstarter_df.category, prefix="category")
    dummy_country=pd.get_dummies(kickstarter_df.country, prefix="country") 
    dummy_deadline_weekday=pd.get_dummies(kickstarter_df.deadline_weekday, prefix="deadline_weekday") 
    dummy_created_at_weekday=pd.get_dummies(kickstarter_df.created_at_weekday, prefix="created_at_weekday") 
    dummy_launched_at_weekday=pd.get_dummies(kickstarter_df.launched_at_weekday, prefix="launched_at_weekday") 
    kickstarter_df = kickstarter_df.join(dummy_category) 
    kickstarter_df = kickstarter_df.join(dummy_country) 
    kickstarter_df = kickstarter_df.join(dummy_deadline_weekday) 
    kickstarter_df = kickstarter_df.join(dummy_created_at_weekday)
    kickstarter_df = kickstarter_df.join(dummy_launched_at_weekday)
    
    # Removing columns that have been dummified
    kickstarter_df = kickstarter_df.drop(columns = ['category', 'country', 'deadline_weekday', 'created_at_weekday', 'launched_at_weekday'])
    kickstarter_df.shape
    
    return kickstarter_df

kickstarter_df = data_preprocessing(df)

# Setting up function for LOGISTIC REGRESSION
def logistic_regression(X):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 5)
    lr = LogisticRegression()
    model = lr.fit(X_train,y_train)
    y_test_pred = model.predict(X_test)
    print("Logistic Regression Results:")
    print("Accuracy:",metrics.accuracy_score(y_test, y_test_pred))
    # Print the confusion matrix
    #metrics.confusion_matrix(y_test, y_test_pred)
    # Calculate the Precision/Recall
    print("Precision:",metrics.precision_score(y_test, y_test_pred))
    print("Recall:",metrics.recall_score(y_test, y_test_pred))
    # Calculate the F1 score
    print("F1:",metrics.f1_score(y_test, y_test_pred))
    # Confusion matrix with label
    print(pd.DataFrame(metrics.confusion_matrix(y_test, y_test_pred, labels=[0,1]), index=['true:0','true:1'], columns=['pred:0', 'pred:1']))
    
# Setting up function for KNN    
def knn(X):
    standardizer = StandardScaler()
    X_std = standardizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.33, random_state = 5)
    
    accuracy=[]
    for i in range (1,40):
        knn = KNeighborsClassifier(n_neighbors=i)
        model = knn.fit(X_train,y_train)
        y_test_pred = model.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test, y_test_pred))
        
    print("\nKNN Results:")
    print("Accuracy:",max(accuracy))
    print("Precision:",metrics.precision_score(y_test, y_test_pred))
    print("Recall:",metrics.recall_score(y_test, y_test_pred))
    print("F1:",metrics.f1_score(y_test, y_test_pred))
    
# Setting up function for decision tree
def decision_tree(X):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=5)   
    accuracy=[]
    for i in range (2,21):
        decisiontree = DecisionTreeClassifier(max_depth=i)
        model = decisiontree.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test, y_test_pred))        

    print("\nDecision Tree Results:")
    print("Accuracy:", max(accuracy))
    print("Precision:",metrics.precision_score(y_test, y_test_pred))
    print("Recall:",metrics.recall_score(y_test, y_test_pred))
    print("F1:",metrics.f1_score(y_test, y_test_pred))
    
# Setting up function for Random Forest
def random_forest(X, total_predictors):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=5)
    accuracy=[]
    for i in range (2,total_predictors):
        randomforest = RandomForestClassifier(random_state=5,max_features=i,n_estimators=100)
        model = randomforest.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test, y_test_pred))        

    print("\nRandom Forest Results:")
    print("Accuracy:", max(accuracy))
    print("Precision:",metrics.precision_score(y_test, y_test_pred))
    print("Recall:",metrics.recall_score(y_test, y_test_pred))
    print("F1:",metrics.f1_score(y_test, y_test_pred))
    
# Setting up function for Gradient Boosting
def gradient_boosting(X):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=5)
    accuracy=[]
    for i in range (2,20):
        gbt = GradientBoostingClassifier(random_state=5,min_samples_split=i,n_estimators=100)
        model = gbt.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test, y_test_pred))        
        #print(metrics.accuracy_score(y_test, y_test_pred))
    print("\nGradient Boosting Results:")
    print("Accuracy:", max(accuracy))
    print("Precision:",metrics.precision_score(y_test, y_test_pred))
    print("Recall:",metrics.recall_score(y_test, y_test_pred))
    print("F1:",metrics.f1_score(y_test, y_test_pred))


X = kickstarter_df.iloc[:,1:]
y = kickstarter_df["state"]

total_predictors = len(X.columns)

# Running all 5 models before feature selection
logistic_regression_accuracy = logistic_regression(X)
knn_accuracy = knn(X)
decision_tree_accuracy = decision_tree(X)
random_forest_accuracy = random_forest(X, total_predictors)
gradient_boosting_accuracy = gradient_boosting(X)

# Feature Selection
# RFE
accuracy = []
for i in range(1,len(kickstarter_df.columns)-1):
    lr = LogisticRegression(max_iter=500)
    rfe = RFE(lr, n_features_to_select=i)
    model = rfe.fit(X, y)
    model.ranking_
    rfe_ranking=pd.DataFrame(list(zip(X.columns,model.ranking_)), columns = ['predictor','ranking'])
    
    rfe_predictors = rfe_ranking[rfe_ranking["ranking"]==1]

    # Running models after selecting predictors with RFE
    X_rfe = kickstarter_df[rfe_predictors["predictor"]]
    y = kickstarter_df["state"]

    total_predictors = len(X_rfe.columns)

    gradient_boosting_accuracy = gradient_boosting(X_rfe)
    accuracy.append(gradient_boosting_accuracy)

# LASSO
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
model = Lasso(alpha=0.01)
model.fit(X_std,y)
model.coef_
lasso_coef = pd.DataFrame(list(zip(X.columns,model.coef_)), columns = ['predictor','coefficient'])

lasso_predictors = lasso_coef[abs(lasso_coef["coefficient"])>0.01]["predictor"]
type(lasso_predictors)
lasso_predictors = lasso_predictors.to_frame()

# Random Forest
from sklearn.feature_selection import SelectFromModel
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=5)

# Selecting best features from random forest
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(X_train, y_train)
rf_selected_feat= X_train.columns[(sel.get_support())]
len(rf_selected_feat)    

# Running models after selecting predictors with RFE
X_rfe = kickstarter_df[rfe_predictors["predictor"]]
y = kickstarter_df["state"]

total_predictors = len(X_rfe.columns)

logistic_regression_accuracy = logistic_regression(X_rfe)
knn_accuracy = knn(X_rfe)
decision_tree_accuracy = decision_tree(X_rfe)
random_forest_accuracy = random_forest(X_rfe, total_predictors)
gradient_boosting_accuracy = gradient_boosting(X_rfe)

# Running models after selecting predictors with LASSO
X_lasso = kickstarter_df[lasso_predictors["predictor"]]
y = kickstarter_df["state"]

total_predictors = len(X_lasso.columns)

logistic_regression_accuracy = logistic_regression(X_lasso)
knn_accuracy = knn(X_lasso)
decision_tree_accuracy = decision_tree(X_lasso)
random_forest_accuracy = random_forest(X_lasso, total_predictors)
gradient_boosting_accuracy = gradient_boosting(X_lasso)

# Running models after selecting predictors with RANDOM FOREST
X_rf = kickstarter_df[rf_selected_feat]
y = kickstarter_df["state"]

total_predictors = len(X_rf.columns)

logistic_regression_accuracy = logistic_regression(X_rf)
knn_accuracy = knn(X_rf)
decision_tree_accuracy = decision_tree(X_rf)
random_forest_accuracy = random_forest(X_rf, total_predictors)
gradient_boosting_accuracy = gradient_boosting(X_rf)

# Hyper paramater tuning
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=5)
gbt = GradientBoostingClassifier(random_state=0)

parameters = {'max_depth':[i for i in range(1,20)],'n_estimators':[50, 100],'learning_rate':[0.01, 0.1, 1],'max_features': ['log2', 'sqrt']}

grid = GridSearchCV(gbt, parameters,verbose=3,scoring='accuracy')

# Training the model
grid.fit(X_train,y_train)

# Providing the best params
print(grid.best_params_)

# Predictions
y_pred = grid.predict(X_test)

# Calculating the accuracy score
print("Accuracy score with decision trees: ", metrics.accuracy_score(y_test, y_pred))
# 0.7857599658921339


#######################################################################################################
################################## GRADING MODEL TO RUN ###############################################
#######################################################################################################

import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier

# Import dataset

# Change paths for the below 2 datasets to grade the model

# Kickstarter full data provided for model training
df_train = pd.read_excel("C:/Users/sonis/Desktop/Courses/Fall/INSY662 - Data Mining & Visualization/Indi Project/Kickstarter-Grading-Sample.xlsx")
df_test = pd.read_excel("C:/Users/sonis/Desktop/Courses/Fall/INSY662 - Data Mining & Visualization/Indi Project/Kickstarter.xlsx")

# Defining data preprocessing function for cleaning the data
def data_preprocessing(kickstarter_df):
    # Since we only need to predict the states as "Successful"/"Failed", filtering the data
    kickstarter_df = kickstarter_df.loc[(kickstarter_df['state']=="successful") | (kickstarter_df['state']=="failed")]
    kickstarter_df.shape
    
    # Removing labelled columns irrelevant to any analaysis
    kickstarter_df = kickstarter_df.drop(columns = ['project_id','name'])
    
    # Removing dates as they are already divided further in other columns
    kickstarter_df = kickstarter_df.drop(columns = ['deadline','state_changed_at', 'created_at', 'launched_at'])
    
    # Removing columns related to state change as the state will be changed after the launch which is not related for our use case
    kickstarter_df = kickstarter_df.drop(columns = ['state_changed_at_weekday','state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr', 'state_changed_at_hr'])
    
    # Removing currency, static_usd_rate as it depends on the country. So only taking country suffices
    kickstarter_df = kickstarter_df.drop(columns = ['currency'])
    # Removing usd_pledged, pledged as it comes after launching on kickstarter
    kickstarter_df = kickstarter_df.drop(columns = ['pledged', 'usd_pledged'])
    
    # Removing backers_count as the project cannot feature on kickstarter on the launch day
    kickstarter_df = kickstarter_df.drop(columns = ['backers_count'])
    
    # Removing name_len & blurb_len since the clean columns for them are included
    kickstarter_df = kickstarter_df.drop(columns = ['name_len','blurb_len'])
    
    # Removing disable_communication as all the values are FALSE
    kickstarter_df = kickstarter_df.drop(columns = ['disable_communication'])
    
    # Checking null values
    kickstarter_df.isnull().sum()
    
    # Removing column launch_to_state_change_days as maximum values are null
    kickstarter_df = kickstarter_df.drop(columns = 'launch_to_state_change_days')
    
    # Removing rest of the NA values
    kickstarter_df = kickstarter_df.dropna()
    kickstarter_df.shape
    
    # Converting state to 0/1
    kickstarter_df["state"] = kickstarter_df["state"].map({'successful': 1, 'failed': 0}).astype(int)
    
    # Rearranging columns
    first_column = kickstarter_df.pop('state')
    kickstarter_df.insert(0, 'state', first_column)
    
    # Dummifying the categorical variables
    dummy_category=pd.get_dummies(kickstarter_df.category, prefix="category")
    dummy_country=pd.get_dummies(kickstarter_df.country, prefix="country") 
    dummy_deadline_weekday=pd.get_dummies(kickstarter_df.deadline_weekday, prefix="deadline_weekday") 
    dummy_created_at_weekday=pd.get_dummies(kickstarter_df.created_at_weekday, prefix="created_at_weekday") 
    dummy_launched_at_weekday=pd.get_dummies(kickstarter_df.launched_at_weekday, prefix="launched_at_weekday") 
    kickstarter_df = kickstarter_df.join(dummy_category) 
    kickstarter_df = kickstarter_df.join(dummy_country) 
    kickstarter_df = kickstarter_df.join(dummy_deadline_weekday) 
    kickstarter_df = kickstarter_df.join(dummy_created_at_weekday)
    kickstarter_df = kickstarter_df.join(dummy_launched_at_weekday)
    
    # Removing columns that have been dummified
    kickstarter_df = kickstarter_df.drop(columns = ['category', 'country', 'deadline_weekday', 'created_at_weekday', 'launched_at_weekday'])
    kickstarter_df.shape
    
    # Correlation Matrix
    kickstarter_df.corr()
    
    # Removing spotlights as the there is high correlation
    kickstarter_df = kickstarter_df.drop(columns = ['backers_count'])
    
    return kickstarter_df

kickstarter_df = data_preprocessing(df_train)
kickstarter_test_df = data_preprocessing(df_test)


# FINAL CLASSIFICATION MODEL
X = kickstarter_df.iloc[:,1:]
y = kickstarter_df['state']

X_test = kickstarter_test_df.iloc[:,1:]
y_test = kickstarter_test_df['state']

gbt = GradientBoostingClassifier(random_state=5,min_samples_split=3,n_estimators=100)
model = gbt.fit(X, y)
y_test_pred = model.predict(X_test)
accuracy.append(metrics.accuracy_score(y_test, y_test_pred))        

print("\nGradient Boosting Results:")
print("Accuracy:", metrics.accuracy_score(y_test, y_test_pred))


#######################################################################################################


#######################################################################################################
############################################ CLUSTERING ###############################################
#######################################################################################################

from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering, KMeans
import matplotlib.pyplot as plt
from matplotlib import pyplot
import plotly.express as px
from pandas.plotting import *
import plotly.io as pio

# Elbow method to show optimal value of k
df = kickstarter_df[['create_to_launch_days', 'staff_pick', 'goal', 'blurb_len_clean', 'launched_at_yr', 'created_at_yr', 'deadline_yr']]

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df)
    Sum_of_squared_distances.append(km.inertia_)

labels =km.labels_
silhouette = silhouette_samples(df, labels)
print("silhoutte score",silhouette_score(df, labels))

plt.plot(K, Sum_of_squared_distances, 'bx-')

# Only numerical columns into consideration
scaler=StandardScaler()
standardized_x=scaler.fit_transform(df)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
df = pca.fit_transform(standardized_x)
reduced_df = pd.DataFrame(df, columns=['PC1','PC2'])
plt.scatter(reduced_df['PC1'], reduced_df['PC2'], alpha=.1, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

kmeans=KMeans(n_clusters=4)
model=kmeans.fit(reduced_df)
labels=model.predict(reduced_df)
reduced_df['cluster'] = labels
list_labels=labels.tolist()
count1=count2=count3=count4=0
for i in list_labels:
    if i==0:
        count1+=1
    elif i==1:
        count2+=1
    elif i==2:
        count3+=1
    elif i==3:
        count4+=1
u_labels=np.unique(labels)
print("Number of datapoints in cluster 1 (K Means):", count1)
print("Number of datapoints in cluster 2 (K Means):", count2)
print("Number of datapoints in cluster 3 (K Means):", count3)
print("Number of datapoints in cluster 4 (K Means):", count4)
for i in u_labels:
    plt.scatter(df[labels == i , 0] , df[labels == i , 1] )
plt.legend(u_labels)
plt.show()

# Dataframe from the centroids
# Clustering

df = kickstarter_df[['create_to_launch_days', 'staff_pick', 'goal', 'blurb_len_clean', 'launched_at_yr', 'created_at_yr', 'deadline_yr']]

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
standardized_x=scaler.fit_transform(df)
standardized_x=pd.DataFrame(standardized_x,columns=df.columns)
df=standardized_x
kmeans=KMeans(n_clusters=4)
model=kmeans.fit(df)
labels=model.predict(df)
df['cluster'] = labels
list_labels=labels.tolist()
count1=count2=count3=count4=0
for i in list_labels:
    if i==0:
        count1+=1
    elif i==1:
        count2+=1
    elif i==2:
        count3+=1
    elif i==3:
        count4+=1
u_labels=np.unique(labels)
print("Number of datapoints in cluster 1 (K Means):", count1)
print("Number of datapoints in cluster 2 (K Means):", count2)
print("Number of datapoints in cluster 3 (K Means):", count3)
print("Number of datapoints in cluster 4 (K Means):", count4)

pio.renderers.default = 'browser'
centroids = pd.DataFrame(kmeans.cluster_centers_)
fig = px.parallel_coordinates(centroids,labels=df.columns,color=u_labels)
fig.show()

