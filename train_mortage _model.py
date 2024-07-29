from ibm_watson_studio_lib import access_project_or_space
wslib = access_project_or_space()
## Set APIKEY
ibmcloud_api_key = 'Xaf2rFwICAKkEzVdnTF6bIlKQahA7_2uUuEDL8XxPRpr' #Wells ...0074
# Install Libraries
import pandas as pd
pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

import seaborn as sns
sns.set(style='darkgrid',palette="deep")

import ibm_db, ibm_db_dbi as dbi


from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn import metrics

#!pip install ibm-cloud-sdk-core

PROJECT_UID= os.environ['PROJECT_ID'] #this assumes you are running in Studio. When running externally please add the project id here.
CPD_URL=os.environ['RUNTIME_ENV_APSX_URL'][len('https://api.'):] #the variable starts with https://api." and you only need the rest 'dataplatform.dev.cloud.ibm.com'
CONTAINER_ID=PROJECT_UID
CONTAINER_TYPE='project'
EXPERIMENT_NAME='mortgage_approval_prediction'

# CPD4D API key. Navigate to hamburger menu -> Acess (IAM) -> API Keys to create one for you (or have your account admin do it)
# The project token is an authorization token that is used to access project resources like data sources, connections, and used by platform APIs.
# PROJECT_ACCESS_TOKEN=project.project_context.accessToken.replace('Bearer ','') #You can create or lookup such a token from the settings tab of this project

DATABASE_CREDENTIALS = {
   'database': 'BLUDB',
   'auth_method': 'username_password',
   'password': 'DataFabric@2022IBM',
   'port': '50001',
   'host': 'db2w-ruggyab.us-south.db2w.cloud.ibm.com',
   'ssl': 'true',
   'username': 'cpdemo'}

# Connect to Database
db2_warehouse_datafabric_trial_dsn = 'DATABASE={};HOSTNAME={};PORT={};PROTOCOL=TCPIP;UID={uid};PWD={pwd};SECURITY=SSL'.format(DATABASE_CREDENTIALS['database'],DATABASE_CREDENTIALS['host'],DATABASE_CREDENTIALS['port'],uid=DATABASE_CREDENTIALS['username'],pwd=DATABASE_CREDENTIALS['password'])
#
db2_warehouse_datafabric_trial_connection = dbi.connect(db2_warehouse_datafabric_trial_dsn)

sql_query='select * from AI_MORTGAGE.MORTGAGE_APPROVAL_VIEW'
# Load Data into dataframe

df_mortgage = pd.read_sql_query(sql_query, con=db2_warehouse_datafabric_trial_connection)

df_mortgage.head()

# save df_mortgage for re-use
wslib.save_data("mortgage_data.csv", df_mortgage.to_csv(index=False).encode(),overwrite=True)


# Specify User Inputs:
# target_col : This is the target column indicating whether a mortgage application is approved or not.
# categorical_cols : A list of all categorical variables that need to be transformed before input into the model. You will use a transformer to impute missing values and
# create dummy variables.
# numerical_cols : A list of all of the numerical features that you input into the model. Later in the code, you use a transformer to impute any missing values in these
# columns. Flag like variables, with values of either 1 or 0 are included in this list.

numerical_cols=['INCOME','YRS_AT_CURRENT_ADDRESS','YRS_WITH_CURRENT_EMPLOYER','NUMBER_OF_CARDS','CREDITCARD_DEBT','LOAN_AMOUNT','CREDIT_SCORE','PROPERTY_VALUE','AREA_AVG_PRICE','LOANS']
target_col='MORTGAGE_APPROVAL'
categorical_cols=['STATE','GENDER','EDUCATION','EMPLOYMENT_STATUS','MARITAL_STATUS','APPLIEDONLINE','RESIDENCE','COMMERCIAL_CLIENT','COMM_FRAUD_INV']

cat_pct = 0.04
cor_pct = 0.90

df_prep=df_mortgage[numerical_cols+[target_col]+categorical_cols].copy()

# Handle categorical variables
# Categorical variables with a high number of unique values can significantly impact performance, because they may add too many one-hot encoded features to the dataset.
# Such categorical variables need to be removed. 
# Also remove the columns which have only one single category for all the rows.

for col in categorical_cols:
    if df_prep[col].nunique() > (cat_pct*df_prep.shape[0]):
        print(col,"removed")
        categorical_cols.remove(col)
        df_prep=df_prep.drop(col,axis=1)

for col in categorical_cols:
    if df_prep[col].nunique() ==1:
        print(col,"removed")
        categorical_cols.remove(col)
        df_prep=df_prep.drop(col,axis=1)
        
# Find correlations among numrerical variables
# Handle correlations
# Correlated data poses the threat of disproportionately reporting the effects of features with relatively similarly. This cell removes highly correlated columns, if any.

corr = df_prep[numerical_cols + [target_col]].corr()
corr = corr.round(3)
corr = corr.abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] >= cor_pct)]

df_prep = df_prep.drop(to_drop, axis=1)

if(len(to_drop)>0):
    numerical_cols.remove(to_drop[0])
    print(to_drop,"Removed because of high correlation")
    
# Build Data Pipelines
# Split Data and Build Transformer

X = df_prep.dropna(axis=0, subset=[target_col]).drop([target_col], axis=1)
y = df_prep.dropna(axis=0, subset=[target_col])[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)

print(len(X), len(y))

X = df_prep.dropna(axis=0, subset=[target_col]).drop([target_col], axis=1)
y = df_prep.dropna(axis=0, subset=[target_col])[target_col]

train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

# train is now 75% of the entire data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio)

# test is now 10% of the initial data set
# validation is now 15% of the initial data set
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

print("Size of Train", len(X_train), "Size of Validation", len(X_validation), "Size of Test", len(X_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
            
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
print("Size of Train", len(X_train), "Size of Validation", len(X_validation), "Size of Test", len(X_test))

# save X_Test for re-use
wslib.save_data("X_test.csv", X_test.to_csv(index=False).encode(),overwrite=True)

# save X_Validation for re-use
wslib.save_data("X_validation.csv", X_validation.to_csv(index=False).encode(),overwrite=True)

# For categorical variables,a 2 step pipeline is created. The SimpleImputer transformer will fill in missing values with 'Other', while the OneHotEncoder transformer will # create dummy variables for each category. The transformers are applied to the features specified in the categorical_cols variable.

# For numerical_colsvariables the pipeline has a single step in our example. You again leverage the SimpleImputer transformer to fill in missing values.
categorical_transformer = Pipeline(steps=[('impute_missing', SimpleImputer(strategy='constant', fill_value='Other')), ('dummy_vars', OneHotEncoder(handle_unknown='ignore'))])
# fill in missing data - you use median to keep the 1, 0  'flag' like variables either 1 or 0
numeric_transformer = Pipeline(steps=[('impute_missing', SimpleImputer(strategy='median'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
            ])
fitted_preprocessor = preprocessor.fit(X_train)

X_train_postprocess = fitted_preprocessor.transform(X_train)

X_test_postprocess = preprocessor.transform(X_test)

X_validation_postprocess = preprocessor.transform(X_validation)

onehot_columns=list(fitted_preprocessor.named_transformers_['cat'].named_steps['dummy_vars'].get_feature_names_out(input_features=categorical_cols))
numeric_features_list = list(numerical_cols)
numeric_features_list.extend(onehot_columns)

# Building Machine Learning Models
# Model Hyperparameter Tuning and Model Selection
# A Random Forest Model will be used for this tutorial, and that will require some hyperparameter tuning. The hyperparameter selection which performs best on the
# imbalanced validation data is selected as final. To determine this, you will be using ROC AUC as the evaluation metric.

l_ne = [10, 50]
l_md = [5,7, 10] 
l_mf = [0.75, 0.9]

highest_test_auc = 0.0
top_ne = 0
top_md = 0
top_mf = 0.0

for ne in l_ne:
    for md in l_md:
        for mf in l_mf:
            clf = RandomForestClassifier(n_estimators=ne, max_depth=md, max_features=mf, random_state=0)
            clf.fit(X_train_postprocess, y_train)
            
            y_pred_train = clf.predict_proba(X_train_postprocess)[:,1]
            y_pred_test = clf.predict_proba(X_test_postprocess)[:,1]
            y_pred_validation = clf.predict_proba(X_validation_postprocess)[:,1]
            
            fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, y_pred_test)
            fpr_train, tpr_train, thresholds_train = metrics.roc_curve(y_train, y_pred_train)
            fpr_validation, tpr_validation, thresholds_validation = metrics.roc_curve(y_validation, y_pred_validation)
            
            auc_test = metrics.auc(fpr_test, tpr_test)
            auc_train = metrics.auc(fpr_train, tpr_train)
            auc_validation = metrics.auc(fpr_validation, tpr_validation)
            
            if auc_test > highest_test_auc:
                print('Training AUC : ' + str(np.round(auc_train, 3)) + ', Test AUC :' + str(np.round(auc_test, 3)) + ' from ' + str(ne) + ' estimators, ' + str(md) + ' max depth and ' + str(mf) + ' max features***')
                top_ne = ne
                top_md = md
                top_mf = mf
                highest_test_auc = auc_test
            else:
                print('Training AUC : ' + str(np.round(auc_train, 3)) + ', Test AUC :' + str(np.round(auc_test, 3)) + ' from ' + str(ne) + ' estimators, ' + str(md) + ' max depth and ' + str(mf) + ' max features')

# Build Final Model
# After finding the best performing hyperparameter combination, you apply them to our final model. The model is built using training data and then evaluated using
# validation data. The final model and results are logged into the factsheet experiment.

model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', RandomForestClassifier(n_estimators=top_ne, max_depth=top_md, max_features=top_mf, random_state=0))])

clf=model_pipeline.fit(X_train, y_train)
y_pred_train=model_pipeline.predict(X_train)
y_pred_test=model_pipeline.predict(X_test)

y_pred_train=model_pipeline.predict_proba(X_train)[:,1]
y_pred_validation = model_pipeline.predict_proba(X_validation)[:,1]
y_pred_test=model_pipeline.predict_proba(X_test)[:,1]



fpr_train, tpr_train, thresholds_train = metrics.roc_curve(y_train, y_pred_train)
fpr_validation, tpr_validation, thresholds_validation = metrics.roc_curve(y_validation, y_pred_validation)
fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, y_pred_test)

auc_train = metrics.auc(fpr_train, tpr_train)
auc_validation = metrics.auc(fpr_validation, tpr_validation)
auc_test = metrics.auc(fpr_test, tpr_test)

print('Training AUC : ' + str(np.round(auc_train, 3)) +  ' and Test AUC :' + str(np.round(auc_test, 3)) + ' from ' + str(top_ne) + ' estimators, ' + str(top_md) + ' max depth and ' + str(top_mf) + ' max features')

# get the optimal threshold based on Youden's index
idx_opt_thres = np.argmax(tpr_test - fpr_test)
opt_threshold = thresholds_test[idx_opt_thres]
print(opt_threshold)
y_pred_class_test = y_pred_test >= opt_threshold
y_pred_class_test = y_pred_class_test.astype(int)
y_pred_class_validation = y_pred_validation >= opt_threshold
y_pred_class_validation = y_pred_validation.astype(int)

# Save the Model 
# You select the top performing model pipeline. In the next steps, you save the model pipeline along with metadata information in the project using 
# ibm-watson-machine-learning client.

model_name = 'Mortgage Approval Prediction Model'

import os
token = os.environ['USER_ACCESS_TOKEN']
url = os.environ['RUNTIME_ENV_APSX_URL']

from ibm_watson_machine_learning import APIClient
WML_CREDENTIALS = {
    "url": os.environ['RUNTIME_ENV_APSX_URL'],
    "token": os.environ['USER_ACCESS_TOKEN'],
    "instance_id" : "openshift",
    "version": '5.0'
}

wml_client = APIClient(WML_CREDENTIALS)

PROJECT_UID= os.environ['PROJECT_ID']
wml_client.set.default_project(PROJECT_UID)
# Set the current project as the default project to save the model.
PROJECT_UID= os.environ['PROJECT_ID']
wml_client.set.default_project(PROJECT_UID)

# Storing Pipeline Details
# Storing the model requires us to curate and specify some properties:
# • The name for the pipeline as specified above
# • Training data reference, that points to the data used to train the model.
# • The Software Specification, that refers to the runtime used in this Notebook and the WML deployment.
# You use the software specification default_py3.10 to store the models.

fields=X_train.columns.tolist()
metadata_dict = {'target_col' : target_col, 'probability_threshold' : opt_threshold, 'numeric_features_list':numeric_features_list,'fields':fields,'categorical_cols':categorical_cols}

training_data_references = [
                {
                    "id": "Mortgage_data",
                    "type": "connection_asset",
                    "connection": {
                        "id": None,
                    },
                    "location": {
                        "select_statement": sql_query,
                        "table_name": "Mortgage_Approval_view"
                    }
                }]
software_spec_uid = wml_client.software_specifications.get_id_by_name("runtime-23.1-py3.10")
print("Software Specification ID: {}".format(software_spec_uid))

model_props = {
        wml_client._models.ConfigurationMetaNames.NAME:model_name,
        wml_client._models.ConfigurationMetaNames.TYPE: "scikit-learn_1.1",
        wml_client._models.ConfigurationMetaNames.SOFTWARE_SPEC_UID: software_spec_uid,
        wml_client._models.ConfigurationMetaNames.LABEL_FIELD:"MORTGAGE_APPROVAL",
        wml_client._models.ConfigurationMetaNames.INPUT_DATA_SCHEMA:[{'id': '1', 'type': 'struct', 'fields': [{"name":column_name,"type":str(column_type[0])} for column_name,column_type in pd.DataFrame(X_train.dtypes).T.to_dict('list').items()]}],
        wml_client._models.ConfigurationMetaNames.TAGS: ['mortgage_prediction_pipeline_tag'],
        wml_client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES: training_data_references,
        wml_client._models.ConfigurationMetaNames.CUSTOM: metadata_dict
    }

print("Storing model .....")

published_model_details = wml_client.repository.store_model(model=model_pipeline, meta_props=model_props, 
                        training_data=df_prep.drop(["MORTGAGE_APPROVAL"], axis=1), training_target=df_prep.MORTGAGE_APPROVAL)
model_uid = wml_client.repository.get_model_id(published_model_details)
print("The model",model_name,"successfully stored in the project")
print("Model ID: {}".format(model_uid))

# Save the model using pickle
import pickle

with open("mortgage_approval.pkl", "wb") as f:
    pickle.dump(clf, f)

!ls -l

wslib.upload_file("mortgage_approval.pkl",overwrite=True)