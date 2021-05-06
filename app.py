import streamlit as st
import pandas as pd
import numpy as np
#import uuid
#import time
import os
import base64

#import random
import seaborn as sns #for visualization
import matplotlib.pyplot as plt #for visualization
from termcolor import colored as cl # text customization
import itertools # advanced tools

#from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score

from sklearn.model_selection import train_test_split 
from imblearn.over_sampling import SMOTE

#import pickle
###################################################
               #model#
###################################################
from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier

import lightgbm as lgb 
from lightgbm import LGBMClassifier
import xgboost as xgb 
from xgboost import XGBClassifier # XGBoost algorithm

import joblib

import warnings
warnings.filterwarnings("ignore")

#global X_test, X_train, y_test, y_train, X, y, input_df

@st.cache(ttl=100)
def download_link(object_to_download, download_filename, download_link_text):
   
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


def load_prediction_models(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model


@st.cache(suppress_st_warning=True,ttl=600)
def main():
    st.title("""
    Financial Fraud Detection Web App 
    """)

    st.sidebar.header('Home')
    menu = ["Home","Train and Test", "Make Prediction"]
    choice = st.sidebar.selectbox("Menu",menu)


    
    if choice == "Train and Test":
        st.subheader("Train and Test Section") 
        uploaded_file = st.sidebar.file_uploader(label="Upload your input CSV file", type=["csv"])


        if uploaded_file is not None:
            print(uploaded_file)
            print("Hello")

            
            try:
                
                input_df = pd.read_csv(uploaded_file)
                #input_df.head(10)
                input_df.drop(['isFlaggedFraud'], inplace = True, axis=1)
                st.subheader("Original Data")
                st.write(input_df.head(10)) 
                st.write(input_df.shape)

                
                alg=['Select a algorithm','LightGBM', 'XGBoost', 'Random Forest']
                classifier = st.sidebar.selectbox('Which algorithm', alg)

                
                if classifier=='LightGBM':
                	if st.sidebar.button("Evaluate"):
                		with st.spinner("Please wait while the process is ongoing."):
		                   #input_df['transaction id'] = [(uuid.uuid4()).int & (1<<32)-1 for _ in range(len(input_df.index))] 	
		                   input_df['balancediffOrig'] = input_df['newbalanceOrig'] - input_df['oldbalanceOrg']
		                   input_df['balancediffDest'] = input_df['newbalanceDest'] - input_df['oldbalanceDest']
		                   input_df[['step','amount','oldbalanceOrg', 'oldbalanceDest', 'newbalanceOrig', 'newbalanceDest', 'balancediffOrig', 'balancediffDest']] = StandardScaler().fit_transform(input_df[['step','amount','oldbalanceOrg','oldbalanceDest','newbalanceOrig','newbalanceDest','balancediffOrig','balancediffDest']])

		                   features = ['step',
		                            'type',
		                            'amount',
		                            'oldbalanceOrg',
		                            'newbalanceOrig',
		                            'oldbalanceDest',
		                            'newbalanceDest',
		                            'transaction_id',
		                            'balancediffOrig',
		                            'balancediffDest',
		                            ]

		                   label = ['isFraud']
		                   X = input_df[features]
		                   y = input_df[label] 
		                   X = X.join(pd.get_dummies(X[['type']], prefix='type')).drop(['type'], axis=1)
		                   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

		                   print(X_train.shape)
		                   print(y_train.shape)
		                   print(X_test.shape)
		                   print(y_test.shape)

		                   #SMOTE for balancing data
		                   sm = SMOTE(random_state = 42)
		                   X_train_smote, y_train_smote = sm.fit_sample(X_train, y_train)


		                   #print(pd.Series(y_SMOTE).value_counts())
		                   #X_train, X_test, y_train, y_test = train_test_split(X_SMOTE, y_SMOTE, random_state=0)
		                   st.subheader("Pre-processed Data")
		                   st.write(X.head(10))
		                   st.write(input_df.shape)

		                   
		                #Model fitting
		                   lgb_clf = LGBMClassifier()

		                   lgb_clf = lgb_clf.fit(X_train_smote, y_train_smote)
		                   
		                   

		                #Model Prediction 
		                   #warnings.filterwarnings("ignore")
		                   train_pred = lgb_clf.predict(X_train_smote)
		                   test_pred = (lgb_clf.predict_proba(X_test)[:,1] >= 0.8).astype(int)
		                 
		                   #Evaluate
		                   st.write('Train Accuracy')
		                   train_acc = roc_auc_score(y_train_smote, train_pred)	                   
		                   st.write('Accuracy:  {:.4}%'.format(train_acc*100))
		                   st.write('Test Accuracy')
		                   test_acc = roc_auc_score(y_test, test_pred)
		                   st.write("Accuracy:  {:.4}%".format(test_acc*100))
		                   #st.error('Number of fraud transactions are {}'.format(fraud)) 
		                   cm_lgb_clf = confusion_matrix(y_test, test_pred)
		                   st.write('Confusion Matrix: ', cm_lgb_clf)


		                   result = pd.DataFrame({'transaction_id':X_test['transaction_id'],'actual':y_test['isFraud'], 'predicted':test_pred})
		                   result['predicted'].replace({0: 'Genuine', 1: 'Fraud'}, inplace=True)
		                   st.write(result[result['actual']==1].head(10))


                if classifier=='Random Forest':
                	if st.sidebar.button("Evaluate"):
                		with st.spinner("Please wait while the process is ongoing."):
		                   #input_df['transaction id'] = [(uuid.uuid4()).int & (1<<32)-1 for _ in range(len(input_df.index))] 	
		                   input_df['balancediffOrig'] = input_df['newbalanceOrig'] - input_df['oldbalanceOrg']
		                   input_df['balancediffDest'] = input_df['newbalanceDest'] - input_df['oldbalanceDest']
		                   input_df[['step','amount','oldbalanceOrg', 'oldbalanceDest', 'newbalanceOrig', 'newbalanceDest', 'balancediffOrig', 'balancediffDest']] = StandardScaler().fit_transform(input_df[['step','amount','oldbalanceOrg','oldbalanceDest','newbalanceOrig','newbalanceDest','balancediffOrig','balancediffDest']])

		                   features = ['step',
		                            'type',
		                            'amount',
		                            'oldbalanceOrg',
		                            'newbalanceOrig',
		                            'oldbalanceDest',
		                            'newbalanceDest',
		                            'transaction_id',
		                            'balancediffOrig',
		                            'balancediffDest',
		                            ]

		                   label = ['isFraud']
		                   X = input_df[features]
		                   y = input_df[label] 
		                   X = X.join(pd.get_dummies(X[['type']], prefix='type')).drop(['type'], axis=1)
		                   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

		                   print(X_train.shape)
		                   print(y_train.shape)
		                   print(X_test.shape)
		                   print(y_test.shape)

		                   #SMOTE for balancing data
		                   #X_SMOTE, y_SMOTE = SMOTE().fit_sample(X_train, y_train)
		                   sm = SMOTE(random_state = 42)
		                   X_train_smote, y_train_smote = sm.fit_sample(X_train, y_train)
		                   st.subheader("Pre-processed Data")


		                   #print(pd.Series(y_SMOTE).value_counts())
		                   #X_train, X_test, y_train, y_test = train_test_split(X_SMOTE, y_SMOTE, random_state=0)

		                   st.write(X.head(10))
		                   st.write(input_df.shape)
		                #Model fitting
		                   rf_clf = RandomForestClassifier(random_state=42)

		                   rf_clf = rf_clf.fit(X_train_smote, y_train_smote)
		                                    

		                #Model Prediction

		                   #print(rf_clf)
		                   train_pred = rf_clf.predict(X_train_smote)
		                   test_pred = (rf_clf.predict_proba(X_test)[:,1] >= 0.8).astype(int)
		                   #test_pred = (xgb_clf.predict_proba(X_test)[:,1] >= 0.8).astype(int)
		                 

		                   #Evaluate
		                   st.write('Train Accuracy')
		                   train_acc = roc_auc_score(y_train_smote, train_pred)	                   
		                   st.write('Accuracy:  {:.4}%'.format(train_acc*100))
		                   st.write('Test Accuracy')
		                   test_acc = roc_auc_score(y_test, test_pred)
		                   st.write('Accuracy:  {:.4}%'.format(test_acc*100))
		                   cm_rf_clf = confusion_matrix(y_test, test_pred)
		                   st.write('Confusion Matrix: ', cm_rf_clf)

		                   #cm_rf_train_clf = confusion_matrix(y_train_smote, train_pred)
		                   #st.write('Confusion Matrix in training set: ', cm_rf_train_clf)
		                  


		                   result = pd.DataFrame({'transaction_id':X_test['transaction_id'],'actual':y_test['isFraud'], 'predicted':test_pred})
		                   result['predicted'].replace({0: 'Genuine', 1: 'Fraud'}, inplace=True)
		                   st.write(result[result['actual']==1].head(10))

		                   #st.write(result[result['actual']==1].head(10))


                if classifier=='XGBoost':
                	if st.sidebar.button("Evaluate"):
                		with st.spinner("Please wait while the process is ongoing."):
	                	   #progress = st.progress(0)
	                	   #for i in range(100):
	                	     #   time.sleep(0.1)
	                	      #  progress.progress(i+1)
		                   #input_df['transaction id'] = [(uuid.uuid4()).int & (1<<32)-1 for _ in range(len(input_df.index))]	
		                   input_df['balancediffOrig'] = input_df['newbalanceOrig'] - input_df['oldbalanceOrg']
		                   input_df['balancediffDest'] = input_df['newbalanceDest'] - input_df['oldbalanceDest']
		                   input_df[['step','amount','oldbalanceOrg', 'oldbalanceDest', 'newbalanceOrig', 'newbalanceDest', 'balancediffOrig', 'balancediffDest']] = StandardScaler().fit_transform(input_df[['step','amount','oldbalanceOrg','oldbalanceDest','newbalanceOrig','newbalanceDest','balancediffOrig','balancediffDest']])

		                   features = ['step',
		                            'type',
		                            'amount',
		                            'oldbalanceOrg',
		                            'newbalanceOrig',
		                            'oldbalanceDest',
		                            'newbalanceDest',
		                            'transaction_id',
		                            'balancediffOrig',
		                            'balancediffDest',
		                            ]

		                   label = ['isFraud']
		                   X = input_df[features]
		                   y = input_df[label] 
		                   X = X.join(pd.get_dummies(X[['type']], prefix='type')).drop(['type'], axis=1)
		                   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

		                   print(X_train.shape)
		                   print(y_train.shape)
		                   print(X_test.shape)
		                   print(y_test.shape)

		                   

		                   #SMOTE for balancing data
		                   sm = SMOTE(random_state = 42)
		                   X_train_smote, y_train_smote = sm.fit_sample(X_train, y_train)
		                   st.subheader("Pre-processed Data")
		                   st.write(X.head(10))
		                   st.write(input_df.shape)


		                #Model fitting
		                   

		                   xgb_clf = XGBClassifier(random_state = 42)

		                   xgb_clf = xgb_clf.fit(X_train_smote, y_train_smote)

		                #Model Prediction

		                

		                   train_pred = xgb_clf.predict(X_train_smote)
		                   test_pred = (xgb_clf.predict_proba(X_test)[:,1] >= 0.8).astype(int)
		                 

		                   #Evaluate
		                   st.write('Train Accuracy')
		                   train_acc = roc_auc_score(y_train_smote, train_pred)	                   
		                   st.write('Accuracy:  {:.4}%'.format(train_acc*100))
		                   st.write('Test Accuracy')
		                   test_acc = roc_auc_score(y_test, test_pred)
		                   st.write('Accuracy:  {:.4}%'.format(test_acc*100))
		                   cm_xgb_clf = confusion_matrix(y_test, test_pred)
		                   st.write('Confusion Matrix: ', cm_xgb_clf)

		                   #cm_xgb_train_clf = confusion_matrix(y_train_smote, train_pred)
		                   #st.write('Confusion Matrix in training set: ', cm_xgb_train_clf)

		                  

		                   result = pd.DataFrame({'transaction_id':X_test['transaction_id'],'actual':y_test['isFraud'], 'predicted':test_pred})
		                   result['predicted'].replace({0: 'Genuine', 1: 'Fraud'}, inplace=True)
		                   st.write(result[result['actual']==1].head(10))
		                  # st.write(result[result['actual']==1].head(10))
	       


            except Exception as e:    
                print(e)                
    
    if choice == "Make Prediction":
        #st.subheader("Summary of Exploratory Data Analysis Section")

        uploaded_file = st.sidebar.file_uploader(label="Upload your input CSV file", type=["csv"])


        if uploaded_file is not None:
            print(uploaded_file)
            print("Hello")


            try:
                df1 = pd.read_csv(uploaded_file)
                #df1['transaction_id'] = [(uuid.uuid4()).int & (1<<32)-1 for _ in range(len(df1.index))]
                #input_df.head(10)
                df1.drop(['isFraud', 'isFlaggedFraud'], inplace = True, axis=1)
                st.subheader("Original Data")
                st.write(df1.head(10)) 
                st.write(df1.shape)

              

                if st.sidebar.button("Predict"):
                	with st.spinner("Please wait while the process is ongoing."):
	                   #df1['transaction_id'] = [(uuid.uuid4()).int & (1<<32)-1 for _ in range(len(df1.index))]
	                   df1['balancediffOrig'] = df1['newbalanceOrig'] - df1['oldbalanceOrg']
	                   df1['balancediffDest'] = df1['newbalanceDest'] - df1['oldbalanceDest']
	                   df1[['step','amount','oldbalanceOrg', 'oldbalanceDest', 'newbalanceOrig', 'newbalanceDest', 'balancediffOrig', 'balancediffDest']] = StandardScaler().fit_transform(df1[['step','amount','oldbalanceOrg','oldbalanceDest','newbalanceOrig','newbalanceDest','balancediffOrig','balancediffDest']])


	                   df1.head(10)
	#df['transaction_id'].astype('int128')
	                   features = ['step',
	                                'type',
	                               'amount',
	                                'oldbalanceOrg',
	                                'newbalanceOrig',
	                                'oldbalanceDest',
	                                'newbalanceDest',
	                                'transaction_id',
	                                'balancediffOrig',
	                                'balancediffDest',
	                                #'isFraud'
	            #'merchant'
	                                 ] 
	                   df2 = df1[features]
	                   # After encoding (scroll right to see new columns)
	                   df2 = df2.join(pd.get_dummies(df2[['type']], prefix='type')).drop(['type'], axis=1)
	                   

	                   predictor = load_prediction_models("model/xgb_clf_model.pkl")
	                   #prediction = predictor.predict(df2)
	                   prediction = (predictor.predict_proba(df2)[:,1] >= 0.6333).astype(int)

	                   result = pd.DataFrame({'transaction_id':df2['transaction_id'], 'predicted':prediction})
	                   result['predicted'].replace({0: 'Genuine', 1: 'Fraud'}, inplace=True)
	                   st.subheader("Results")
	                   st.write(result[result['predicted']=="Fraud"].head(100))
	                   #result[result['predicted']=="Fraud"].to_csv('Fraud_result.csv', index=False)
	                   #st.write(result[result['actual']==1].head(10))
	                   fraud = len(result[result['predicted']=='Fraud'])
	                   st.error('Number of fraud transactions are {}'.format(fraud))
	                   #if st.button('Download Dataframe as CSV'):
	                   tmp_download_link = download_link(result[result['predicted']=='Fraud'], 'Fraud_result.csv', 'Click here to download the result')
	                   st.markdown(tmp_download_link, unsafe_allow_html=True)
	                   #st.success('Your Fraud Results are already save.')
	                #if st.button("Download file"):
	                   


##################### Result of predicting #############################

                   #df2.head()





            #          cases = len(input_df)
        # Filter data by the labels. Safe and Fraud transaction
             #         safe = len(input_df[input_df['isFraud']==0])
              #        fraud = len(input_df[input_df['isFraud']==1])
               #       fraud_percentage = round(fraud/safe*100, 2)

                #      print(cl('CASE COUNT', attrs = ['bold']))
                 #     print(cl('--------------------------------------------', attrs = ['bold']))
                 #     print(cl('Total number of cases are {}'.format(cases), attrs = ['bold']))
                  #    print(cl('Number of Non-fraud cases are {}'.format(safe), attrs = ['bold']))
                 #     print(cl('Number of Non-fraud cases are {}'.format(fraud), attrs = ['bold']))
                 #     print(cl('Percentage of fraud cases is {}'.format(fraud_percentage), attrs = ['bold']))
                  #    print(cl('--------------------------------------------', attrs = ['bold']))




        #visual representation of instances per class
                 #     st.bar_chart(input_df.isFraud.value_counts()) 
        #Distribution of the frequency of all transactions              
                 #     fig = plt.figure(figsize=(10, 3))
                 #     sns.distplot(input_df.step)
                  #    plt.title('Distribution of Transactions over the Time')   
                  #    st.pyplot(fig)
        #Fraud transaction boxplot for amount distribution

                  #    fig=plt.figure(figsize=(10,3))
                  #    plt.title('Fraud Transaction Amount Distribution')
                  #    ax = sns.boxplot(input_df["amount"])    
                  #    st.pyplot(fig)
       

                #if st.sidebar.button("View Line Chart"): 
                #Distribution of the frequency of all transactions
                   # fig = plt.figure(figsize=(10, 3))
                   # sns.distplot(input_df.step)
                   # plt.title('Distribution of Transactions over the Time')   
                   # st.pyplot(fig)    


                #if st.sidebar.button("View Boxplot"):  #Fraud transaction boxplot for amount distribution
                    #fig=plt.figure(figsize=(10,3))
                   # plt.title('Fraud Transaction Amount Distribution')
                    #ax = sns.boxplot(input_df["amount"])    
                    #st.pyplot(fig)


            except Exception as e:    
                print(e)

if __name__ == '__main__':
    main()

