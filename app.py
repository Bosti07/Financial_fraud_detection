import streamlit as st
import pandas as pd
import numpy as np
import uuid
import time
import os
import base64

import random
import seaborn as sns #for visualization
import matplotlib.pyplot as plt #for visualization
from termcolor import colored as cl # text customization
import itertools # advanced tools

#from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
from sklearn.metrics import roc_auc_score, accuracy_score

from sklearn.model_selection import train_test_split 
from imblearn.over_sampling import SMOTE

###################################################
               #model#
###################################################
from sklearn.ensemble import RandomForestClassifier


import lightgbm as lgb 
from lightgbm import LGBMClassifier
import xgboost as xgb 
from xgboost import XGBClassifier # XGBoost algorithm

import joblib

import warnings
warnings.filterwarnings("ignore")


hide_streamlit_style = """
                           <style>
                           #MainMenu {visibility: hidden;}
                           footer {visibility: hidden;}
                           </style>
                      """
st.markdown(hide_streamlit_style, unsafe_allow_html = True)

	

def download_link(object_to_download, download_filename, download_link_text):
   
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


def load_prediction_models(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model



def main():
    #st.markdown("""<h1 style='text-align: center; color: White;'>Welcome to Frapp</h1>
    #	           <h1 style='text-align: center; '>🕵️‍♂️</h1>""", unsafe_allow_html=True)
    #st.markdown("<h1 style='text-align: center; color: Black;'>🕵️‍♂️</h1>", unsafe_allow_html=True)
    #st.markdown("<h1 style='text-align: center; color: Black;'>:shipit:</h1>", unsafe_allow_html=True)
   # st.title(""" 
   #     **_Welcome to Frapp_**
   #        	        \n🕵️‍♂️
   # """)
   # col1, col2, col3 = st.sidebar.beta_columns([1,1,1])

    #with col1:
   # 	st.write("")

    #with col2:
   # 	st.write("⚠️")

    #with col3:
   # 	st.write("")

    col1, col2, col3 = st.sidebar.beta_columns([1,1,1])

    with col1:
    	st.write("")

    with col2:
    	st.markdown("""<h2 style='text-align: center; color: Black;'>⚠️</h1>
    		           <h1 style='text-align: center; color: Black;'>Frapp</h1>""", unsafe_allow_html=True)

    with col3:
    	#st.header('Frapp')
    	st.write("")
    #st.sidebar.write("Fraud Detection Web Application")
    st.sidebar.markdown("<h4 style='color: Black;'>Fraud Detection Web Application</h4>", unsafe_allow_html=True)
    #st.sidebar.header('Frapp')
    menu = ["Home","Train and Test", "Make Prediction"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def read_file(uploaded_file):
    
        return pd.read_csv(uploaded_file)

    def plot_metrics(metrics_list):
    	st.set_option('deprecation.showPyplotGlobalUse', False)


    	if 'Confustion Matrix' in metrics_list:
    		st.subheader("Confusion Matrix ")
    		plot_confusion_matrix(model, X_test, y_test, display_labels= class_names)
    		st.pyplot() 
    	if 'ROC Curve' in metrics_list:
     
            st.subheader("ROC Curve")
            plot_roc_curve(model, X_test, y_test)
            st.pyplot()

    if choice == "Home":
    	st.markdown("""<h1 style='text-align: center; color: Black;'>Welcome to Frapp</h1>
    		<h1 style='text-align: center; '>🕵️‍♂️</h1>""", unsafe_allow_html=True)
    	st.markdown("Frapp is a fraud detection web application that offers to train and test and make predictions of your financial datasets. Imagine a platform making it easy for aspiring data scientists!", unsafe_allow_html=True)
    	print("Home")            

    class_names = ['Genuine', 'Fraud']
    if choice == "Train and Test":
        st.markdown("""<h1 style='text-align: center; color: Black;'>Train and test with Frapp!</h1>
    		           <h1 style='text-align: center; '>🕵️‍♂️</h1>""", unsafe_allow_html=True)
        st.markdown("""<h4 style='text-align: left;'>1. Check the file requirements and modify first your variables.</h4>
        			   <h4 style='text-align: left;'>2. Upload your CSV file.</h4>
        			   <h4 style='text-align: left;'>3. Choose from the algorithm provided.</h4>
        			   <h4 style='text-align: left;'>4. Choose the metric you want to use to evaluate your model.</h4>
        			   <h4 style='text-align: Center;'>Then it's done! Wait for the results to be displayed, it may take a while...</h4>

        			""", unsafe_allow_html=True)

        optional = st.sidebar.beta_expander("File requirements", False)
        optional.write("step")
        optional.write("type")
        optional.write("amount")
        optional.write("nameOrig")
        optional.write("nameDest")
        optional.write("oldBalanceOrig")
        optional.write("newBalanceOrig")
        optional.write("oldBalanceDest")
        optional.write("newBalanceDest")
        optional.write("transaction_id")
        optional.write("isFraud(for train and test only)")
        optional.warning("All the columns must be followed accordingly or the system may not be able to produce appropriate results.")
        #st.subheader("Train and Test Section")
        #upload_file() 
        uploaded_file = st.sidebar.file_uploader(label="Upload your input CSV file", type=["csv"])

        
        if uploaded_file is not None:
            print(uploaded_file)
            print("Hello")

            
            try:
               
                #input_df = read_file(uploaded_file)

                input_df = pd.read_csv(uploaded_file)
                input_df.drop(['isFlaggedFraud'], inplace = True, axis = 1)
                st.write("")
                st.write("")
                st.write("")
                st.subheader("Original Data")
                st.dataframe(input_df.head(10)) 
                st.write("The Dataset has",input_df.shape[0], "rows", "and",input_df.shape[1], "columns." )

                
                alg=['Select a algorithm','LightGBM', 'Random Forest','XGBoost']
                classifier = st.sidebar.selectbox('Which algorithm', alg)

                
                if classifier=='LightGBM':                	
                	metrics = st.sidebar.multiselect("What metrics to plot? 📊",('Confustion Matrix', 'ROC Curve'))
                	if st.sidebar.button("Start 👨‍🔬"):
                		with st.spinner("Please wait while the process is ongoing."):
		                   input_df[['step','amount','oldBalanceOrig', 'oldBalanceDest', 'newBalanceOrig', 'newBalanceDest']] = StandardScaler().fit_transform(input_df[['step','amount','oldBalanceOrig','oldBalanceDest','newBalanceOrig','newBalanceDest']])

		                   features = ['step',
		                            'type',
		                            'amount',
		                            'oldBalanceOrig',
		                            'newBalanceOrig',
		                            'oldBalanceDest',
		                            'newBalanceDest',
		                            'transaction_id',
		                            ]

		                   label = ['isFraud']
		                   X = input_df[features]
		                   y = input_df[label] 
		                   X = X.join(pd.get_dummies(X[['type']], prefix='type')).drop(['type'], axis=1)
		                   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

		                   print(X_train.shape)
		                   print(y_train.shape)
		                   print(X_test.shape)
		                   print(y_test.shape)

		                   #SMOTE for balancing data
		                   sm = SMOTE(random_state = 1)
		                   X_train_smote, y_train_smote = sm.fit_sample(X_train, y_train)


		                   
		                   #st.subheader("Pre-processed Data")
		                   #st.write(X.head(10))
		                   #st.write(input_df.shape)

		                   
		                #Model fitting
		                   model = LGBMClassifier()

		                   model = model.fit(X_train_smote, y_train_smote)
		                   
		                   

		                #Model Prediction
		                   test_pred = (model.predict_proba(X_test)[:,1] >= 0.5).astype(int)
		                 
		                   #Evaluate
		                   st.write('Test Accuracy')
		                   test_acc = roc_auc_score(y_test, test_pred)
		                   st.write("**_Accuracy_**:  ", test_acc.round(3))
		                  
		                   
		                   st.subheader("RESULTS:")
		                   result = pd.DataFrame({'transaction_id':X_test['transaction_id'],'actual':y_test['isFraud'], 'predicted':test_pred})
		                   result['predicted'].replace({0: 'Genuine', 1: 'Fraud'}, inplace=True)
		                   st.table(result[result['actual']==1].head(10))
		                   plot_metrics(metrics)


                if classifier=='Random Forest':
                	metrics = st.sidebar.multiselect("What metrics to plot? 📊",('Confustion Matrix', 'ROC Curve'))
                	if st.sidebar.button("Start 👨‍🔬"):
                		with st.spinner("Please wait while the process is ongoing."):

		                   input_df[['step','amount','oldBalanceOrig', 'oldBalanceDest', 'newBalanceOrig', 'newBalanceDest']] = StandardScaler().fit_transform(input_df[['step','amount','oldBalanceOrig','oldBalanceDest','newBalanceOrig','newBalanceDest']])

		                   features = ['step',
		                            'type',
		                            'amount',
		                            'oldBalanceOrig',
		                            'newBalanceOrig',
		                            'oldBalanceDest',
		                            'newBalanceDest',
		                            'transaction_id'	                            
		                            ]

		                   label = ['isFraud']
		                   X = input_df[features]
		                   y = input_df[label] 
		                   X = X.join(pd.get_dummies(X[['type']], prefix='type')).drop(['type'], axis=1)
		                   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

		                   print(X_train.shape)
		                   print(y_train.shape)
		                   print(X_test.shape)
		                   print(y_test.shape)

		                   #SMOTE for balancing data
		                  
		                   sm = SMOTE(random_state = 1)
		                   X_train_smote, y_train_smote = sm.fit_sample(X_train, y_train)
		                   #st.subheader("Pre-processed Data")

		                   #st.write(X.head(10))
		                   #st.write(input_df.shape)
		                #Model fitting
		                   model = RandomForestClassifier()

		                   model = model.fit(X_train_smote, y_train_smote)
		                                    

		                #Model Prediction

		                   test_pred = (model.predict_proba(X_test)[:,1] >= 0.5).astype(int)
		                 
		                   #Evaluate

		                   st.write('Test Accuracy')
		                   test_acc = roc_auc_score(y_test, test_pred)
		                   st.write("**_Accuracy_**:  ", test_acc.round(3))

		                   st.subheader("RESULTS:")		                  
		                   result = pd.DataFrame({'transaction_id':X_test['transaction_id'],'actual':y_test['isFraud'], 'predicted':test_pred})
		                   result['predicted'].replace({0: 'Genuine', 1: 'Fraud'}, inplace=True)
		                   st.table(result[result['actual']==1].head(10))
		                   plot_metrics(metrics)



                if classifier=='XGBoost':
                	metrics = st.sidebar.multiselect("What metrics to plot? 📊",('Confustion Matrix', 'ROC Curve'))
                	if st.sidebar.button("Start 👨‍🔬"):
                		with st.spinner("Please wait while the process is ongoing."):
	               	
		                  
		                   input_df[['step','amount','oldBalanceOrig', 'oldBalanceDest', 'newBalanceOrig', 'newBalanceDest']] = StandardScaler().fit_transform(input_df[['step','amount','oldBalanceOrig','oldBalanceDest','newBalanceOrig','newBalanceDest']])

		                   features = ['step',
		                            'type',
		                            'amount',
		                            'oldBalanceOrig',
		                            'newBalanceOrig',
		                            'oldBalanceDest',
		                            'newBalanceDest',
		                            'transaction_id'
		      
		                            ]

		                   label = ['isFraud']
		                   X = input_df[features]
		                   y = input_df[label] 
		                   X = X.join(pd.get_dummies(X[['type']], prefix='type')).drop(['type'], axis=1)
		                   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

		                   print(X_train.shape)
		                   print(y_train.shape)
		                   print(X_test.shape)
		                   print(y_test.shape)

		                   

		                   #SMOTE for balancing data
		                   sm = SMOTE(random_state = 1)
		                   X_train_smote, y_train_smote = sm.fit_sample(X_train, y_train)
		                   #st.subheader("Pre-processed Data")
		                   #st.write(X.head(10))
		                   #st.write(input_df.shape)


		                #Model fitting
		                   

		                   model = XGBClassifier(  random_state = 1,
                        						   learning_rate = 0.05, 
                                                   max_depth = 5,
                        						   n_estimators = 300, 
                        						   colsample_bytree = 0.7, 
                        						   gamma = 0.0,
                        						   )

		                   model = model.fit(X_train_smote, y_train_smote, eval_metric=["error", "logloss"])

		                #Model Prediction

		                
		                   test_pred = model.predict_proba(X_test)[:,1] >= 0.8
		                 

		            
		                   st.write('Test Accuracy')
		                   test_acc = roc_auc_score(y_test, test_pred)
		                   st.write("**_Accuracy_**:  ", test_acc.round(3))

		                   st.subheader("RESULTS:")

		                   
		                   result = pd.DataFrame({'transaction_id':X_test['transaction_id'],'actual':y_test['isFraud'], 'predicted':test_pred})
		                   result['predicted'].replace({0: 'Genuine', 1: 'Fraud'}, inplace=True)
		                   st.table(result[result['actual']==1].head(10))
		                   plot_metrics(metrics)
		                  # st.write(result[result['actual']==1].head(10))
	       


            except Exception as e:    
                print(e)                
    
    if choice == "Make Prediction":
        st.markdown("""<h1 style='text-align: center; color: Black;'>Predict fraud with Frapp!</h1>
    		           <h1 style='text-align: center; '>🕵️‍♂️</h1>""", unsafe_allow_html=True)

        st.markdown("""<h4 style='text-align: left;'>1. Check the file requirements and modify first your variables.</h4>
        			   <h4 style='text-align: left;'>2. Upload your CSV file.</h4>
        			   <h4 style='text-align: left;'>3. Your file will be scanned through our own model using XGBoost.</h4>
					   <h4 style='text-align: Center;'>Then it's done! Wait for the results to be displayed, it may take a while...</h4>

        			""", unsafe_allow_html=True)

        uploaded_file = st.sidebar.file_uploader(label="Upload your input CSV file", type=["csv"])


        if uploaded_file is not None:
            print(uploaded_file)
            print("Hello")


            try:
                df1 = pd.read_csv(uploaded_file)
                df1.drop(['isFraud', 'isFlaggedFraud'], inplace = True, axis=1)
                st.write("")
                st.write("")
                st.write("")
                st.subheader("Original Data")
                st.dataframe(df1.head(10)) 
              

                if st.sidebar.button("Start"):
                	with st.spinner("Please wait while the process is ongoing."):
                		#filtering only transfer and cash_out data
                	   df3=df1[df1['type'].isin(['TRANSFER','CASH_OUT'])]
	
	                   features = ['step',
	                                'type',
	                               'amount',
	                                'oldBalanceOrig',
	                                'newBalanceOrig',
	                                'oldBalanceDest',
	                                'newBalanceDest',
	                                'transaction_id'
	                                
	                                 ] 
	                   df2 = df3[features]
	                   # After encoding (scroll right to see new columns)
	                   df2 = df2.join(pd.get_dummies(df2[['type']], prefix='type')).drop(['type'], axis=1)
	                   

	                   predictor = load_prediction_models("model/xgb_clf_model.pkl")
	                   #prediction = predictor.predict(df2)
	                   prediction = (predictor.predict_proba(df2)[:,1] >= 0.87).astype(int)

	                   result = pd.DataFrame({'transaction_id':df2['transaction_id'], 'predicted':prediction})
	                   result['predicted'].replace({0: 'Genuine', 1: 'Fraud'}, inplace=True)
	                   st.markdown("""
									<style>
									.big-font {
    											font-size:35px !important;
											  }
									</style>
						""", unsafe_allow_html=True)
	                   st.markdown('<p class="big-font">Results </p>', unsafe_allow_html=True)
	                   optional = st.beta_expander("Display Results", False)

	                   optional.table(result[result['predicted']=="Fraud"].head(100))

	                   fraud = len(result[result['predicted']=='Fraud'])
	                   st.error('Number of fraud transactions are {}'.format(fraud))

	                   tmp_download_link = download_link(result[result['predicted']=='Fraud'], 'Fraud_result.csv', 'Click here to download the result')
	                   st.markdown(tmp_download_link, unsafe_allow_html=True)

	                                         

##################### Result of predicting #############################

                 


            except Exception as e:    
                print(e)

if __name__ == '__main__':
    main()

