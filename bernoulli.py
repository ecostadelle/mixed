import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
import warnings

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, MinMaxScaler, Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

from tableshift import get_dataset
from tableshift.core.features import PreprocessorConfig

from ud_bagging import UDBaggingClassifier
from ud_naive_bayes import InterpretableBernoulliNB

splits    = ['train',
             'validation',
             'id_test',
             'ood_test',
             'ood_validation']



names = ['ASSISTments',
         'Childhood Lead',
         'College Scorecard',
         'Diabetes',
         'FICO HELOC',
         'Food Stamps',
         'Hospital Readmission',
         'Hypertension',
        #  'ICU Length of Stay',
        #  'ICU Mortality',
         'Income',
        #  'Public Health Insurance',
         'Sepsis',
         'Unemployment',
         'Voting',
         ]


datasets  = ['assistments',
             'nhanes_lead',
             'college_scorecard',
             'brfss_diabetes',
             'acsfoodstamps',
             'heloc',
             'diabetes_readmission',
             'brfss_blood_pressure',
            #  'mimic_extract_los_3',
            #  'mimic_extract_mort_hosp',
             'acsincome',
            #  'acspubcov',
             'physionet',
             'acsunemployment',
             'anes',
             ]

warnings.filterwarnings("ignore")

print('{0: <25}'.format('DATASET'), end=" & ")
print('acc_id', end=" & ")
print('f-1_id', end=" & ")
print('acc_ood', end=" & ")
print('f-1_odd', end=" & ")
print('acc_odd1', end=" & ")
print('f-1_odd1', end=" & ")
print('acc_odd2', end=" & ")
print('f-1_odd2', end=" & ")
print('acc_odd3', end=" & ")
print('f-1_odd3', end=" & ")
print('acc_odd4', end=" & ")
print('f-1_odd4')


# for j in [0,2,6,7,11]:
#     d=datasets[j]
    
# # 
for j,d in enumerate(datasets):
    print('{0: <25}'.format(names[j]), end=" & ")
    
    dset = get_dataset(
        name=d, 
        cache_dir='../tableshift/tmp', 
        use_cached=True
        )
    
    X_a, y_a, _, _ = dset.get_pandas('train')
    
    model_a = UDBaggingClassifier(base_estimator=InterpretableBernoulliNB(binarize=0.5),
                                    n_estimators=500, max_features="sqrt", 
                                    n_jobs=-1, random_state=2)
    model_a.fit(X_a.values, y_a)
    
    X_id, y_id, _, _ = dset.get_pandas('id_test')
    
    y_hat_id = model_a.predict(X_id)
    acc_id = accuracy_score(y_id, y_hat_id)
    f1_id = f1_score(y_id, y_hat_id)
    
    print(f'{acc_id:.03f}', end=" & ")
    print(f'{f1_id:.03f}', end=" & ")
    
    X_b, y_b, _, _ = dset.get_pandas('ood_test')
    
    model_b = UDBaggingClassifier(base_estimator=InterpretableBernoulliNB(binarize=0.5),
                                    n_estimators=500, max_features="sqrt", 
                                    n_jobs=-1, random_state=2)
    
    y_hat_b = model_a.predict(X_b.values)
    
    
    acc_ood = accuracy_score(y_b, y_hat_b)
    f1_ood = f1_score(y_b, y_hat_b)
    
    print(f'{acc_ood:.03f}', end=" & ")
    print(f'{f1_ood:.03f}', end=" & ")
    
    if f1_ood==0:
        print (f'{i} has a NaN.')
    else:
        model_b.fit(X_b.values, y_hat_b)
        
        dbcp_a = model_a.feature_importances_
        dbcp_b = model_b.feature_importances_
        
        mssf_a = model_a.sufficiency_based_feature_importances(X_a.values)
        mssf_b = model_b.sufficiency_based_feature_importances(X_b.values)
        
        strategy = [dbcp_b, dbcp_b-dbcp_a, mssf_b, mssf_b-mssf_a]
        
        for i,s in enumerate(strategy):

            min, max = s.min(), s.max()
            w = (s-min)/(max-min)
            p = w/w.sum()
            model_adapted_brs = UDBaggingClassifier(base_estimator=InterpretableBernoulliNB(binarize=0.5),
                                                    n_estimators=500, max_features="sqrt", 
                                                    n_jobs=-1, random_state=2, 
                                                    biased_subspaces=True, feature_bias=p)
            
            model_adapted_brs.fit(X_a.values, y_a)
            y_hat = model_adapted_brs.predict(X_b.values)
            acc_oodi = accuracy_score(y_b, y_hat)
            f1_oodi = f1_score(y_b, y_hat)

            print(f'{i} {acc_oodi:.03f}', end=" & ")
            print(f'{i} {f1_oodi:.03f}', end=" & ")
        print()