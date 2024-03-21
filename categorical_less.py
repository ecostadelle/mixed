import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import os
import pandas as pd
import time
import warnings

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, MinMaxScaler, Normalizer, OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

from tableshift import get_dataset
from tableshift.core.features import PreprocessorConfig

from ud_bagging import UDBaggingClassifier
from ud_naive_bayes import InterpretableBernoulliNB, InterpretableMultinomialNB, InterpretableCategoricalNB


def print_header():
    print("\\captionof{table}{Model Adaptation with ensemble of ", end="")
    print("Categorical (simplified)", end="")
    print(" \\textit{Na\\\"ive} Bayes}")
    print("\\begin{tabular}{lcc|cc|cc|cc|cc|cc}")
    print("& \\multicolumn{2}{c}{ID}") 
    print("& \\multicolumn{2}{c}{OOD}") 
    print("& \\multicolumn{2}{c}{Strategy 1}") 
    print("& \\multicolumn{2}{c}{Strategy 2}") 
    print("& \\multicolumn{2}{c}{Strategy 3}") 
    print("& \\multicolumn{2}{c}{Strategy 4} \\\\")

    print(f'{"DATASET":25}', end="")
    print(' & ACC', end="")
    print(' & F-1', end="")
    print(' & ACC', end="")
    print(' & F-1', end="")
    print(' & ACC', end="")
    print(' & F-1', end="")
    print(' & ACC', end="")
    print(' & F-1', end="")
    print(' & ACC', end="")
    print(' & F-1', end="")
    print(' & ACC', end="")
    print(' & F-1 \\\\')
    print("\\midrule")
    
def print_footer():
    print('\\bottomrule')
    print('\\end{tabular}')
    print('\\vspace{10pt}')
    
splits    = ['train',
             'validation',
             'id_test',
             'ood_test',
             'ood_validation']


data = [
    [ 'ASSISTments',             'assistments'             ],
    [ 'Childhood Lead',          'nhanes_lead'             ],
    [ 'College Scorecard',       'college_scorecard'       ], 
    [ 'Diabetes',                'brfss_diabetes'          ],
    [ 'FICO HELOC',              'heloc'                   ],
    [ 'Food Stamps',             'acsfoodstamps'           ],
    [ 'Hospital Readmission',    'diabetes_readmission'    ],    
    [ 'Hypertension',            'brfss_blood_pressure'    ],    
    #[ 'ICU Length of Stay'       'mimic_extract_los_3'     ],    
    #[ 'ICU Mortality',           'mimic_extract_mort_hosp' ],        
    [ 'Income',                  'acsincome'               ],
    #[ 'Public Health Insurance', 'acspubcov'               ],
    [ 'Sepsis',                  'physionet'               ],
    [ 'Unemployment',            'acsunemployment'         ],
    [ 'Voting',                  'anes'                    ]
    ]

warnings.filterwarnings("ignore")

print_header()

for dataset,identifier in data:
    time_ = time.time()
    print(f'{dataset:25}', end="")
    
    dset = get_dataset(
        name=identifier, 
        cache_dir='../tableshift/tmp', 
        use_cached=True
        )
    
    encode = OrdinalEncoder()
    
    X_a, y_a, _, _ = dset.get_pandas('train')
    X_id, y_id, _, _ = dset.get_pandas('id_test')
    X_b, y_b, _, _ = dset.get_pandas('ood_test')
    
    encode.fit(pd.concat([X_a, X_id,X_b]))
    
    X_an  = encode.transform(X_a)
    X_idn = encode.transform(X_id)
    X_bn  = encode.transform(X_b)
    
    del X_a
    del X_id
    del X_b
    
    
    model_a = UDBaggingClassifier(base_estimator=InterpretableCategoricalNB(min_categories=2),
                                    n_estimators=500, max_features="sqrt", 
                                    n_jobs=-1, random_state=2)
    model_a.fit(X_an, y_a)
    

    
    y_hat_id = model_a.predict(X_idn)
    acc_id = accuracy_score(y_id, y_hat_id)
    f1_id = f1_score(y_id, y_hat_id)
    f1_ood = None
    
    print(f' & {acc_id:.03f}', end="")
    print(f' & {f1_id:.03f}', end="")
    
    # if f1_id:

    y_hat_b = model_a.predict(X_bn)
    
    acc_ood = accuracy_score(y_b, y_hat_b)
    f1_ood = f1_score(y_b, y_hat_b)
    
    print(f' & {acc_ood:.03f}', end="")
    print(f' & {f1_ood:.03f}', end="")
    
    if f1_ood > 0:
    
        # model_b = UDBaggingClassifier(base_estimator=InterpretableCategoricalNB(min_categories=2),
        #                                 n_estimators=500, max_features="sqrt", 
        #                                 n_jobs=-1, random_state=2)
    
        # model_b.fit(X_bn, y_hat_b)
        
        
        dbcp_a = model_a.feature_importances_by_(X_idn)
        dbcp_b = model_a.feature_importances_by_(X_bn)
        
        mssf_a = model_a.sufficiency_based_feature_importances(X_idn)
        mssf_b = model_a.sufficiency_based_feature_importances(X_bn)
        
        strategy = [dbcp_b, dbcp_b-dbcp_a, mssf_b, mssf_b-mssf_a]
        
        for i,s in enumerate(strategy):

            min, max = s.min(), s.max()
            w = (s-min)/(max-min)
            p = w/w.sum()
            model_adapted_brs = UDBaggingClassifier(base_estimator=InterpretableCategoricalNB(min_categories=2),
                                                    n_estimators=500, max_features="sqrt", 
                                                    n_jobs=-1, random_state=2, 
                                                    biased_subspaces=True, feature_bias=p)
            
            model_adapted_brs.fit(X_an, y_a)
            y_hat = model_adapted_brs.predict(X_bn)
            acc_oodi = accuracy_score(y_b, y_hat)
            f1_oodi = f1_score(y_b, y_hat)

            print(f' & {acc_oodi:.03f}', end="")
            print(f' & {f1_oodi:.03f}', end="")
    print(f' \\\\ % {int(time.time() - time_):6d} seconds')
print_footer()