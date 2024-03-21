import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import os
import pandas as pd
import time
import warnings

# rich
import random
import time

from rich.live import Live
from rich.table import Table

from rich.layout import Layout
# end rich

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, MinMaxScaler, Normalizer, OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

from tableshift import get_dataset
from tableshift.core.features import PreprocessorConfig
from tableshift.core.tasks import get_task_config

from ud_bagging import UDBaggingClassifier, balanced_weight_vector
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
    # [ 'ASSISTments',             'assistments'             ],
    # [ 'Childhood Lead',          'nhanes_lead'             ],
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
        initialize_data=False,
        use_cached=True
    )

    X_a, y_a, _, _ = dset.get_pandas('train')
    X_id, y_id, _, _ = dset.get_pandas('id_test')
    X_b, y_b, _, _ = dset.get_pandas('ood_test')

    X_a.reset_index(drop=True, inplace=True)
    X_id.reset_index(drop=True, inplace=True)
    X_b.reset_index(drop=True, inplace=True)
    
    idx_a = X_a.dropna().index
    idx_id = X_id.dropna().index
    idx_b = X_b.dropna().index
    
    y_a = y_a.values[idx_a]
    y_id = y_id.values[idx_id]
    y_b = y_b.values[idx_b]

    hidden_cat = {
        f.name: f.value_mapping
        for f in get_task_config(identifier).feature_list.features 
        if f.kind.__name__ != 'CategoricalDtype' 
        and not f.is_target 
        and f.value_mapping 
        and f.name in X_a.columns
    }

    for f in hidden_cat:
        X_a[f] = X_a[f].astype('str')
        X_id[f] = X_id[f].astype('str')
        X_b[f] = X_b[f].astype('str')


    discretizer = KBinsDiscretizer(
        n_bins=5, 
        encode='ordinal', 
        strategy='quantile', 
        subsample=200_000, 
        random_state=0
    )
    encode = OrdinalEncoder(
        handle_unknown='use_encoded_value', 
        unknown_value=np.nan, 
        encoded_missing_value=np.nan
    )

    num_feats = X_a.columns[X_a.dtypes != 'object']
    cat_feats = X_a.columns[X_a.dtypes == 'object']

    if len(num_feats)>0:
        X_a.loc[idx_a,num_feats] = discretizer.fit_transform(X_a.loc[idx_a,num_feats])
        X_id.loc[idx_id,num_feats] = discretizer.transform(X_id.loc[idx_id,num_feats])
        X_b.loc[idx_b,num_feats] = discretizer.transform(X_b.loc[idx_b,num_feats])

    X_a.loc[idx_a,X_a.columns] = encode.fit_transform(X_a)
    X_id.loc[idx_id,X_a.columns] = encode.transform(X_id)
    X_b.loc[idx_b,X_b.columns] = encode.transform(X_b)
    
    idx_id = X_id.dropna().index
    idx_b = X_b.dropna().index
    
    y_id = y_id[idx_id]
    y_b = y_b[idx_b]
    
    
    model = UDBaggingClassifier(
        estimator=InterpretableCategoricalNB(
            min_categories=2, 
            alpha=1e-10, 
            force_alpha=True
        ),
        n_estimators=500, max_features=int(np.sqrt(X_a.shape[1])), 
        n_jobs=-1, random_state=2
    )
    
    sample_weight=None
    sample_weight=balanced_weight_vector(y_a)
    model.fit(X_a.values, y_a, sample_weight=sample_weight)
    
    y_hat_id = model.predict(X_id.values[idx_id])
    acc_id = accuracy_score(y_id, y_hat_id)
    f1_id = f1_score(y_id, y_hat_id)
    f1_ood = None
    
    print(f' & {acc_id:.03f}', end="")
    print(f' & {f1_id:.03f}', end="")
    
    # if f1_id:

    y_hat_b = model.predict(X_b.values[idx_b])
    
    acc_ood = accuracy_score(y_b, y_hat_b)
    f1_ood = f1_score(y_b, y_hat_b)
    
    print(f' & {acc_ood:.03f}', end="")
    print(f' & {f1_ood:.03f}', end="")
    
    if f1_ood > 0:
        
        dbcp_a = model.feature_importances_by_(X_id.values[idx_id])
        dbcp_b = model.feature_importances_by_(X_b.values[idx_b])
        
        mssf_a = model.sufficiency_based_feature_importances(X_id.values[idx_id])
        mssf_b = model.sufficiency_based_feature_importances(X_b.values[idx_b])
        
        strategy = [dbcp_b, dbcp_b-dbcp_a, mssf_b, mssf_b-mssf_a]
        
        for i,s in enumerate(strategy):

            min, max = s.min(), s.max()
            w = (s-min)/(max-min)
            p = w/w.sum()
            model_adapted_brs = UDBaggingClassifier(
                estimator=InterpretableCategoricalNB(
                    min_categories=2, 
                    alpha=1e-10, 
                    force_alpha=True
                ),
                n_estimators=500, max_features=int(np.sqrt(X_a.shape[1])), 
                n_jobs=-1, random_state=2, 
                biased_subspaces=True, feature_bias=p
            )
            
            model_adapted_brs.fit(X_a.values, y_a, sample_weight=sample_weight)
            y_hat = model_adapted_brs.predict(X_b.values[idx_b])
            acc_oodi = accuracy_score(y_b, y_hat)
            f1_oodi = f1_score(y_b, y_hat)

            print(f' & {acc_oodi:.03f}', end="")
            print(f' & {f1_oodi:.03f}', end="")
    print(f' \\\\ % {int(time.time() - time_):6d} seconds')
print_footer()