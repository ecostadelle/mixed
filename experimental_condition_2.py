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
from rich.table import Column,Table
from rich.panel import Panel
from rich.console import Console
from rich import box

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


from util import save_result


c = Console(width=150)
layout= Layout()
table1 = Table(
    Column(header='Dataset', width=20), 
    Column(header='ID', width=5),
    Column(header='OOD', width=5),
    Column(header='S1', width=5),
    Column(header='S2', width=5),
    Column(header='S3', width=5),
    Column(header='S4', width=5),
    title="Accuracy")
table2 = Table(
    Column(header='ID', width=5),
    Column(header='OOD', width=5),
    Column(header='S1', width=5),
    Column(header='S2', width=5),
    Column(header='S3', width=5),
    Column(header='S4', width=5), 
    Column(header='time'),
    title="F1-Score")
layout.split_row(table1, table2)
c.print(layout)

drop_columns = {
    'anes': ['VCF0901b'],
    'acsfoodstamps': ['ST'],
    'acsincome': ['ST']
}

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

cols = ['Dataset', 'ID', 'OOD', 'S1', 'S2', 'S3', 'S4']

df_result = {
    'acc': pd.DataFrame(index=np.array(data)[:,1], columns=cols),
    'f-1': pd.DataFrame(index=np.array(data)[:,1], columns=cols)}

df_result['acc'].Dataset = np.array(data)[:,0]
df_result['f-1'].Dataset = np.array(data)[:,0]


try:
    for dataset,identifier in data:
        time_ = time.time()
        
        table1.add_row(f"{dataset}")
        table2.add_row("")
        c.print(layout)
        
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
        
        y_a.reset_index(drop=True, inplace=True)
        y_id.reset_index(drop=True, inplace=True)
        y_b.reset_index(drop=True, inplace=True)
        
        if identifier in drop_columns.keys():
            X_a.drop(drop_columns[identifier], inplace=True, axis=1)
            X_id.drop(drop_columns[identifier], inplace=True, axis=1)
            X_b.drop(drop_columns[identifier], inplace=True, axis=1)
        
        fillna = True
        max_features=int(np.sqrt(X_a.shape[1]))
        balanced = False
        
        if fillna:
            X_a.fillna(-1, inplace=True)
            X_id.fillna(-1, inplace=True)
            X_b.fillna(-1, inplace=True)
        
        idx_a = X_a.dropna().index
        idx_id = X_id.dropna().index
        idx_b = X_b.dropna().index
        
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

        X_a.loc[idx_a,:] = encode.fit_transform(X_a.loc[idx_a,:].astype('str'))
        X_id.loc[idx_id,:] = encode.transform(X_id.loc[idx_id,:].astype('str'))
        X_b.loc[idx_b,:] = encode.transform(X_b.loc[idx_b,:].astype('str'))
        
        idx_a = X_a.dropna().index
        idx_id = X_id.dropna().index
        idx_b = X_b.dropna().index
            
        y_a = y_a.loc[idx_a].values
        y_id = y_id.loc[idx_id].values
        y_b = y_b.loc[idx_b].values
        
        
        model = UDBaggingClassifier(
            estimator=InterpretableCategoricalNB(
                min_categories=2, 
                alpha=1e-10, 
                force_alpha=True
            ),
            n_estimators=500, max_features=max_features, 
            n_jobs=-1, random_state=2
        )
        
        if balanced:
            sample_weight=balanced_weight_vector(y_a)
        else:
            sample_weight=None
        
        model.fit(X_a.loc[idx_a,:].values, y_a, sample_weight=sample_weight)
        
        y_hat_id = model.predict(X_id.loc[idx_id,:].values)
        acc_id = accuracy_score(y_id, y_hat_id)
        f1_id = f1_score(y_id, y_hat_id)
        f1_ood = None
        
        table1.columns[1]._cells[-1] = f'{acc_id:.03f}'
        table2.columns[0]._cells[-1] = f'{f1_id:.03f}'
        c.print(layout)
        
        df_result['acc'].loc[identifier,'ID'] = acc_id
        df_result['f-1'].loc[identifier,'ID'] = f1_id
        
        
        y_hat_b = model.predict(X_b.loc[idx_b,:].values)
        
        acc_ood = accuracy_score(y_b, y_hat_b)
        f1_ood = f1_score(y_b, y_hat_b)
        
        df_result['acc'].loc[identifier,'OOD'] = acc_ood
        df_result['f-1'].loc[identifier,'OOD'] = f1_ood
        
        table1.columns[2]._cells[-1] = f'{acc_ood:.03f}'
        table2.columns[1]._cells[-1] = f'{f1_ood:.03f}'
        c.print(layout)
        
        if f1_ood > 0:
            
            dbcp_a = model.feature_importances_by_(X_id.loc[idx_id,:].values)
            dbcp_b = model.feature_importances_by_(X_b.loc[idx_b,:].values)
            
            mssf_a = model.sufficiency_based_feature_importances(X_id.loc[idx_id,:].values)
            mssf_b = model.sufficiency_based_feature_importances(X_b.loc[idx_b,:].values)
            
            strategies = [dbcp_b, dbcp_b-dbcp_a, mssf_b, mssf_b-mssf_a]
            
            for i,s in enumerate(strategies):

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
                
                model_adapted_brs.fit(X_a.loc[idx_a,:].values, y_a, sample_weight=sample_weight)
                y_hat = model_adapted_brs.predict(X_b.loc[idx_b,:].values)
                acc_oodi = accuracy_score(y_b, y_hat)
                f1_oodi = f1_score(y_b, y_hat)

                table1.columns[i+3]._cells[-1] = f'{acc_oodi:.03f}'
                table2.columns[i+2]._cells[-1] = f'{f1_oodi:.03f}'
                c.print(layout)
                
                df_result['acc'].loc[identifier,cols[i+3]] = acc_oodi
                df_result['f-1'].loc[identifier,cols[i+3]] = f1_oodi
                
                
        table2.columns[6]._cells[-1] = f'{int(time.time() - time_)}'
        c.clear()
        c.print(layout)
        save_result(df_result)
except Exception as error:
    save_result(df_result)
    print(error)