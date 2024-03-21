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

c = Console(record=True, width=150)
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
    'college_scorecard': [
        'UGDS_WHITENH',
        'UGDS_API',
        'UGDS_AIANOld',
        'UGDS_BLACKNH',
        'UGDS_HISPOld',
        'AVGFACSAL',
        'PFTFAC',
        'TUITIONFEE_IN',
        'TUITIONFEE_OUT',
        'UGDS_NHPI',
        'UGDS_AIAN',
        'UGDS_ASIAN',
        'UGDS_HISP',
        'UGDS_2MOR',
        'UGDS_WHITE',
        'UGDS_BLACK',
        'PCTPELL',
        'ADM_RATE_ALL',
        'ADM_RATE',
        'veteran',
        'faminc',
        'md_faminc',
        'female',
        'agege24',
        'age_entry',
        'dependent',
        'loan_ever',
        'pell_ever',
        'age_entry_sq',
        'married',
        'first_gen',
        'unemp_rate',
        'pct_asian',
        'pct_hispanic',
        'poverty_rate',
        'median_hh_inc',
        'pct_white',
        'pct_black',
        'pct_ba',
        'pct_grad_prof',
        'pct_born_us',
        'TUITIONFEE_PROG',
        'DISTANCEONLY',
        'COSTT4_A',
        'SATMTMID',
        'SATVRMID',
        'ACTCMMID',
        'ACTENMID',
        'ACTMTMID',
        'COSTT4_P',
        'UG',
        'UG_NRA',
        'UG_WHITENH',
        'UG_UNKN',
        'UG_API',
        'UG_AIANOld',
        'UG_HISPOld',
        'UG_BLACKNH',
        'PPTUG_EF2',
        'HBCU',
        'LOCALE',
        'SATWRMID',
        'NPT4_PROG',
        'AccredAgency',
        'CCSIZSET',
        'ACTWRMID',
        'locale2'
    ],
    'diabetes_readmission': ['medical_specialty', 'payer_code', 'weight'],
    'physionet': [
        'DBP',
        'Unit2',
        'Unit1',
        'Temp',
        'Glucose',
        'Potassium',
        'Hct',
        'FiO2',
        'Hgb',
        'pH',
        'BUN',
        'WBC',
        'Magnesium',
        'Creatinine',
        'Platelets',
        'Calcium',
        'PaCO2',
        'BaseExcess',
        'Chloride',
        'HCO3',
        'Phosphate',
        'SaO2',
        'EtCO2',
        'PTT',
        'Lactate',
        'AST',
        'Alkalinephos',
        'Bilirubin_total',
        'TroponinI',
        'Fibrinogen',
        'Bilirubin_direct'
    ],
    'anes': ['VCF0428', 'VCF0224', 'VCF0218', 'VCF0429', 'VCF0901b'],
    'acsfoodstamps': ['ST'],
    'acsincome': ['ST']
}

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
        n_estimators=500, max_features=int(np.sqrt(X_a.shape[1])), 
        n_jobs=-1, random_state=2
    )
    
    sample_weight=None
    sample_weight=balanced_weight_vector(y_a)
    model.fit(X_a.loc[idx_a,:].values, y_a, sample_weight=sample_weight)
    
    y_hat_id = model.predict(X_id.loc[idx_id,:].values)
    acc_id = accuracy_score(y_id, y_hat_id)
    f1_id = f1_score(y_id, y_hat_id)
    f1_ood = None
    
    table1.columns[1]._cells[-1] = f'{acc_id:.03f}'
    table2.columns[0]._cells[-1] = f'{f1_id:.03f}'
    c.print(layout)
    
    y_hat_b = model.predict(X_b.loc[idx_b,:].values)
    
    acc_ood = accuracy_score(y_b, y_hat_b)
    f1_ood = f1_score(y_b, y_hat_b)
    
    table1.columns[2]._cells[-1] = f'{acc_ood:.03f}'
    table2.columns[1]._cells[-1] = f'{f1_ood:.03f}'
    c.print(layout)
    
    if f1_ood > 0:
        
        dbcp_a = model.feature_importances_by_(X_id.loc[idx_id,:].values)
        dbcp_b = model.feature_importances_by_(X_b.loc[idx_b,:].values)
        
        mssf_a = model.sufficiency_based_feature_importances(X_id.loc[idx_id,:].values)
        mssf_b = model.sufficiency_based_feature_importances(X_b.loc[idx_b,:].values)
        
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
            
            model_adapted_brs.fit(X_a.loc[idx_a,:].values, y_a, sample_weight=sample_weight)
            y_hat = model_adapted_brs.predict(X_b.loc[idx_b,:].values)
            acc_oodi = accuracy_score(y_b, y_hat)
            f1_oodi = f1_score(y_b, y_hat)

            table1.columns[i+3]._cells[-1] = f'{acc_oodi:.03f}'
            table2.columns[i+2]._cells[-1] = f'{f1_oodi:.03f}'
            c.print(layout)
            
            
    table2.columns[6]._cells[-1] = f'{int(time.time() - time_)}'
    c.clear()
    c.print(layout)
    c.save_html(f'results/rich_experiment.html')