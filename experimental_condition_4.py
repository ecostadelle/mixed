import pandas as pd
import numpy as np
import warnings
import time

# rich
from rich.console import Console
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


from util import save_result, create_table


c = Console(width=150)

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

df_result = {
    'acc': create_table(np.array(data)[:,1]),
    'f-1': create_table(np.array(data)[:,1])}

df_result['acc'].iloc[:,0] = np.array(data)[:,0]
df_result['f-1'].iloc[:,0] = np.array(data)[:,0]

cols = df_result['acc'].columns


for dataset,identifier in data:
    try:
        time_ = time.time()
        
        
        dset = get_dataset(
            name=identifier,
            initialize_data=False,
            use_cached=True
        )

        X_a, y_a, _, _ = dset.get_pandas('train')
        X_id, y_id, _, _ = dset.get_pandas('id_test')
        X_b, y_b, _, _ = dset.get_pandas('ood_test')

        drop=True
        balanced = True
        sqrt_=int(np.sqrt(X_a.shape[1]))


        X_a.reset_index(drop=True, inplace=True)
        X_id.reset_index(drop=True, inplace=True)
        X_b.reset_index(drop=True, inplace=True)

        y_a.reset_index(drop=True, inplace=True)
        y_id.reset_index(drop=True, inplace=True)
        y_b.reset_index(drop=True, inplace=True)

        if drop and identifier in drop_columns.keys():
            X_a.drop(drop_columns[identifier], inplace=True, axis=1, errors='ignore')
            X_id.drop(drop_columns[identifier], inplace=True, axis=1, errors='ignore')
            X_b.drop(drop_columns[identifier], inplace=True, axis=1, errors='ignore')

        cat_dict = {
            f.name: f.value_mapping
            for f in get_task_config(identifier).feature_list.features 
            if f.kind.__name__ == 'CategoricalDtype' 
            and not f.is_target 
            and f.name in X_a.columns
        }

        cat_hidden = {
            f.name
            for f in get_task_config(identifier).feature_list.features 
            if f.kind.__name__ != 'CategoricalDtype' 
            and not f.is_target 
            and f.value_mapping 
            and f.name in X_a.columns
        }

        cat_feats = set(cat_dict.keys())
        obj_feats = set(X_a.columns[X_a.dtypes == 'object']) - cat_feats
        num_feats = set(X_a.columns) - obj_feats - cat_feats - cat_hidden

        for feat in num_feats:
            try:
                out,bins = pd.qcut(X_a[feat],5,retbins=True,duplicates='drop')
                X_a[feat] = out
                X_id[feat] = pd.cut(X_id[feat],bins)
                X_b[feat] = pd.cut(X_b[feat],bins)
            except:
                print(feat)

        for feat in cat_feats:
            if cat_dict[feat]:
                cat_type = pd.CategoricalDtype(categories={v: float(k) for k,v in cat_dict[feat].items()})
            else:
                cat_type = X_a[feat].astype(str).astype('category').dtype
            X_a[feat] = X_a[feat].astype(cat_type)
            X_id[feat] = X_id[feat].astype(cat_type)
            X_b[feat] = X_b[feat].astype(cat_type)

        # Verify if all features are categories
        verify = X_a.columns[X_a.dtypes != 'category']
        if len(verify) > 0:
            for feat in verify:
                try:
                    out,bins = pd.qcut(X_a[feat],5,retbins=True,duplicates='drop')
                    X_a[feat] = out
                    X_id[feat] = pd.cut(X_id[feat],bins)
                    X_b[feat] = pd.cut(X_b[feat],bins)
                except:
                    print(feat)

        X_a = X_a.apply(lambda Xj: Xj.cat.codes)
        X_id = X_id.apply(lambda Xj: Xj.cat.codes)
        X_b = X_b.apply(lambda Xj: Xj.cat.codes)

        model = UDBaggingClassifier(
            estimator=InterpretableCategoricalNB(
                min_categories=2, 
                alpha=1e-10, 
                force_alpha=True
            ),
            n_estimators=500, max_features=sqrt_, 
            n_jobs=-1, random_state=2
        )

        if balanced:
            sample_weight=balanced_weight_vector(y_a)
        else:
            sample_weight=None

        model.fit(X_a.values, y_a.values, sample_weight=sample_weight)

        y_hat_id = model.predict(X_id.values)
        acc_id = accuracy_score(y_id.values, y_hat_id)
        f1_id = f1_score(y_id.values, y_hat_id)
        f1_ood = None

        df_result['acc'].loc[identifier,cols[1]] = acc_id
        df_result['f-1'].loc[identifier,cols[1]] = f1_id

        y_hat_b = model.predict(X_b.values)

        acc_ood = accuracy_score(y_b.values, y_hat_b)
        f1_ood = f1_score(y_b.values, y_hat_b)

        df_result['acc'].loc[identifier,cols[2]] = acc_ood
        df_result['f-1'].loc[identifier,cols[2]] = f1_ood

        if f1_ood > 0:
            
            dbcp_a = model._dbcp(X_id.values)
            dbcp_b = model._dbcp(X_b.values)
            
            mssf_a = model.sufficiency_based_feature_importances(X_id.values)
            mssf_b = model.sufficiency_based_feature_importances(X_b.values)
            
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
                    n_estimators=500, max_features=sqrt_, 
                    n_jobs=-1, random_state=2, 
                    biased_subspaces=True, feature_bias=p
                )
                
                model_adapted_brs.fit(X_a.values, y_a, sample_weight=sample_weight)
                y_hat = model_adapted_brs.predict(X_b.values)
                acc_oodi = accuracy_score(y_b.values, y_hat)
                f1_oodi = f1_score(y_b.values, y_hat)
                
                df_result['acc'].loc[identifier,cols[i+3]] = acc_oodi
                df_result['f-1'].loc[identifier,cols[i+3]] = f1_oodi
        
        c.print(f'{identifier}: {int(time.time() - time_)}')

        save_result(df_result)
        c.print(df_result['f-1'])
    except Exception as error:
        save_result(df_result)
        c.print(df_result['f-1'])
        c.print(error)
        pass
