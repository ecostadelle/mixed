import time

from tableshift import get_dataset
from tableshift.configs.benchmark_configs import BENCHMARK_CONFIGS

import logging

LOG_LEVEL = logging.DEBUG

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


data = [
    # [ 'ASSISTments',             'assistments'             ], # ok 
    # [ 'Childhood Lead',          'nhanes_lead'             ], # 'Document does not match SAS Version 5 or 6 Transport (XPORT) format'
    # [ 'College Scorecard',       'college_scorecard'       ], # ok
    # [ 'Diabetes',                'brfss_diabetes'          ], # ok
    # [ 'FICO HELOC',              'heloc'                   ], # ok
    # [ 'Food Stamps',             'acsfoodstamps'           ], # ok
    # [ 'Hospital Readmission',    'diabetes_readmission'    ], # ok
    # [ 'Hypertension',            'brfss_blood_pressure'    ], # ok
    #[ 'ICU Length of Stay'       'mimic_extract_los_3'     ],    
    #[ 'ICU Mortality',           'mimic_extract_mort_hosp' ],        
    # [ 'Income',                  'acsincome'               ], # ok
    [ 'Public Health Insurance', 'acspubcov'               ], # Killed
    # [ 'Sepsis',                  'physionet'               ], # ok
    # [ 'Unemployment',            'acsunemployment'         ], # ok
    # [ 'Voting',                  'anes'                    ] # ok
    ]


for i,d in enumerate(data):
    time_ = time.time()
    
    
    expt_config = BENCHMARK_CONFIGS[d[1]]
    expt_config.preprocessor_config.categorical_features = 'passthrough'
    expt_config.preprocessor_config.numeric_features = 'passthrough'
    
    try:
        dset = get_dataset(
            name = d[1],
            preprocessor_config = expt_config.preprocessor_config,
            
        )
        dset.to_sharded()
        
        print(f'{d[0]:25}', end=' ')
        print('done', end=' ')
        print(f'{int(time.time() - time_):6d} s')
        
    except Exception as error:
        print(f'{d[0]:25}', end='')
        print(error)