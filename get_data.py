from tableshift import get_dataset

dset = get_dataset(name='college_scorecard', cache_dir = '../tableshift/tmp', use_cached = True)

print(len(dset.get_pandas('train')))