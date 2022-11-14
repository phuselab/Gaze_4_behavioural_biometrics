import numpy as np
import scipy.io as sio
from os.path import join, isdir
from os import makedirs

def save_OU_features(values, dataset, filename, type='unknown', method='unknown'):
	# Define the saving directory
	fix_len = np.shape(values)[0]
	fixations_dict = {'id': np.reshape(values[:, 0], (fix_len, 1)),
					'B00': np.reshape(values[:, 1], (fix_len, 1)),
					'B01': np.reshape(values[:, 2], (fix_len, 1)),
					'B11': np.reshape(values[:, 3], (fix_len, 1)),
					'S00': np.reshape(values[:, 4], (fix_len, 1)),
					'S01': np.reshape(values[:, 5], (fix_len, 1)),
					'S11': np.reshape(values[:, 6], (fix_len, 1))}
	# pl.plot_values(fixations_dict)
	dir_name = 'features/'+dataset+'_'+type+'_'+method
	if not isdir(dir_name):
		makedirs(dir_name)
	sio.savemat(join(dir_name, filename), fixations_dict)


def save_all_features(values, dataset, filename, type='unknown', method='unknown'):
	# Define the saving directory
	fix_len = np.shape(values)[0]
	fixations_dict = {'id': np.reshape(values[:, 0], (fix_len, 1)),
					'B00': np.reshape(values[:, 1], (fix_len, 1)),
					'B01': np.reshape(values[:, 2], (fix_len, 1)),
					'B11': np.reshape(values[:, 3], (fix_len, 1)),
					'S00': np.reshape(values[:, 4], (fix_len, 1)),
					'S01': np.reshape(values[:, 5], (fix_len, 1)),
					'S11': np.reshape(values[:, 6], (fix_len, 1)),
					'B00_s': np.reshape(values[:, 7], (fix_len, 1)),
					'B01_s': np.reshape(values[:, 8], (fix_len, 1)),
					'B11_s': np.reshape(values[:, 9], (fix_len, 1)),
					'S00_s': np.reshape(values[:, 10], (fix_len, 1)),
					'S01_s': np.reshape(values[:, 11], (fix_len, 1)),
					'S11_s': np.reshape(values[:, 12], (fix_len, 1)),
					'mu_ig': np.reshape(values[:, 13], (fix_len, 1)),
					'loc_ig': np.reshape(values[:, 14], (fix_len, 1)),
					'scale_ig': np.reshape(values[:, 15], (fix_len, 1)),
					'alpha': np.reshape(values[:, 16], (fix_len, 1)),
					'beta': np.reshape(values[:, 17], (fix_len, 1)),
					'kappa': np.reshape(values[:, 18], (fix_len, 1)),
					'loc_vm': np.reshape(values[:, 19], (fix_len, 1)),
					'scale_vm': np.reshape(values[:, 20], (fix_len, 1))}
	# pl.plot_values(fixations_dict)
	dir_name = 'new_features/'+dataset+'_'+type+'_'+method
	if not isdir(dir_name):
		makedirs(dir_name)
	sio.savemat(join(dir_name, filename), fixations_dict)

def save_fix_features(values, dataset, filename, type='unknown', method='unknown'):
	# Define the saving directory
	fix_len = np.shape(values)[0]
	fixations_dict = {'id': np.reshape(values[:, 0], (fix_len, 1)),
					'B00': np.reshape(values[:, 1], (fix_len, 1)),
					'B01': np.reshape(values[:, 2], (fix_len, 1)),
					'B11': np.reshape(values[:, 3], (fix_len, 1)),
					'S00': np.reshape(values[:, 4], (fix_len, 1)),
					'S01': np.reshape(values[:, 5], (fix_len, 1)),
					'S11': np.reshape(values[:, 6], (fix_len, 1)),
					'mu_ig': np.reshape(values[:, 7], (fix_len, 1)),
					'loc_ig': np.reshape(values[:, 8], (fix_len, 1)),
					'scale_ig': np.reshape(values[:, 9], (fix_len, 1))}
	# pl.plot_values(fixations_dict)
	dir_name = 'features/'+dataset+'_'+type+'_'+method
	if not isdir(dir_name):
		makedirs(dir_name)
	sio.savemat(join(dir_name, filename), fixations_dict)



def save_event_features(data, dataset, filename, type='unknown', method='unknown', dset='train'):
	dir_name = 'new_features/'+dataset+'_'+type+'_'+method + '/' + dset
	if not isdir(dir_name):
		makedirs(dir_name)
	np.save(join(dir_name, filename), data)