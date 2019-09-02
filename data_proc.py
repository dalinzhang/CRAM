import os.path
import scipy.io as sio
import pickle
import numpy as np

from collections import OrderedDict
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.datautil.splitters import concatenate_sets

#######################
# reference: github: TNTLFreiburg: braindecode/examples/bcic_iv_2a.py
#######################


def data_gen(subject, high_cut_hz=38, low_cut_hz=0):
	data_sub = {}
	for i in range(len(subject)):
		subject_id = subject[i]
		data_folder = '/home/dadafly/program/bci_data/data_folder'
		ival = [-500, 4000]
		factor_new = 1e-3
		init_block_size = 1000
		
		train_filename = 'A{:02d}T.gdf'.format(subject_id)
		test_filename = 'A{:02d}E.gdf'.format(subject_id)
		train_filepath = os.path.join(data_folder, train_filename)
		test_filepath = os.path.join(data_folder, test_filename)
		train_label_filepath = train_filepath.replace('.gdf', '.mat')
		test_label_filepath = test_filepath.replace('.gdf', '.mat')
		
		train_loader = BCICompetition4Set2A(
		    train_filepath, labels_filename=train_label_filepath)
		test_loader = BCICompetition4Set2A(
		    test_filepath, labels_filename=test_label_filepath)
		
		train_cnt = train_loader.load()
		test_cnt = test_loader.load()
		
		
		train_loader = BCICompetition4Set2A(
		    train_filepath, labels_filename=train_label_filepath)
		test_loader = BCICompetition4Set2A(
		    test_filepath, labels_filename=test_label_filepath)
		
		train_cnt = train_loader.load()
		test_cnt = test_loader.load()
		
		# train set process
		train_cnt = train_cnt.drop_channels(['STI 014', 'EOG-left',
		                                     'EOG-central', 'EOG-right'])
		assert len(train_cnt.ch_names) == 22
		
		train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
		train_cnt = mne_apply(
		    lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, train_cnt.info['sfreq'],
		                           filt_order=3, axis=1), train_cnt)
		
		train_cnt = mne_apply(
		    lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
		                                              init_block_size=init_block_size,
		                                              eps=1e-4).T, train_cnt)
		
		# test set process
		test_cnt = test_cnt.drop_channels(['STI 014', 'EOG-left',
		                                   'EOG-central', 'EOG-right'])
		assert len(test_cnt.ch_names) == 22
		test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
		test_cnt = mne_apply(
		    lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, test_cnt.info['sfreq'],
		                           filt_order=3, axis=1), test_cnt)
		test_cnt = mne_apply(
		    lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
		                                              init_block_size=init_block_size,
		                                              eps=1e-4).T, test_cnt)
		
		marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
		                          ('Foot', [3]), ('Tongue', [4])])
		
		train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
		test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)
		
		data_sub[str(subject_id)] = concatenate_sets([train_set, test_set])
		if i == 0:
			dataset = data_sub[str(subject_id)]
		else:
			dataset = concatenate_sets([dataset, data_sub[str(subject_id)]])
	assert len(data_sub) == len(subject)

	return dataset


if __name__ == '__main__':
	for j in range(1,10):
		train_subject = [k for k in range(1,10) if k != j]
		test_subject = [j]
		train_dataset = data_gen(train_subject, high_cut_hz=125, low_cut_hz=0)
		test_dataset = data_gen(test_subject, high_cut_hz=125, low_cut_hz=0)

		train_X = train_dataset.X
		train_y = train_dataset.y
		test_X = test_dataset.X
		test_y = test_dataset.y

		idx = list(range(len(train_y)))
		np.random.shuffle(idx)
		train_X = train_X[idx]
		train_y = train_y[idx]
		sio.savemat('/home/dadafly/program/bci_data/data_folder/cross_sub/cross_subject_data_'+str(j)+'.mat', {"train_x": train_X, "train_y": train_y, "test_x": test_X, "test_y": test_y})


