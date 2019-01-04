import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import mne
from mne.io import concatenate_raws
import pandas as pd

from sklearn.model_selection import train_test_split

def classification_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row_data = list(filter(None, row_data))
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    return dataframe

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return sp.csr_matrix.todense(adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt))


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)
    # adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # return sparse_to_tuple(adj_normalized)
    return adj_normalized


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    for i in range(len(features)):
        rowsum = np.array(features[i].sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features[i] = r_mat_inv.dot(features[i])
    # return sparse_to_tuple(features)
    return features


def windows(data, size, step):
	start = 0
	while ((start+size) < data.shape[0]):
		yield int(start), int(start + size)
		start += step


def segment_signal_without_transition(data, window_size, step):
	segments = []
	for (start, end) in windows(data, window_size, step):
		if(len(data[start:end]) == window_size):
			segments = segments + [data[start:end]]
	return np.array(segments)


def segment_dataset(X, window_size, step):
	win_x = []
	for i in range(X.shape[0]):
		win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
	win_x = np.array(win_x)
	return win_x


def load_data(sub_list, event_codes, slice_attention, window_size = 320, step = 10, tmin = 1, tmax = 4.1, adj_type='vanilla'):
    # sub_list = list(range(1,88))+[90,91]+list(range(93,100))+list(range(101,110))
    physionet_paths = [mne.datasets.eegbci.load_data(sub_id,event_codes) for sub_id in sub_list]
    physionet_paths = np.concatenate(physionet_paths)
    parts = [mne.io.read_raw_edf(path, preload=True,stim_channel='auto') for path in physionet_paths]

    raw = concatenate_raws(parts)

    # add filter
    # raw.filter(4., 30., fir_design='firwin', skip_by_annotation='edge')
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014')
    
    epoched = mne.Epochs(raw, events, dict(left=2, right=3), tmin=1, tmax=4.1, proj=False, picks=picks, baseline=None, preload=True)
    
    X = (epoched.get_data() * 1e6).astype(np.float32)
    y = (epoched.events[:,2] - 2).astype(np.int64)
    if slice_attention == True:
        X = np.transpose(X, [0,2,1])
        X = segment_dataset(X, window_size, step)
        X = np.transpose(X, [0, 1, 3, 2])
        num_node = X.shape[2]
    else:
        num_node = X.shape[1]

    print("read data end...")

    adj = get_adj(num_node, adj_type)

    print("prepare adjacency matrix end...")
    # train validataion test split
    # features = preprocess_features(X)
    features_train, features_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 33, shuffle = True)
    y_train = np.array(pd.get_dummies(y_train))
    y_test = np.array(pd.get_dummies(y_test))
    print("split data end...")
    print("load data end")
    return adj, features_train, features_test, y_train, y_test


def get_adj(num_node, adj_type):
    '''
    channel seq: Fc5.,Fc3.,Fc1.,Fcz.,Fc2.,Fc4.,Fc6.,C5..,C3..,C1..,Cz..,C2..,C4..,C6..,Cp5.,Cp3.,Cp1.,Cpz.,Cp2.,Cp4.,Cp6.,Fp1.,Fpz.,Fp2.,Af7.,Af3.,Afz.,Af4.,Af8.,F7..,F5..,F3..,F1..,Fz..,F2..,F4..,F6..,F8..,Ft7.,Ft8.,T7..,T8..,T9..,T10.,Tp7.,Tp8.,P7..,P5..,P3..,P1..,Pz..,P2..,P4..,P6..,P8..,Po7.,Po3.,Poz.,Po4.,Po8.,O1..,Oz..,O2..,Iz..
				 0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27   28   29   30   31   32   33   34   35   36   37   38   39   40   41   42   43   44   45   46   47   48   49   50   51   52   53   54   55   56   57   58   59   60   61   62   63
    '''
    self_link = [(i,i) for i in range(num_node)]
    if (adj_type == 'vanilla'): # neighboring connection with up, down, left, right
        neighbor_link = [(1,2),(1,31),(1,39),(1,8),
                         (2,3),(2,32),(2,9),
                         (3,4),(3,33),(3,10),
                         (4,5),(4,34),(4,11),
                         (5,6),(5,35),(5,12),
                         (6,7),(6,36),(6,13),
                         (7,40),(7,37),(7,14),
                         (8,9),(8,15),(8,41),
                         (9,10),(9,16),
                         (10,11),(10,17),
                         (11,12),(11,18),
                         (12,13),(12,19),
                         (13,14),(13,20),
                         (14,21),(14,42),
                         (15,16),(15,45),(15,48),
                         (16,17),(16,49),
                         (17,18),(17,50),
                         (18,19),(18,51),
                         (19,20),(19,52),
                         (20,21),(20,53),
                         (21,46),(21,54),
                         (22,23),(22,26),
                         (23,24),(23,27),
                         (24,28),
                         (25,26),(25,32),
                         (26,37),(26,33),
                         (27,28),(27,34),
                         (28,29),(28,35),
                         (29,36),
                         (30,31),(30,39),
                         (31,32),
                         (33,34),
                         (34,35),
                         (35,36),
                         (36,37),
                         (37,38),
                         (38,40),
                         (39,41),
                         (40,42),
                         (41,43),(41,45),
                         (42,44),(42,46),
                         (45,47),
                         (46,55),
                         (47,48),
                         (48,49),
                         (49,50),(49,56),
                         (50,51),(50,57),
                         (51,52),(51,58),
                         (52,53),(52,59),
                         (53,54),(53,60),
                         (54,55),
                         (56,57),
                         (57,58),(57,61),
                         (58,59),(58,62),
                         (59,60),(59,63),
                         (61,62),
                         (62,63),(62,64)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_link]
        edge = neighbor_link
        # edge = self_link + neighbor_link
        # construct adjacency matrix
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[i, j] = 1
            A[j, i] = 1
        adj = normalize_adj(A)

    elif (adj_type == 'vanilla+'): # neighboring connections with allover
          neighbor_link = [(1, 2), (1,31),(1,39),(1,8),(1,30),(1,32),(1,9),(1,41),
                           (2, 3), (2,32),(2,9),(2,31),(2,33),(2,10),(2,8),
                           (3, 4), (3,33),(3,10),(3,32),(3,34),(3,11),(3,9),
                           (4, 5), (4,34),(4,11),(4,33),(4,45),(4,12),(4,10),
                           (5, 6), (5,35),(5,12),(5,34),(5,36),(5,13),(5,11),
                           (6, 7), (6,36),(6,13),(6,35),(6,37),(6,14),(6,12),
                           (7, 40),(7,37),(7,14),(7,36),(7,38),(7,42),(7,13),
                           (8, 9), (8,15),(8,41),(8,39),(8,16),(8,45),
                           (9, 10), (9,16),(9,15),(9,17),
                           (10, 11), (10,17),(10,16),(10,18),
                           (11, 12), (11,18),(11,17),(11,19),
                           (12, 13), (12,19),(12,18),(12,20),
                           (13, 14), (13,20),(13,19),(13,21),
                           (14, 21), (14,42),(14,40),(14,46),(14,20),
                           (15, 16), (15,45),(15,48),(15,41),(15,47),(15,49),
                           (16, 17), (16,49),(16,48),(16,50),
                           (17, 18), (17,50),(17,51),(17,49),
                           (18, 19), (18,51),(18,50),(18,52),
                           (19, 20), (19,52),(19,51),(19,53),
                           (20, 21), (20,53),(20,52),(20,54),
                           (21, 46), (21,54),(21,42),(21,53),(21,55),
                           (22, 23), (22,26),(22,25),(22,27),
                           (23, 24), (23,27),(23,26),(23,28),
                           (24, 28), (24,27),(24,29),
                           (25, 26), (25,32),(25,31),(25,33),
                           (26, 37), (26,33),(26,34),(26,32),
                           (27, 28), (27,34),(27,33),(27,35),
                           (28, 29), (28,35),(28,34),(28,36),
                           (29, 36), (29,35),(29,37),
                           (30, 31), (30,39),
                           (31, 32), (31,39),
                           (33, 34),
                           (34, 35),
                           (35, 36),
                           (36, 37),
                           (37, 38), (37,40),
                           (38, 40),
                           (39, 41), (39,43),
                           (40, 42), (40,44),
                           (41, 43), (41,45),
                           (42, 44), (42,46),
                           (43, 45),
                           (44, 46),
                           (45, 47), (45,48),
                           (46, 55), (46,54),
                           (47, 48),
                           (48, 49), (48,56),
                           (49, 50), (49,56),(49,57),
                           (50, 51), (50,57),(50,56),(50,58),
                           (51, 52), (51,58),(51,57),(51,59),
                           (52, 53), (52,59),(52,58),(52,60),
                           (53, 54), (53,60),(53,59),
                           (54, 55), (54,60),
                           (56, 57), (56,61),
                           (57, 58), (57,61),(57,62),
                           (58, 59), (58,62),(58,61),(58,63),
                           (59, 60), (59,63),(59,62),
                           (60, 63), 
                           (61, 62), (61,64),
                           (62, 63), (62,64),
                           (63, 64)]
          neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_link]
          edge = self_link + neighbor_link
          # edge = neighbor_link
          # construct adjacency matrix
          A = np.zeros((num_node, num_node))
          for i, j in edge:
              A[i, j] = 1
              A[j, i] = 1
          adj = normalize_adj(A)
    elif (adj_type == 'distance'):# distance connections with all other nodes
          A = np.zeros([num_node, num_node])
          loc = pd.read_csv("/home/dalinzhang/program/19AAAI/GHAM/EEG_adjacency_matrix.csv", index_col= False)
          x = np.array(loc['x(mm)'])
          y = np.array(loc['y(mm)'])
          z = np.array(loc['z(mm)'])
          for m in range(num_node):
            for n in range(num_node):
                if m == n:
                    A[m,n] = 1
                else:
                    A[m,n]=np.power(np.power((x[m]-x[n]),2)+np.power((y[m]-y[n]),2)+np.power((z[m]-z[n]),2),-0.5)
          adj = normalize_adj(A)

    elif (adj_type == 'distance+'):# distance connections with all other nodes all without self
          A = np.zeros([num_node, num_node])
          loc = pd.read_csv("/home/dalinzhang/program/19AAAI/GHAM/EEG_adjacency_matrix.csv", index_col= False)
          x = np.array(loc['x(mm)'])
          y = np.array(loc['y(mm)'])
          z = np.array(loc['z(mm)'])
          for m in range(num_node):
            for n in range(num_node):
                if m == n:
                    A[m,n] = 0
                else:
                    A[m,n]=np.power(np.power((x[m]-x[n]),2)+np.power((y[m]-y[n]),2)+np.power((z[m]-z[n]),2),-0.5)
          adj = normalize_adj(A)

    elif (adj_type == 'distance++'): # thershold distance without self 
        A = np.zeros([num_node, num_node])
        loc = pd.read_csv("/home/dalinzhang/program/19AAAI/GHAM/EEG_adjacency_matrix.csv", index_col=False)
        x = np.array(loc['x(mm)'])
        y = np.array(loc['y(mm)'])
        z = np.array(loc['z(mm)'])
        for m in range(num_node):
            for n in range(num_node):
                if m == n:
                    A[m, n] = 0
                else:
                    A[m, n] = np.power(
                        np.power((x[m] - x[n]), 2) + np.power((y[m] - y[n]), 2) + np.power((z[m] - z[n]), 2), -0.5)
        x_loc = np.expand_dims(np.where(A<np.mean(A))[0], axis=1)
        y_loc = np.expand_dims(np.where(A<np.mean(A))[1], axis=1)
        loc = np.append(x_loc,y_loc, axis=1)
        for i,j in loc:
            A[i, j] = 0
        adj = normalize_adj(A)

    elif (adj_type == 'distance+++'): # threshold distance optimal self
        A = np.zeros([num_node, num_node])
        loc = pd.read_csv("/home/dalinzhang/program/19AAAI/GHAM/EEG_adjacency_matrix.csv", index_col=False)
        x = np.array(loc['x(mm)'])
        y = np.array(loc['y(mm)'])
        z = np.array(loc['z(mm)'])
        for m in range(num_node):
            for n in range(num_node):
                if m != n:
                    A[m, n] = np.power(
                        np.power((x[m] - x[n]), 2) + np.power((y[m] - y[n]), 2) + np.power((z[m] - z[n]), 2), -0.5)
        x_loc = np.expand_dims(np.where(A<np.mean(A))[0], axis=1)
        y_loc = np.expand_dims(np.where(A<np.mean(A))[1], axis=1)
        loc = np.append(x_loc,y_loc, axis=1)
        for i,j in loc:
            A[i, j] = 0
        for k in range(num_node):
            A[k,k]=np.mean(A[k])
        adj = normalize_adj(A)
    return adj



def construct_feed_dict(features, adj, labels, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
