import numpy as np
import copy
import time
from sklearn.neighbors import NearestNeighbors
import numba as nb
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Activation, Input, concatenate
from keras import backend as K

def get_NearestNeighbors(X, k_neighbors):
	# Returns the distances and indices of the K nearest neighbors exluding the data itself
	neighbors = NearestNeighbors(n_neighbors = k_neighbors + 1, algorithm = 'ball_tree', 
								n_jobs = -1).fit(X)
	distances, indices = neighbors.kneighbors(X)
	distances = distances[:, 1:]
	indices = indices[:, 1:]

	return distances, indices

def affinity_computation(X, k_neighbors):
	n_samples = X.shape[0]

	neighbors = NearestNeighbors(n_neighbors = k_neighbors + 1, algorithm = 'ball_tree', n_jobs = -1).fit(X)
	distances, indices = neighbors.kneighbors(X)

	distancesSq = distances ** 2
	sigmaSq = np.sum(distancesSq) / ((k_neighbors) * n_samples) 

	W = np.zeros((n_samples, n_samples))

	for i in range(n_samples):
		for j in range(1, k_neighbors + 1):
			ind = indices[i, j]
			W[i, ind] = np.exp(-distancesSq[i, j] / sigmaSq)

	return distances, indices, W

def compute_cluster_affinity(W, labels_tab):
	n_clusters = len(labels_tab)

	A_unsym = np.zeros([n_clusters, n_clusters])
	A_sym = np.zeros([n_clusters, n_clusters])

	for i in range(n_clusters):
		for j in range(i + 1, n_clusters):
			A_unsym[i, j] = np.dot(np.sum(W[labels_tab[i],:][:, labels_tab[j]], 0), 
								np.sum(W[labels_tab[j],:][:, labels_tab[i]], 1))
			A_unsym[j, i] = np.dot(np.sum(W[labels_tab[j],:][:, labels_tab[i]], 0), 
								np.sum(W[labels_tab[i],:][:, labels_tab[j]], 1))

			A_sym[i, j] = (A_unsym[i, j] / (len(labels_tab[i])**2)) + (A_unsym[j, i] / len(labels_tab[j])**2)
			A_sym[j, i] = A_sym[i, j]
	
	return A_unsym, A_sym

def initialize_clusters(indices, n_samples):
	cluster_status = -np.ones(n_samples, dtype = np.int32)
	count = 0
	for i_samples in range(n_samples):
		cur_idx = i_samples
		cur_cluster_idxs = []
		while cluster_status[cur_idx] == -1:
			cur_cluster_idxs.append(cur_idx)
			neighbor = indices[cur_idx, 0]
			cluster_status[cur_idx] = -2
			cur_idx = neighbor
			if len(cur_cluster_idxs) > 50:
				break
		if cluster_status[cur_idx] < 0:
			cluster_status[cur_idx] = count
			count += 1
		for j in range(len(cur_cluster_idxs)):
			cluster_status[cur_cluster_idxs[j]] = cluster_status[cur_idx]

	label_indices = []
	labels = np.zeros(n_samples, dtype = np.int32)
	for _ in range(count):
		label_indices.append([])
	for i_samples in range(n_samples):
		label_indices[cluster_status[i_samples]].append(i_samples)
		labels[i_samples] = cluster_status[i_samples]

	return label_indices, labels

def search_clusters(A_s, K_c):
	indices_sorted = np.argsort(-A_s)
	A_sorted = -np.sort(-A_s)

	if A_sorted.shape[0] < 100:
		aff_c = A_sorted[:, 0]
	else:
		aff_c = A_sorted[:, 0] + (K_c * A_sorted[:, 0] - np.sum(A_sorted[:, 1:K_c], axis = 1)) / (K_c - 1)
	
	idx_a = np.argmax(aff_c)
	idx_b = indices_sorted[idx_a, 0]

	return idx_a, idx_b

def reorganize_clusters(A_us, Y_tab, idx_a, idx_b):

	n_a, n_b = len(Y_tab[idx_a]), len(Y_tab[idx_b])

	A_us_new = copy.deepcopy(A_us)
	Y_tab_new = []
	for i in range(A_us.shape[0]):
		if i != idx_a:
			A_us_new[idx_a, i] += A_us[idx_b, i]
			A_us_new[i, idx_a] = (n_a**2) * A_us[i, idx_a] / (n_a + n_b)**2 + (n_b**2) * A_us[i, idx_b] / (n_a + n_b)**2

	A_us_new = np.r_[A_us_new[:idx_b, :], A_us_new[idx_b + 1:, :]]
	A_us_new = np.c_[A_us_new[:, :idx_b], A_us_new[:, idx_b + 1:]]


	for i in range(len(Y_tab)):
		if i == idx_a:
			Y_tab_new.append(Y_tab[i] + Y_tab[idx_b])
		elif i != idx_b:
			Y_tab_new.append(Y_tab[i])
	#print(len(Y_tab), len(Y_tab_new))
	# Y_tab[idx_a] += Y_tab[idx_b]
	# Y_tab = Y_tab[:idx_b] + Y_tab[idx_b + 1:]

	A_s_new = np.zeros(A_us_new.shape)
	for i in range(A_s_new.shape[0]):
		for j in range(i + 1, A_s_new.shape[1]):
			A_s_new[i, j] = (A_us_new[i, j] / (len(Y_tab[i])**2)) + (A_us_new[j, i] / len(Y_tab[j])**2)
			A_s_new[j, i] = A_s_new[i, j]

	return A_us_new, A_s_new, Y_tab_new

def run_agglomerative_clustering(A_us, A_s, Y_tab, N_iter, K_c):
	t = 0
	while t < N_iter:
		idx_a, idx_b = search_clusters(A_s, K_c)
		# print("found clusters to merge: time = %f" % (t2 - t1))
		A_us, A_s, Y_tab = reorganize_clusters(A_us, Y_tab, idx_a, idx_b)
		# print("reorganized clusters: time = %f" % (t3 - t2))
		t += 1
	return Y_tab

def make_triplets(labels, n_clusters, n_neg_perPosPair):
	a_ind, p_ind, n_ind = [], [], []

	labels_perclass = [[] for _ in range(n_clusters)]
	for i in range(len(labels)):
		labels_perclass[labels[i]].append(i)
	# labels_perclass = np.array(labels_perclass)

	# print(labels[:100])
	# print(len(labels_perclass))
	# print(labels_perclass[0])
	# print(labels_perclass[1])

	for i in range(len(labels_perclass)):
		for j in range(len(labels_perclass[i])):
			for k in range(j + 1, len(labels_perclass[i])):
				neg_clusters = np.r_[np.arange(len(labels_perclass))[:i], np.arange(len(labels_perclass))[i+1:]]
				list_possible_neg = labels_perclass[neg_clusters[0]]
				for l in neg_clusters[1:]:
					list_possible_neg = list_possible_neg + labels_perclass[l]
				list_possible_neg = np.ndarray.flatten(np.array(list_possible_neg))
				n_ind_list = np.random.choice(list_possible_neg, n_neg_perPosPair, replace = False)
				for m in n_ind_list:
					a_ind.append(labels_perclass[i][j])
					p_ind.append(labels_perclass[i][k])
					n_ind.append(m)

	a_ind = np.array(a_ind)
	p_ind = np.array(p_ind)
	n_ind = np.array(n_ind)

	return a_ind, p_ind, n_ind

def triplet_loss(y_true, y_pred):

	out_shape = int(int(y_pred.shape[1]) / 3)
	anchor = y_pred[:, :out_shape]
	positive = y_pred[:, out_shape:2 * out_shape]
	negative = y_pred[:, 2 * out_shape:]

	# distance between the anchor and the positive
	pos_dist = K.sum(K.square(anchor - positive), axis=1)

	# distance between the anchor and the negative
	neg_dist = K.sum(K.square(anchor - negative), axis=1)

	# compute loss
	basic_loss = 2 * pos_dist - neg_dist + 0.5
	loss = K.maximum(basic_loss, 0.0)
 
	return loss

def convert_labels_tab(labels_tab):
	n_samples = 0
	for i in range(len(labels_tab)):
		n_samples += len(labels_tab[i])
	labels = np.zeros((n_samples,), dtype = np.int32)

	for i in range(len(labels_tab)):
		for j in range(len(labels_tab[i])):
			labels[labels_tab[i][j]] = i

	return labels