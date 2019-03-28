import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                dists[i_test, i_train] = np.sum(abs(X[i_test] - self.train_X[i_train]))
        
        return dists

    def compute_distances_one_loop(self, X):
        '''
        Computes distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            # TODO: Fill the whole row of dists[i_test]
            # without additional loops
            dists[i_test] = np.sum(abs(self.train_X - X[i_test]), axis=1)
        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes distance from every sample of X to every training sample
        Fully vectorizes the calculations

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        # TODO: Implement computing all distances with no loops!
        dists = abs(X[:,None] - self.train_X).sum(axis=2)

        
        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        dists_sorted_list = []
        k_labels = []
        for i in range(num_test):
            dists_dict = {}
            for index, distanse in enumerate(dists[i]):
                dists_dict[index] = distanse
            dists_sorted=sorted(dists_dict.items(), key=lambda value: value[1])[:self.k]
            dists_sorted_list.append([i[0] for i in dists_sorted])
        k_labels = ([self.train_y[i].astype(int) for i in dists_sorted_list])
        pred = [sum(i) > self.k // 2 for i in k_labels ]
        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        dists_sorted_list = []
        for n in range(num_test):

            dists_dict = {}
            for index, distanse in enumerate(dists[n]):
                dists_dict[index] = distanse
            dists_sorted=sorted(dists_dict.items(), key=lambda value: value[1])[:self.k]
            dists_sorted_list.append([i[0] for i in dists_sorted])
            
            for index_naberhoods in dists_sorted_list:
                labels_naberhood = {}
                for i in index_naberhoods:
                   
                    if self.train_y[i] in labels_naberhood:
                        labels_naberhood[self.train_y[i]] += 1
                    else:
                        labels_naberhood[self.train_y[i]] = 1
                labels_naberhood = sorted(labels_naberhood.items(), key=lambda value: value[0])[:self.k]
                predict = labels_naberhood[0][0]
            pred[n] = predict
        return pred
