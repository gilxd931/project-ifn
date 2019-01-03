"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from ._ifn_network import IfnNetwork, AttributeNode
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.metrics import mutual_info_score

class IfnClassifier():
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, alpha = 0.99):
        self.alpha = alpha
        network = IfnNetwork()
        # network.add_root(3)
        # network.print_root()
        # a = AttributeNode(0)
        # a.add_next(2)
        # a.add_next(1)
        # a.add_next(3)
        # a.add_next(4)
        # a.add_next(4)
        # a.add_next(4)
        # a.print_next()

    def fit(self, X, y):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True)

        if not np.array_equal(y, y.astype(bool)):
            raise ValueError("Found array y that is not binary")

        for row in X:
            if not np.array_equal(row, row.astype(bool)):
                raise ValueError("Found array y that is not binary")

        # create list of all attributes
        attributes_array = list(range(0, len(X[0])))

        max_MI =[]

        chosen_attribute =-1
        for attribute in attributes_array:
            attribute_data = []
            for record in X:
                attribute_data.append(record[attribute])
            max_MI.append(metrics.adjusted_mutual_info_score(attribute_data, y))


        i=0
        temp_max_MI=0;
        for row in max_MI:
            if(row>temp_max_MI):
                temp_max_MI=row
                chosen_attribute=i
            i=i+1

        print(chosen_attribute)
        self.is_fitted_ = True

        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return np.ones(X.shape[0], dtype=np.int64)
