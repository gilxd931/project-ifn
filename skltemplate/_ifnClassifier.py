"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import random
import operator
from ._ifn_network import IfnNetwork, AttributeNode, HiddenLayer
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances



from scipy import stats


def drop_records(X, index_atr, y, index_node):
    ans_x = []
    ans_y = []
    for i in range(len(y)):
        if X[i][index_atr] == index_node:
            ans_x.append(X[i])
            ans_y.append(y[i])
    return np.array(ans_x), np.array(ans_y)



class IfnClassifier():
    attributes_array=[]
    update_attributes_array=[]
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
        self.network = IfnNetwork()

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

        total_records = len(y)
        unique, counts = np.unique(np.array(y), return_counts=True)
        class_count = np.asarray((unique, counts)).T


        # create list of all attributes
        updated_attributes_array = list(range(0, len(X[0])))

        self.network.build_target_layer(np.unique(y))

        attributes_mi = {}
        unique_values_per_attribute = {}

        # get the attribute that holds the maximal mutual information
        for attribute in updated_attributes_array:
            attribute_data = []
            for record in X:
                attribute_data.append(record[attribute])
            unique_values_per_attribute[attribute] = np.unique(attribute_data)
            attributes_mi[attribute] = metrics.mutual_info_score(attribute_data, y)

        chosen_attribute = max(attributes_mi, key=attributes_mi.get)
        updated_attributes_array.remove(chosen_attribute)

        # create new hidden layer of the maximal mutual information attribute and set the layer nodes
        first_layer = HiddenLayer(chosen_attribute)
        self.network.root_node.first_layer = first_layer
        nodes_list = []
        for i in unique_values_per_attribute[chosen_attribute]:
            x_y_tuple = drop_records(X, chosen_attribute, y, i)
            nodes_list.append(AttributeNode(i, chosen_attribute, x_y_tuple[0], x_y_tuple[1]))
        first_layer.set_nodes(nodes_list)
        current_layer = first_layer

        print('nodes for layer ' + str(current_layer.index) + ' are: ')
        first_layer.print()

        while len(updated_attributes_array) > 0:
            # get the attribute that holds the maximal mutual information
            nodes_info_per_attribute = {}
            for node in current_layer.nodes:
                total_node_mi = 0
                for attribute in updated_attributes_array:
                    if attribute not in nodes_info_per_attribute:
                        nodes_info_per_attribute[attribute] = []
                    attribute_data = []
                    for record in node.partial_x:
                        attribute_data.append(record[attribute])
                    unique_values_per_attribute[attribute] = np.unique(attribute_data)
                    node_mi = metrics.mutual_info_score(attribute_data, node.partial_y)
                    total_node_mi += node_mi
                    statistic = 2 * np.log(2) * total_records * node_mi
                    critical = stats.chi2.ppf(self.alpha, 1)
                    if critical < statistic:
                        node_info_tuple = (node.index, node_mi)
                    else:
                        node_info_tuple = (node.index, 0)

                    nodes_info_per_attribute[attribute].append(node_info_tuple)

            max_node_mi = 0
            chosen_index = -1
            for attribute_index in nodes_info_per_attribute:
                node_mi = 0
                for node_info in nodes_info_per_attribute[attribute_index]:
                    node_mi += node_info[1]
                if node_mi > max_node_mi:
                    max_node_mi = node_mi
                    chosen_index = attribute_index

            chosen_attribute = chosen_index



            # stop building the network if all layer's nodes are terminal
            if chosen_attribute == -1:
                break

            # set terminal nodes
            for node_tuple in nodes_info_per_attribute[chosen_attribute]:
                if node_tuple[1] != 0:  # means chi2 test didnt pass
                    node = current_layer.get_node(node_tuple[0])
                    if node is not None:
                        node.set_terminal()
                        # add weight

            nodes_list = []
            for i in unique_values_per_attribute[chosen_attribute]:
                x_y_tuple = drop_records(current_layer.get_node(i).partial_x,
                                         chosen_attribute, current_layer.get_node(i).partial_y, i)
                nodes_list.append(AttributeNode(i, chosen_attribute, x_y_tuple[0], x_y_tuple[1]))
            updated_attributes_array.remove(chosen_attribute)
            new_layer = HiddenLayer(chosen_attribute)
            current_layer.next_layer = new_layer
            current_layer = new_layer

            current_layer.set_nodes(nodes_list)

            print('nodes for layer ' + str(current_layer.index) + ' are: ')
            current_layer.print()

        # that means we used all of the attributes so we have to set the last layer's nodes to be terminal
        if len(updated_attributes_array) == 0:
            for node in current_layer.nodes:
                node.set_terminal()
                node.print_info()
                # set node weight


        # `fit` should always return `self`
        return self





    def calIMPerNode(self, X, y):
        attributes_mi = {}

        for attribute in self.updated_attributes_array:
            attribute_data = []
            for record in X:
                attribute_data.append(record[attribute])
            attributes_mi[attribute] = metrics.adjusted_mutual_info_score(attribute_data, y)
        return attributes_mi



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
