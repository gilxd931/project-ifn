"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from ._ifn_network import IfnNetwork, AttributeNode, HiddenLayer
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy import stats
import math
import collections


def calc_MI(x, y, total_records):
    partial_records = len(y)
    unique, counts = np.unique(np.array(y), return_counts=True)
    class_count = np.asarray((unique, counts)).T
    unique, counts = np.unique(np.array(x), return_counts=True)
    data_count = np.asarray((unique, counts)).T

    data_dic = collections.defaultdict(int)

    for i in range(len(y)):
        data_class_tuple = x[i], y[i]
        data_dic[data_class_tuple] = data_dic[data_class_tuple] + 1

    total_mi = 0
    for key, value in data_dic.items():

        curr_class_count = None
        for c_count in class_count:
            if c_count[0] == key[1]:
                curr_class_count = c_count[1]

        curr_data_count = None
        for d_count in data_count:
            if d_count[0] == key[0]:
                curr_data_count = d_count[1]

        mutual_information = (value / total_records) * math.log((value / partial_records) / ((curr_data_count / partial_records) *  (curr_class_count / partial_records)), 2)

        total_mi += mutual_information
    return total_mi


def drop_records(X, atr_index, y, node_index):
    new_x = []
    new_y = []
    for i in range(len(y)):
        if X[i][atr_index] == node_index:
            new_x.append(X[i])
            new_y.append(y[i])
    return np.array(new_x), np.array(new_y)



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

    def __init__(self, alpha = 0.999):
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

        num_of_classes = len(np.unique(y))
        self.network.build_target_layer(np.unique(y))

        attributes_mi = {}
        unique_values_per_attribute = {}
        curr_node_index = 1

        # get the attribute that holds the maximal mutual information
        for attribute in updated_attributes_array:
            attribute_data = []
            for record in X:
                attribute_data.append(record[attribute])

            unique_values_per_attribute[attribute] = np.unique(attribute_data)
            mutual_info_score = calc_MI(attribute_data, y, total_records)
            statistic = 2 * np.log(2) * total_records * mutual_info_score
            critical = stats.chi2.ppf(self.alpha, ((num_of_classes - 1) *
                                                   ((len(unique_values_per_attribute[attribute])) - 1)
                                                   ))
            if critical < statistic:
                attributes_mi[attribute] = mutual_info_score
            else:
                attributes_mi[attribute] = 0

        chosen_attribute = max(attributes_mi, key=attributes_mi.get)
        updated_attributes_array.remove(chosen_attribute)

        # create new hidden layer of the maximal mutual information attribute and set the layer nodes
        first_layer = HiddenLayer(chosen_attribute)
        self.network.root_node.first_layer = first_layer
        nodes_list = []
        for i in unique_values_per_attribute[chosen_attribute]:
            x_y_tuple = drop_records(X, chosen_attribute, y, i)
            nodes_list.append(AttributeNode(curr_node_index, 0, chosen_attribute, x_y_tuple[0], x_y_tuple[1]))
            curr_node_index += 1
        first_layer.set_nodes(nodes_list)
        current_layer = first_layer

        print('nodes for layer ' + str(current_layer.index) + ' are: ')
        first_layer.print()

        while len(updated_attributes_array) > 0:
            # get the attribute that holds the maximal mutual information
            nodes_info_per_attribute = {}
            for node in current_layer.nodes:
                for attribute in updated_attributes_array:
                    if attribute not in nodes_info_per_attribute:
                        nodes_info_per_attribute[attribute] = []
                    attribute_data = []
                    for record in node.partial_x:
                        attribute_data.append(record[attribute])
                    unique_values_per_attribute[attribute] = np.unique(attribute_data)
                    node_mi = calc_MI(attribute_data, node.partial_y, total_records)
                    statistic = 2 * np.log(2) * total_records * node_mi
                    critical = stats.chi2.ppf(self.alpha, ((num_of_classes - 1) *
                                                           ((len(unique_values_per_attribute[attribute])) - 1)
                                                           ))
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
                if node_tuple[1] == 0:  # means chi2 test didnt pass
                    node = current_layer.get_node(node_tuple[0])
                    if node is not None:
                        node.set_terminal()
                        # add weight

            for curr_layer_node in current_layer.nodes:
                if not curr_layer_node.is_terminal:
                    nodes_list = []
                    for i in unique_values_per_attribute[chosen_attribute]:
                        x_y_tuple = drop_records(curr_layer_node.partial_x,
                                                 chosen_attribute, curr_layer_node.partial_y, i)
                        nodes_list.append(AttributeNode(curr_node_index, curr_layer_node.index,
                                                        chosen_attribute, x_y_tuple[0], x_y_tuple[1]))
                        curr_node_index += 1

            updated_attributes_array.remove(chosen_attribute)
            new_layer = HiddenLayer(chosen_attribute)
            current_layer.next_layer = new_layer
            current_layer = new_layer

            current_layer.set_nodes(nodes_list)

            print('nodes for layer ' + str(current_layer.index) + ' are: ')
            current_layer.print()

        # that means we used all of the attributes so we have to set the last layer's nodes to be terminal
        # or all nodes in current layer are terminal
        if len(updated_attributes_array) == 0 or chosen_attribute == -1:
            for node in current_layer.nodes:
                node.set_terminal()
                node.print_info()
                # set node weight


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
