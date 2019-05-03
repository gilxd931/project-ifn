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

        mutual_information = (value / total_records) * math.log(
            (value / partial_records) / ((curr_data_count / partial_records) * (curr_class_count / partial_records)), 2)

        total_mi += mutual_information
    return total_mi


def calc_weight(y, class_count, total_records):
    weight_per_class = {}
    for class_info in class_count:
        # partial_y = np.extract(y, np.where(y == [class_info[0]]), axis=0)
        cut_len = len(np.extract(y == [class_info[0]], y))
        if cut_len != 0:
            weight = (cut_len / total_records) * (math.log((cut_len / len(y)) / (class_info[1] / total_records), 2))
            weight_per_class[class_info[0]] = (weight, (cut_len / len(y)))
        else:
            weight_per_class[class_info[0]] = (0, 0)
    return weight_per_class


def drop_records(X, atr_index, y, node_index):
    new_x = []
    new_y = []
    for i in range(len(y)):
        if X[i][atr_index] == node_index:
            new_x.append(X[i])
            new_y.append(y[i])
    return np.array(new_x), np.array(new_y)


class IfnClassifier():
    attributes_array = []
    update_attributes_array = []
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    def __init__(self, alpha=0.999):
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

        global total_records
        global all_nodes_continuous_atts_data
        global attribute_node_mi_data
        global nodes_info_per_attribute
        global split_points
        global num_of_classes
        global unique_values_per_attribute

        total_records = len(y)
        unique, counts = np.unique(np.array(y), return_counts=True)
        class_count = np.asarray((unique, counts)).T

        nodes_info_per_attribute = {}
        # map that contains all the split points for all attributes
        split_points = {}

        # continuous and categorical attributes
        continuous_attributes_type = {}

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
            if len(unique_values_per_attribute[attribute]) / len(attribute_data) < 0.03:
                continuous_attributes_type[attribute] = False
            else:
                continuous_attributes_type[attribute] = True
                split_points[attribute] = []

            if continuous_attributes_type[attribute]:
                data_class_array = []
                for i in range(len(attribute_data)):
                    data_class_array.append((attribute_data[i], y[i]))
                data_class_array.sort(key=lambda tup: tup[0])

                nodes_info_per_attribute[attribute] = []
                nodes_info_per_attribute[attribute].append((0, 0))
                recursive_split_points(data_class_array, attribute, self, 0)

                if not bool(split_points[1]):  # there are not split points
                    attributes_mi[attribute] = 0
                else:
                    sum_of_splits = sum([pair[1] for pair in nodes_info_per_attribute[attribute]])
                    attributes_mi[attribute] = sum_of_splits
            else:
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

        # will need it for future calculations
        original_unique_values_per_attribute = unique_values_per_attribute.copy()

        # create new hidden layer of the maximal mutual information attribute and set the layer nodes
        first_layer = HiddenLayer(chosen_attribute)
        self.network.root_node.first_layer = first_layer
        nodes_list = []
        if continuous_attributes_type[chosen_attribute]:
            unique_values_per_attribute[chosen_attribute] = np.arange(len(split_points[chosen_attribute]))
        for i in unique_values_per_attribute[chosen_attribute]:
            x_y_tuple = drop_records(X, chosen_attribute, y, i)
            nodes_list.append(AttributeNode(curr_node_index, i, 0, chosen_attribute, x_y_tuple[0], x_y_tuple[1]))
            curr_node_index += 1
        first_layer.set_nodes(nodes_list)
        current_layer = first_layer

        # initialize map values to empty lists
        split_points = {key: [] for key in split_points}

        while len(updated_attributes_array) > 0:
            # get the attribute that holds the maximal mutual information
            nodes_info_per_attribute = {}
            all_nodes_continuous_atts_data = {}
            attribute_node_mi_data = {}
            all_continous_vars_data = {}

            # for each node in current layer
            for node in current_layer.nodes:
                # for each attribute
                for attribute in updated_attributes_array:
                    if attribute not in nodes_info_per_attribute:
                        nodes_info_per_attribute[attribute] = []
                    attribute_data = []
                    for record in node.partial_x:
                        attribute_data.append(record[attribute])

                    if continuous_attributes_type[attribute]:
                        data_class_array = []
                        all_continous_vars_data[attribute] = data_class_array

                        if attribute in all_nodes_continuous_atts_data:
                            all_nodes_att_map = all_nodes_continuous_atts_data[attribute]
                        else:
                            all_nodes_att_map = {}

                        for i in range(len(attribute_data)):
                            data_class_array.append((attribute_data[i], node.partial_y[i]))

                        data_class_array.sort(key=lambda tup: tup[0])
                        distinct_attribute_data = original_unique_values_per_attribute[attribute]

                        iter_att = iter(distinct_attribute_data)
                        next(iter_att)
                        for T in iter_att:
                            t_attribute_data = []
                            new_y = []
                            for data_class_tuple in data_class_array:
                                if data_class_tuple[0] < T:
                                    t_attribute_data.append(0)
                                else:
                                    t_attribute_data.append(1)

                                new_y.append(data_class_tuple[1])

                            t_mi = calc_MI(t_attribute_data, new_y, total_records)
                            statistic = 2 * np.log(2) * total_records * t_mi
                            critical = stats.chi2.ppf(self.alpha, (num_of_classes - 1))

                            if critical < statistic:
                                if attribute not in attribute_node_mi_data.keys():
                                    attribute_node_mi_data[attribute] = {}

                                if node.index not in attribute_node_mi_data[attribute].keys():
                                    attribute_node_mi_data[attribute][node.index] = {}

                                attribute_node_mi_data[attribute][node.index][T] = t_mi
                                if T in all_nodes_att_map:
                                    all_nodes_att_map[T] += t_mi
                                else:
                                    all_nodes_att_map[T] = t_mi
                        all_nodes_continuous_atts_data[attribute] = all_nodes_att_map

                    else:
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

            for attribute_index, continuous_att_mi_data in all_nodes_continuous_atts_data.items():
                if bool(continuous_att_mi_data):
                    split_point = max(continuous_att_mi_data, key=continuous_att_mi_data.get)
                    split_points[attribute_index].append(split_point)
                    for node in current_layer.nodes:
                        if node.index in attribute_node_mi_data[attribute_index].keys():
                            if split_point in attribute_node_mi_data[attribute_index][node.index].keys():
                                node_info_tuple = (
                                    node.index, attribute_node_mi_data[attribute_index][node.index][split_point])
                                nodes_info_per_attribute[attribute_index].append(node_info_tuple)

                                # check recursively sub intervals of split point for node
                                sub_interval_0 = []
                                sub_interval_1 = []

                                # for i in range(len(X)):
                                #     interval_tuple = X[i][attribute_index], y[i]
                                #     if interval_tuple[0] < split_point:
                                #         sub_interval_0.append(interval_tuple)
                                #     else:
                                #         sub_interval_1.append(interval_tuple)
                                for elem in all_continous_vars_data[attribute_index]:
                                    if elem[0] < split_point:
                                        sub_interval_0.append(elem)
                                    else:  # elif elem[0] > split_point:
                                        sub_interval_1.append(elem)

                                if len(sub_interval_0) > 0:
                                    recursive_split_points(sub_interval_0, attribute_index, self, node.index)
                                if len(sub_interval_1) > 0:
                                    recursive_split_points(sub_interval_1, attribute_index, self, node.index)

                            else:
                                new_node_tuple = (node.index, 0)
                                nodes_info_per_attribute[attribute_index].append(new_node_tuple)
                        else:
                            new_node_tuple = (node.index, 0)
                            nodes_info_per_attribute[attribute_index].append(new_node_tuple)
                else:
                    nodes_info_per_attribute[attribute_index].append((1, 0))

                unique_values_per_attribute[attribute_index] = np.array([0, 1])  # split to two nodes

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
                        # add weight to terminal node
                        node.set_weight_probability_pair(calc_weight(node.partial_y, class_count, total_records))

            nodes_info_per_attribute = {}  # initialize max nodes mi data
            nodes_list = []
            for curr_layer_node in current_layer.nodes:
                if not curr_layer_node.is_terminal:

                    # if chosen att is continuous we convert the partial x values their positions by the splits values
                    if continuous_attributes_type[chosen_attribute]:
                        unique_values_per_attribute[chosen_attribute] = np.\
                            arange(len(split_points[chosen_attribute]) + 1)

                        split_points[chosen_attribute].sort()
                        counter0 = 0
                        counter1 = 0
                        counter2 = 0


                        # delete until 10 lines from here
                        for record in X:
                            returned_split_index = find_split_position(record[chosen_attribute],
                                                                       split_points[chosen_attribute])
                            if returned_split_index == 0 :
                                counter0 += 1
                            elif returned_split_index == 1 :
                                counter1 +=1
                            else:
                                counter2 += 1
                            record[chosen_attribute] = returned_split_index

                        for record in curr_layer_node.partial_x:
                            returned_split_index = find_split_position(record[chosen_attribute],
                                                                       split_points[chosen_attribute])
                            record[chosen_attribute] = returned_split_index

                    for i in unique_values_per_attribute[chosen_attribute]:
                        x_y_tuple = drop_records(curr_layer_node.partial_x,
                                                 chosen_attribute, curr_layer_node.partial_y, i)
                        nodes_list.append(AttributeNode(curr_node_index, i, curr_layer_node.index,
                                                        chosen_attribute, x_y_tuple[0], x_y_tuple[1]))
                        curr_node_index += 1

            updated_attributes_array.remove(chosen_attribute)
            new_layer = HiddenLayer(chosen_attribute)
            current_layer.next_layer = new_layer
            current_layer = new_layer

            current_layer.set_nodes(nodes_list)

            # initialize map values to empty lists
            split_points = {key: [] for key in split_points}

        # that means we used all of the attributes so we have to set the last layer's nodes to be terminal
        # or all nodes in current layer are terminal
        if len(updated_attributes_array) == 0 or chosen_attribute == -1:
            for node in current_layer.nodes:
                node.set_terminal()
                # add weight to terminal node
                node.set_weight_probability_pair(calc_weight(node.partial_y, class_count, total_records))

        # `fit` should always return `self`
        self.is_fitted_ = True
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
        predicted = []
        for record in X:
            curr_layer = self.network.root_node.first_layer
            prev_node_index = 0
            found_terminal_node = False
            while curr_layer is not None and not found_terminal_node:
                record_value = record[curr_layer.index]
                for node in curr_layer.nodes:
                    if node.inner_index == record_value and node.prev_node == prev_node_index:
                        chosen_node = node
                        if chosen_node.is_terminal:
                            max_weight = -math.inf
                            predicted_class = -math.inf
                            for class_index, weight_prob_pair in chosen_node.weight_probability_pair.items():
                                if weight_prob_pair[0] > max_weight:
                                    max_weight = weight_prob_pair[0]
                                    predicted_class = class_index
                            predicted.append(predicted_class)
                            found_terminal_node = True
                        else:
                            curr_layer = curr_layer.next_layer
                            prev_node_index = chosen_node.index
                        break

        return np.array(predicted)

    def predict_proba(self, X):
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
        predicted = []
        for record in X:
            curr_layer = self.network.root_node.first_layer
            prev_node_index = 0
            found_terminal_node = False
            while curr_layer is not None and not found_terminal_node:
                record_value = record[curr_layer.index]
                for node in curr_layer.nodes:
                    if node.inner_index == record_value and node.prev_node == prev_node_index:
                        chosen_node = node
                        if chosen_node.is_terminal:
                            found_terminal_node = True
                            weights_of_node = []
                            for class_index, weight_prob_pair in chosen_node.weight_probability_pair.items():
                                weights_of_node.append((weight_prob_pair[1]))
                            predicted.append(weights_of_node)
                        else:
                            curr_layer = curr_layer.next_layer
                            prev_node_index = chosen_node.index
                        break

        return np.array(predicted)


def recursive_split_points(sub_interval, attribute, self, node_index):

    sub_interval_values = [i[0] for i in sub_interval]
    distinct_attribute_data = np.unique(sub_interval_values)

    sub_data_map = {}

    iter_att = iter(distinct_attribute_data)
    next(iter_att)
    for T in iter_att:
        t_attribute_data = []
        new_y = []
        for data_class_tuple in sub_interval:
            if data_class_tuple[0] < T:
                t_attribute_data.append(0)
            else:
                t_attribute_data.append(1)

            new_y.append(data_class_tuple[1])

        t_mi = calc_MI(t_attribute_data, new_y, total_records)
        statistic = 2 * np.log(2) * total_records * t_mi
        critical = stats.chi2.ppf(self.alpha, (num_of_classes - 1))

        if critical < statistic:
            sub_data_map[T] = t_mi

    if bool(sub_data_map):
        split_point = max(sub_data_map, key=sub_data_map.get)
        if split_point not in split_points[attribute]:
            split_points[attribute].append(split_point)
        if attribute not in nodes_info_per_attribute:
            nodes_info_per_attribute[attribute] = []

        for node_tuple in nodes_info_per_attribute[attribute]:
            if node_tuple[0] == node_index:
                new_node_tuple = (node_index, node_tuple[1] + sub_data_map[split_point])
                nodes_info_per_attribute[attribute].append(new_node_tuple)
                nodes_info_per_attribute[attribute].remove(node_tuple)

        sub_interval_0 = []
        sub_interval_1 = []
        for elem in sub_interval:
            if elem[0] < split_point:
                sub_interval_0.append(elem)
            else:
                sub_interval_1.append(elem)

        if len(sub_interval_0) > 0:
            recursive_split_points(sub_interval_0, attribute, self, node_index)
        if len(sub_interval_1) > 0:
            recursive_split_points(sub_interval_1, attribute, self, node_index)
    # else:
    #     node_exist = False
    #     for node_tuple in nodes_info_per_attribute[attribute]:
    #         if node_tuple[0] == node_index:
    #             node_exist = True
    #     if not node_exist:
    #         new_node_tuple = (node_index, 0)
    #         nodes_info_per_attribute[attribute].append(new_node_tuple)


def find_split_position(record, positions):
    if record < positions[0]:
        return 0

    if record >= positions[len(positions)-1]:
        return len(positions)

    for i in range(len(positions)):
        first_position = positions[i]
        second_position = positions[i+1]
        if first_position <= record <= second_position:
            return i+1

