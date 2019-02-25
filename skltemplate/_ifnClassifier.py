"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import operator
from ._ifn_network import IfnNetwork, AttributeNode, Attribute_layer
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.metrics import mutual_info_score

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
        self.attributes_array = list(range(0, len(X[0])))
        self.update_attributes_array=self.attributes_array;

        max_MI ={}

        for attribute in self.attributes_array:
            attribute_data = []
            for record in X:
                attribute_data.append(record[attribute])
            max_MI[attribute]=metrics.adjusted_mutual_info_score(attribute_data, y)


        chosen_attribute =max(max_MI, key=max_MI.get)
        self.update_attributes_array.remove(chosen_attribute)


        layer = Attribute_layer(chosen_attribute)
        layer.set_nodes([AttributeNode(0), AttributeNode(1)])

        self.network.root_node.set_layer(layer)
        currentLayer =self.network.root_node.first_layer;
        for i in list(range(0,len(self.attributes_array))):
            arrayOfMI=[]
            for node in currentLayer.nodes:
                arrayOfMI.append(self.calIMPerNode(X,y))
            nextIndexLayer=self.getNextLayer(arrayOfMI)
            self.update_attributes_array.remove(nextIndexLayer)
            for node in currentLayer.nodes:
                node.next.append([AttributeNode(0), AttributeNode(1)])
            print('Layer number: ' + str(i) + '.  attribute number: ' + str(nextIndexLayer))
            layer = Attribute_layer(nextIndexLayer)
            layer=self.setNodes(len(currentLayer.nodes),layer)
            currentLayer.next_layer=layer
            currentLayer=layer

        self.is_fitted_ = True

        # `fit` should always return `self`
        return self

    def setNodes(self,numNode,layer):
        for i in list(range(0, numNode)):
            layer.set_nodes([AttributeNode(0), AttributeNode(1)])
        return layer


    def getAttribute(self):
        tempDic = {}
        for row in self.update_attributes_array:
            tempDic[row] = 0
        return tempDic

    def getNextLayer(self,arrayOfMI):
        tempDic=self.getAttribute()
        for dic in arrayOfMI:
            for key,value in dic.items():
                tempDic[key]=tempDic[key]+value
        return max(tempDic, key=tempDic.get)

    def calIMPerNode(self,X,y):
        max_MI = {}

        for attribute in self.update_attributes_array:
            attribute_data = []
            for record in X:
                attribute_data.append(record[attribute])
            max_MI[attribute] = metrics.adjusted_mutual_info_score(attribute_data, y)
        return max_MI



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
