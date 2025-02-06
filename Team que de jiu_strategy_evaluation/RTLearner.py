# Contains the code for the regression Decision Tree class.  

import numpy as np
import math
import matplotlib.pyplot as plt
import random
# import pdb
class RTLearner:
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        """
        As mentioned: Multiple calls to testPolicy() method need to return exactly the same result. 
        """
        self.seed = 904053395
        random.seed(self.seed)
        np.random.seed(self.seed) 

    def author(self):
        return "zdong312"
    
    def study_group():
        return "zdong312"

    def add_evidence(self, data_x, data_y):

        self.tree = self.build_tree(data_x, data_y)
        if self.verbose:
            print("RTLearner")
            print("Tree shape: ", self.tree.shape)
            print("Tree structure")
            print(self.tree)

    def build_tree(self, data_x, data_y):
        if data_x.shape[0] <= self.leaf_size:
            return np.asarray([np.nan, np.mean(data_y), np.nan, np.nan])
        # If all the y values are the same, return a leaf node
        # np.isclose check if two arrays are element-wise equal within a tolerance we can use atol for absolute tolerance or rtol for relative tolerance
        if np.all(np.isclose(data_y,data_y[0])):
            return np.asarray([np.nan, data_y[0], np.nan, np.nan])
        
        
        ### This is the else situation
        random_index = random.randrange(data_x.shape[1]) # randomly choose the best feature to split on.
        random_1, random_2 = random.sample(range(data_x.shape[0]), 2) # randomly choose two points to split on.
        SplitVal = (data_x[random_1][random_index] + data_x[random_2][random_index]) / 2 # choose the split value as the average of the two points
        
        left_mask = data_x[:,random_index] <= SplitVal # from numerical value to boolean value
        # If all the y values are the same, return a leaf node
        # np.isclose check if two arrays are element-wise equal within a tolerance we can use atol for absolute tolerance or rtol for relative tolerance
        if np.all(np.isclose(left_mask,left_mask[0])):
            return np.asarray([np.nan, np.mean(data_y), np.nan, np.nan])
        
        right_mask = data_x[:,random_index] > SplitVal  ## 这也不一样

        
        lefttree = self.build_tree(data_x[left_mask], data_y[left_mask])
        righttree = self.build_tree(data_x[right_mask], data_y[right_mask])

        if lefttree.ndim == 1:
            node = np.array([random_index, SplitVal, 1, 2])
        else:
            node = np.array([random_index, SplitVal, 1, lefttree.shape[0] + 1])

        return np.vstack((node, lefttree, righttree))     


    def query(self, points):
        ## simplified version
        return np.array([self.query_point(point) for point in points])

    def query_point(self, point):
        node = 0
        # we check if the node is Nan, if not we move to the next node, if it is Nan, we return the value of the node
        while not np.isnan(self.tree[node][0]):
            
            split_value = point[int(self.tree[node][0])]
            # pdb.set_trace()
            # print('-'*10)
            # print(split_value) # the value of the feature we are splitting on
            # print(self.tree[node][0]) # the best index of the feature we are splitting on
            # print(self.tree[node][1]) # the split value of the feature we are splitting on
            # print(self.tree[node][2]) # the left node
            # print(self.tree[node][3]) # the right node

            # we keep moving to the next node based on the split value, and tracking the nodes. 
            if split_value <= self.tree[node][1]:
                node += int(self.tree[node][2])
            else:
                node += int(self.tree[node][3])
        return self.tree[node][1]

