# Contains the code for the regression Bag Learner (i.e., a BagLearner containing Random Trees).  

import numpy as np

class BagLearner:
    """
     The “kwargs” are keyword arguments that are passed on to the learner’s constructor and they can vary according to the learner (see example below). 
     The “bags” argument is the number of learners you should train using Bootstrap Aggregation. 
     If boost is true, then you should implement boosting (optional implementation). 
     If verbose is True, your code can generate output; otherwise, the code should be silent. 
    """
    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = []
        for i in range(0, bags):
            self.learners.append(learner(**kwargs))
    def author(self):
        return "zdong312" 
    def study_group():
        return "zdong312"
    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            indices = np.random.choice(data_x.shape[0], data_x.shape[0], replace=True)
            learner.add_evidence(data_x[indices], data_y[indices])
    def query(self, points):
        results = np.zeros((points.shape[0], len(self.learners)))
        for i, learner in enumerate(self.learners):
            results[:, i] = learner.query(points)
        return np.mean(results, axis=1)
