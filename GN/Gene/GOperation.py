'''

@author: ben
'''


class GOperation:

    def __init__(self, function, num_params, name, grad=None):

        self._function = function
        self._paramsNums = num_params
        self._arity = num_params
        self._name = name
        self._gradiant = grad

    def get_name(self):
        return self._name

    def gradiant(self):
        return self.gradiant

    def get_func(self):
        return self._function

    def get_arity(self):
        return self._paramsNums
