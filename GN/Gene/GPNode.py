
from __future__ import division

from abc import abstractmethod, ABCMeta

class GPNode(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def evaluate( self, _input ):
        raise NotImplementedError('This is an abstract class')    

    @abstractmethod
    def __repr__( self):
        raise NotImplementedError('This is an abstract class')


    @abstractmethod
    def display( self, indent = 0 ):
        raise NotImplementedError('This is an abstract class')

    @abstractmethod
    def nodes( self):
        raise NotImplementedError('This is an abstract class')
