import sys, os, re
import numpy as np
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import copy
from .Setup import Setup
from .Xdb import Xdb


class AnaList:
    
    def __init__(self, Xana, cmap='jet'):

        self.Xana = Xana
        self.colors = None
        self.cmap = cmap
        self.update_colors()
        self.markers = ['o',]
        
    def __str__(self):
        return 'Analist: super class for analysis classes.'

    def __repr__(self):
        return self.__str__()

    def update_colors(self, cmap=None, factor=1, repeat=1,):
        """Return vector of colors for plots
        """
        factor = int(factor)
        if cmap is None:
            cmap = self.cmap
        else:
            self.cmap = cmap

        self.colors = []
        cm = plt.get_cmap(cmap)
        ci = np.linspace(0,1,factor)
        for i in range(factor):
            self.colors.append(cm(ci[i]))
        self.colors = np.tile(self.colors, (repeat, 1))

    def update_markers(self, nmarkers, change_marker=0):
        """Return vector of markers for plots
        """
        self.markers = ['o',]
        if change_marker:
            mlist = list(matplotlib.markers.MarkerStyle.markers.keys())[:-4]
            del mlist[2], mlist[1]
            self.markers.extend(mlist)
        else:
            self.markers = ['o',]*nmarkers
    

    
   



