#! /usr/bin/env python
from sys import argv, exit, stdout, path
import numpy as np
import os
import string
import matplotlib
from matplotlib import pyplot as plt


def niceplot(
    ax, kind="default", autoscale=True, grid=True, lfs=10, labelsize=14, ticksize=12
):
    if autoscale:
        ax.autoscale(enable=True, axis="both")
    # plt.legend(bbox_to_anchor=(1, 1))#, bbox_transform=plt.gcf().transFigure)
    ax.tick_params(labelsize=ticksize)
    ax.xaxis.label.set_size(labelsize)
    ax.yaxis.label.set_size(labelsize)
    ax.minorticks_on()
    if grid:
        ax.grid(1)
    ax.tick_params("both", length=6, width=1, which="major", direction="out")
    ax.tick_params("both", length=3, width=0.5, which="minor", direction="out")
    plt.setp(ax.get_lines(), linewidth=1.5)
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color((0.68, 0.68, 0.68))
        if isinstance(child, matplotlib.legend.Legend):
            texts = child.get_texts()
            for t in texts:
                t.set_fontsize(lfs)


"""
def get_color(i):
    colors = [(31,119,180),
              (255,127,14),
              (44,160,44),
              (214,39,40),
              (148,103,189),
              (140,86,75),
              (127,127,127),
              (188,189,34),
              (23,190,207),
              (23,190,207),
              (255,128,14),
              (171,171,171),
              (95,158,209),
              (89,89,89),
              (0,107,164)]
    i = i%(len(colors)-1)
#    plt.autoscale(enable=True, axis='both', tight=True)
    colors[:] = [[x/255. for x in y] for y in colors]
    return colors[i]
#    ax.legend(bbox_to_anchor=(0, 1), loc=2)

def set_axes_probs(ax):
    ax.tick_params(labelsize=14)
    ax.minorticks_on()
    ax.tick_params('both', length=8, width=1, which='major', direction='out')
    ax.tick_params('both', length=4, width=.5, which='minor', direction='out')
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    plt.tight_layout()
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color((.5,.5,.5))
            child.set_linewidth(8)
      
def niceplot(name):
  fig = plt.gcf()
  fig.set_facecolor('white')
  if name=='plot_poissongamma':
    ax = plt.gca()
    set_axes_probs(ax)
    ax.set_xlabel(r'$\langle k\rangle$')
    ax.set_ylabel(r'$\mathsf{photon probability}$')
    for idx,line in enumerate(ax.get_children()):
      line.set_color(get_color(idx))
      line.set_linewidth(2.0)
    #plt.autoscale(enable=True, axis='x', tight=True)
    ax.set_ylim([1e-7,2.])
  if name=='plot_jfunction':
    ax = plt.gca()
    set_axes_probs(ax)
    ax.set_xlabel(r'$n$')
    ax.set_ylabel(r'$\beta$')
    ax.set_xscale('log')

    #for idx,line in enumerate(ax.get_lines()):
    #  line.set_color(get_color(idx))
    #  line.set_linewidth(2.0)
    #plt.autoscale(enable=True, axis='x', tight=True)
    #ax.set_ylim([1e-7,2.])
"""
