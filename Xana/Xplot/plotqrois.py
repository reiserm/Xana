import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from Xplot.niceplot import niceplot
from SaxsAna.integrate import get_soq
from mpl_toolkits.axes_grid1 import make_axes_locatable


def shadeqrois(ax, qv, dqv, alpha=0.3, cmap='inferno', coords='data'):
    boxes = []
    
    # Loop over data points; create box from errors at each point
    cmap = plt.get_cmap(cmap)
    clrs = cmap(np.linspace(.1,.9,qv.size))
    for qi,ci,dqi in zip(qv,clrs,dqv):
        q1 = qi - dqi/2
        q2 = qi + dqi/2
        if coords == 'axes':
            yl = ax.get_ylim()
        elif coords == 'data':
            x, y = ax.lines[-1].get_data()
            y = y[np.argmin(np.abs(q1 - x)):np.argmin(np.abs(q2 - x))]
            yl = (y.min(), y.max())
        rect = patches.Rectangle((q1, yl[0]), dqi, yl[1]-yl[0])
        boxes.append(rect)

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(boxes, facecolors=clrs, alpha=alpha,
                         edgecolor='k')

    # Add collection to axes
    ax.add_collection(pc)
    
def shade_wedges(ax, setup, alpha=0.3, cmap='inferno', qsec=(0,0), mirror=False):
    yl = ax.get_ylim()
    wedges = []
    r = setup['r']
    phiv = setup['phiv']
    nr = len(r)
    nph = len(phiv)
        
    center = setup['ctr'] - qsec[::-1]
    
    # Loop over data points; create box from errors at each point
    cmap = plt.get_cmap(cmap)
    clrs = cmap(np.linspace(0,1,nr*nph))
    
    for ri in r:
        for phi in phiv:
            w = patches.Wedge(center, ri[0], -(phi[0]+phi[1]), -phi[0] , width=ri[1])
            wedges.append(w)
            if mirror:
                w = patches.Wedge(center, ri[0], -(phi[0]+phi[1])-180, -phi[0]-180 , width=ri[1])
                wedges.append(w)

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(wedges, facecolors=clrs, alpha=alpha,
                         edgecolor='k')

    # Add collection to axes
    lims = (ax.get_xlim(), ax.get_ylim())
    ax.add_collection(pc)
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])

def plotqrois(Isaxs, mask, setup, method='S(Q)', d=0, shade=False, color='r', ax=None, label='',
              mirror=False):
    dim = Isaxs.shape

    if 'qsec' in setup and d==0:
        y1, x1 = setup['qsec'][0]
        y2, x2 = setup['qsec'][1]
    else:
        x1,x2 = ( max( setup['ctr'][0]-d, 0 ), min( setup['ctr'][0]+d, dim[1] ) )
        y1,y2 = ( max( setup['ctr'][1]-d, 0 ), min( setup['ctr'][1]+d, dim[0] ) )

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(6,5))

    if method == 'S(Q)' or method == 1:
        q, I, e  = get_soq(Isaxs, mask, setup)
        ax.loglog(q, I, '.-', color=color, label=label)
        ax.set_xlabel(r'q [$\mathrm{nm}^{-1}$]')
        ax.set_ylabel(r'S(Q)')
        niceplot(ax)
        if shade:
            shadeqrois(ax, setup['qv'], setup['dqv'])
        plt.tight_layout()
    elif method == 'ROIS' or method == 2:
        
        ## uncomment to check the masked qrois
        # Isaxs = Isaxs.copy()
        # for q in setup['qroi']:
        #     Isaxs[q[0],q[1]] = 1000

        saxs_sec = (Isaxs*mask)[y1:y2,x1:x2]
        im = ax.imshow(saxs_sec, cmap=plt.get_cmap('jet'), norm=LogNorm())
        
        shade_wedges(ax, setup, alpha=0.3, cmap='inferno', qsec=(y1,x1), mirror=mirror)

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cl = plt.colorbar(im, cax=cax)
        ax.xaxis.set_visible(0)
        ax.yaxis.set_visible(0)

        niceplot(ax, autoscale=False)
        plt.grid()
        plt.tight_layout()
    plt.show()

