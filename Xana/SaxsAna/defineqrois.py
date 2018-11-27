import numpy as np
from ..Xplot.plotqrois import plotqrois

def getqroi(saxs, setup, qr, phir=None, mask=None, mirror=False):
    if mask is None:
        mask = np.ones_like(saxs)
    mask = mask.astype(np.uint8)
    
    wavelength = setup['lambda']/10
    pix_size = setup['pix_size'][0]*1e-6
    distance = setup['distance']
    cx, cy = setup['ctr']
    dim2, dim1 = np.shape(saxs)
    wf = 4 * np.pi / wavelength
    [X,Y] = np.mgrid[1-cy:dim2+1-cy,1-cx:dim1+1-cx]
    radius = np.sqrt(X**2 + Y**2)
    q = wf * np.sin(np.arctan(radius * pix_size / distance) / 2)
    phi = np.arctan2(-X, Y)

    qv = qr[:,0]
    dqv = qr[:,1]
    
    if phir is None:
        phiv = [0.]
        dphi = [360.]
    else:
        phiv = phir[:,0]
        dphi = phir[:,1]
        
    phiv = phiv*np.pi/180
    dphi = dphi*np.pi/180
        
    ind = []
    r = []
    ph = []

    for i in range(len(qv)):
        for j in range(len(phiv)):
            tmp_q = (q>=(qv[i]-dqv[i]/2)) & (q<=(qv[i]+dqv[i]/2)) & mask
            phit = phi.copy()
            phit = (phit - phiv[j]) % (2*np.pi)
            if mirror:
                tmp_phi = (phit <= dphi[j]) | (((phit - np.pi) % (2*np.pi)) <= dphi[j])
            else:
                tmp_phi = (phit<=(dphi[j]))
            tmp = np.where(tmp_q & tmp_phi)
            del phit
            if len(tmp[0]):
                ind.append(tmp)
                r_min = radius[tmp].min()
                r_max = radius[tmp].max()
                r.append((r_max, r_max - r_min))
    return ind, r

def flatten_init(inp):
    
    def convert(s):
        if np.issubdtype(type(s[0]), np.number):
            return (np.array([s[0]]), s[1])
        else:
            return s
    
    def get_stack(p):
        p = convert(p)
        return np.hstack((p[0],np.ones(len(p[0]))*p[1])).reshape(-1,2, order='F')

    i = 0
    for p in inp:
        stack = get_stack(p)
        if i == 0:
            rois = stack.copy()
        else:
            rois = np.vstack((rois,stack))
        i += 1
    return rois
        
def defineqrois(setup, Isaxs, mask=None, qv_init=None, phiv_init=[(0,360)], 
                plot=False, d=250, mirror=False, **kwargs):
    if qv_init is None:
        try:
            qv_init = setup['qv_init']
            if phiv_init is None and 'phiv_init' in setup.keys():
                phiv_init = setup['phiv_init']
        except KeyError:
            print('Setup does not contain "qv_init" to initialize Q ROIs.')
    else:
        qv_init = flatten_init(qv_init)
        phiv_init = flatten_init(phiv_init)
        setup['qv_init'] = qv_init
        setup['phiv_init'] = phiv_init

    if mask is None:
        mask = np.ones_like(Isaxs, dtype=np.bool)

    phiv_init[:,0] -= phiv_init[:,1]/2
    
    qroi, r = getqroi(Isaxs, setup, qv_init, mask=mask, phir=phiv_init, mirror=mirror)

    setup['dqv'] = qv_init[:,1]
    setup['phiv'] = phiv_init
    setup['qv'] = np.tile(qv_init[:,0],setup['phiv'].shape[0])
    setup['r'] = r

    gproi = np.ones(len(qroi),np.float32)
    for i in range(len(qroi)):
        gproi[i] = np.sum(mask[qroi[i]])
    setup['gproi'] = gproi
    setup['qroi'] = qroi
    setup['qv'] = setup['qv'][:len(setup['qroi'])]
        
    xmin = min([x[0].min() for x in setup['qroi']])
    ymin = min([x[1].min() for x in setup['qroi']])
    xmax = max([x[0].max() for x in setup['qroi']])
    ymax = max([x[1].max() for x in setup['qroi']])

    setup['qsec'] = ((xmin, ymin), (xmax, ymax))
    print('Added the following Q-values [nm-1]:\n{}'.format(setup['qv']))

    if plot:
        plotqrois(Isaxs, mask, setup, method=plot, d=d, shade=True, mirror=mirror, **kwargs)
