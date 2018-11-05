#! /usr/bin/env python

#etermine if a point is inside a given polygon or not
# Polygon is a list of (x,y) pairs.
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import string
from matplotlib.path import Path
from pylab import *
from readinput import *
from sys import argv,exit,stdout
import sys
import Image
from psana import *
import scipy.ndimage as ndimage




####### input  #################
def on_click(event):
    global key, x, y,lc,data,im,xy,mask,xx,yy,px,lx,lm,tmp
    if not event.inaxes: 
        xy=[]
        return
    x,y=int(event.xdata), int(event.ydata)
    xx.append([x])
    yy.append([y])
    xy.append([y,x])
    lc.set_data(xx,yy)
    draw()
def on_click_key(event):
    global key, x, y,lc,data,im,xy,mask,xx,yy,px,lx,lm,tmp,savemask
    key=event.key
    if not event.inaxes: 
        xy=[]
        return
    x,y=int(event.xdata), int(event.ydata)
    xx.append([x])
    yy.append([y])
    xy.append([y,x])
    if key=='a':
        xx=[]
        yy=[]
        xy=[]  
        x=0
        y=0
        lc.set_data(xx,yy)
        lm.set_data(xx,yy)
        draw()
    if key=='m': 
        xx[-1]=xx[0]
        yy[-1]=yy[0]
        xy[-1]=xy[0]
        path = Path(xy)
        ind=path.contains_points(points).reshape(ly,lx).T
        data.mask[ind]=True 
        data.mask[original_mask]=True
        im.set_data(data)
        im.set_clim(vmax=data.max())
        xx=[]
        yy=[]
        xy=[]  
        ind=[]
        lc.set_data(xx,yy)
        lm.set_data(xx,yy)
        draw()
        x=0
        y=0 
    if key=='u':
        xx[-1]=xx[0]
        yy[-1]=yy[0]
        xy[-1]=xy[0]
        path = Path(xy)
        ind=path.contains_points(points).reshape(ly,lx).T
        data.mask[where((ind==True)&(original_mask==False))]=False
        im.set_data(data)
        im.set_clim(vmax=data.max())
        xx=[]
        yy=[]
        xy=[] 
        ind=[]
        lc.set_data(xx,yy)
        lm.set_data(xx,yy)
        draw()
        x=0
        y=0 
    if key=='w':
        maskname=savemask
        np.save(maskname,array(ma.getmask(data),dtype=int))
        os.system("./plot2d.py %s -title %s -wn makemask &" % (maskname,maskname))
        close()
        return 0
    if key=='e': 
        close()
        return 0

def on_move(event):
    global lm,x,y
    if not event.inaxes: return
    xm,ym=int(event.xdata), int(event.ydata)
    # update the line positions
    if x!=0: 
        lm.set_data((x,xm),(y,ym))
        draw()

###############################################################################3

def get2x2arr(cspad2x2_evt):
   small=cspad2x2_evt.data()
   chip1=small[:,:,0]
   chip2=small[:,:,1]
   sml=np.zeros((400,400),dtype="int16")
   o0=2
   sml[o0:185+o0,0:193]=chip1[:,0:193]
   o1=3
   sml[o0:185+o0,194+o1:387+o1]=chip1[:,194:387]
   o0=202
   sml[o0:185+o0,0:193]=chip2[:,0:193]
   o1=3
   sml[o0:185+o0,194+o1:387+o1]=chip2[:,194:387]
   return sml

def smallzeroordermask():
   sml=np.ones((400,400),dtype="int16")
   o0=2
   sml[o0:185+o0,0:193]=0
   o1=3
   sml[o0:185+o0,194+o1:387+o1]=0
   o0=202
   sml[o0:185+o0,0:193]=0
   o1=3
   sml[o0:185+o0,194+o1:387+o1]=0
   return sml
    
def getCSPADgeometry():
   slab_location = np.array([[2,2],[3,2],[0,2],[0,3],[1,0],[0,0],
                             [2,0],[2,1],[4,2],[4,3],[5,0],[4,0],
                             [6,1],[6,0],[7,2],[6,2],[5,4],[4,4],
                             [6,5],[6,4],[6,6],[7,6],[4,7],[4,6],
                             [2,5],[2,4],[2,6],[3,6],[0,6],[0,7],
                             [0,4],[1,4]])
   slab_location2 = np.array([[ 419, 547],[ 627, 542],[   0, 538],[  -2, 751],[ 213, 119],[   2, 118],
                              [ 428, 134],[ 429, 347],[ 830, 508],[ 831, 721],[1047,  87],[ 835,  87],
                              [1262, 297],[1261,  84],[1450, 516],[1236, 517],[1091, 916],[ 879, 915],
                              [1306,1132],[1308, 919],[1298,1345],[1512,1345],[ 882,1534],[ 883,1321],
                              [ 455,1178],[ 458, 960],[ 449,1391],[ 660,1391],[  28,1384],[  28,1597],
                              [  47, 962],[ 259, 962]])
   quad_offset = np.array([[0, 0], [6, -15], [ 3, -1],[16, -3]])
   isUD = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
   isLR = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1])
   isTran=np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0])
   return (slab_location,slab_location2,quad_offset,isUD,isLR,isTran)

def zeroordermask():
   mask0=np.ones((1800,1800),dtype="int")
   (slab_location,slab_location2,quad_offset,isUD,isLR,isTran)=getCSPADgeometry()
   #modifying cspad image... 
   i=0
   #running over 4 CSPAD quadrants....
   for n in range(0,4):  
         #quad = cspad.quads(n)
         #d = quad.data()
         #running over 8 modules per quadrant....
         for j in range(0,8):
             buffer1 = np.zeros((185,388),dtype="int32")
             buffer1[:,193:198] =1        

             quad_ind =  np.ceil(i/8)   
             slab_location_final = slab_location2[i,:] + [50, -20] + quad_offset[quad_ind,:]
             if isTran[i]==1:
                 buffer1=np.transpose(buffer1)
             if isLR[i]==1:
                 buffer1 = np.fliplr(buffer1)
             if isUD[i]==1:
                 buffer1 = np.flipud(buffer1)

             buffer2 = buffer1
             slab_size = np.shape(buffer2)
             mask0[slab_location_final[0]:slab_location_final[0]+slab_size[0],
                         slab_location_final[1]:slab_location_final[1]+slab_size[1]]=buffer2
             i+=1
   #cspad image was modified
   return mask0
       
###########################################################
def usage_message():
    print("Usage is: %s [options]" % sys.argv[0])
    print("options are:")
    print("      -dark <darkfile>   uses pixels above threshold of dark file")
    print("      -t <threshold>     high threshold")
    print("      -lt <threshold>    low threshold")    
    print("      -m <maskfile>      removes additional pixels from existing mask file")
    print("      -s <saxsfile>      loads a SAXS file in order to see beamstop")
    print("      -sa <saveas>       specifies output file")
    print("      -2x2               sif 2x2 panel is used")
    sys.exit(0)

def getpos(str):
   for ii in range(len(sys.argv)):
       if sys.argv[ii]==str:
           return ii
       
def getoptions():
    global savemask
    options={}
    if "-t" in sys.argv:
        ii=getpos("-t")
        options['th_hotpix']=float(sys.argv[ii+1])
    if "-lt" in sys.argv:
        ii=getpos("-lt")
        options['th_coldpix']=float(sys.argv[ii+1])
    if "-s" in sys.argv:
        ii=getpos("-s")
        options['saxs']=np.load(sys.argv[ii+1].rstrip(".npy").rstrip(".edf")+".npy")
    if "-m" in sys.argv: 
        ii=getpos("-m") 
        maskfile = sys.argv[ii+1].rstrip(".npy").rstrip(".edf")+".npy"
        options['mask']=np.load(maskfile)
    if "-sa" in sys.argv:
        ii=getpos("-sa")
        savemask = sys.argv[ii+1].rstrip(".npy").rstrip(".edf")+".npy"
        options['savemask']=savemask
    return options


################################################################
######## MAIN ##################################################
xpixels=1800
ypixels=1800
big=1

#read input parameters
if len(sys.argv) <2 or "-h" in sys.argv:
   usage_message()

#read input parameters
options = getoptions()

if "-2x2" in sys.argv:
    xpixels=400
    ypixels=400
    big=0

#mask area of CSPAD that is dead
if big:
    zero_mask=zeroordermask()
else:
    zero_mask=smallzeroordermask()
current_mask=zero_mask

#masking with hot pixels from a dark image
if '-dark' in sys.argv:
    ii=getpos("-dark")
    darkf=sys.argv[ii+1]
    dark=np.load(darkf)

    tmpname="./tmpdir/dark.npy"
    np.save(tmpname,dark)
    title="Dark\ %s" % darkf
    os.system("./plot2d.py %s -wn makemask -title %s -min 0 &" % (tmpname,title))

    if 'th_hotpix' in options:
        th_hotpix = options['th_hotpix']
    else:
        th_hotpix = 2000

    if 'th_coldpix' in options:
        th_coldpix = options['th_coldpix']
    else:
        th_coldpix = 1000

    dark_mask=current_mask

    hot_mask=np.zeros((xpixels,ypixels),dtype="int")
    hot_mask[dark>th_hotpix]=1
    hot_mask[dark<th_coldpix]=1
    
    #get neighbors of hot_mask
    struct=np.ones((3,3), dtype=bool)
    hotN_mask=ndimage.binary_dilation(hot_mask,structure=struct).astype(hot_mask.dtype)
    dark_mask[hotN_mask==1]=1
    
    tmpname="./tmpdir/dark_mask.npy"
    np.save(tmpname,dark_mask)
    title="dark\ mask\ with\ neighbours"
    os.system("./plot2d.py %s -wn makemask -title %s -min 0 &" % (tmpname,title))

    current_mask=dark_mask

#masking with an existing mask file
if 'mask' in options:
    loaded_mask=options['mask']
    current_mask[loaded_mask==1]=1

    tmpname="./tmpdir/current_mask.npy"
    np.save(tmpname,current_mask)
    title="mask\ plus\ extra\ mask"
    os.system("./plot2d.py %s -wn makemask -title %s -min 0 &" % (tmpname,title))

#masking by hand a visible beamstiop or so from a data image    
#using for example a saxs image 
if 'saxs' in options:
    saxs_image=options['saxs']
    print("\nUse now cursor and keyboard:\n   m to mask\n   u to unmask\n   a to cancel ROI\n   w to save\n   e to exit.")
    #current_mask = zip(*current_mask[::-1])
    if '-new' in sys.argv:
        current_mask=np.zeros((1750,1750))

    data=np.ma.array(saxs_image,mask=current_mask)
    original_mask=ma.getmask(data)
    
    lx,ly=shape(data)
    x, y = meshgrid(arange(lx), arange(ly))
    x, y = x.flatten(), y.flatten()
    points = vstack((x,y)).T

    key=[]
    x=0
    y=0
    xy=[]
    xx=[]
    yy=[]

    rc('image',origin = 'lower')
    rc('image',interpolation = 'nearest')
    data[np.where(data<0.0001)]=0.0001
    data=log10(data)
    fig = figure()
    fig.canvas.get_tk_widget().focus_force()
    px=subplot(111)
    im=imshow(data,vmax=data.max())
    colorbar()
    lc,=px.plot((0,0),(0,0),'-+w',linewidth=1.5,markersize=8,markeredgewidth=1.5)
    lm,=px.plot((0,0),(0,0),'-+w',linewidth=1.5,markersize=8,markeredgewidth=1.5)
    px.set_xlim(0,ly)
    px.set_ylim(0,lx)
    cidb=connect('button_press_event',on_click)
    cidk=connect('key_press_event',on_click_key)
    cidm=connect('motion_notify_event',on_move)
    show()

#no interactive mouse clicking a better mask... One still needs to save mask
else:
    if savemask=="":
       savemask="tmpdir/mask_tmp.npy"
    np.save(savemask,current_mask)
    
