#! /usr/bin/env python2.7
import numpy as np
import os
from DropletizeYuri.XPCS_mario import dropimgood, dropimgood_nn
import numpy.ma as ma
import EdfFile
import time


def loadedf(filename, imgn=0):
    f = EdfFile.EdfFile(filename)
    return f.GetData(imgn)


def count(a):
    res = [0] * 10
    ct = 1
    for i in range(1, len(a)):
        if a[i] >= 1:
            if a[i] == a[i - 1]:
                ct += 1
            else:
                if ct <= 9:
                    res[ct - 1] += 1
                ct = 1
    return res


mf = "/gpfs/exfel/data/user/reiserm/GeO2_ID10/"
if not "dark" in locals():
    darks = np.load(mf + "darks.npy")
    dark = np.mean(darks, axis=-1)
if not "mask" in locals():
    mask = ~np.load(mf + "mask.npy")
if not "data" in locals():
    if os.path.isfile(mf + "tmp_dat.npy"):
        data = np.load(mf + "tmp_dat.npy")
    else:
        print("Loading data...")
        tmp = loadedf("./data/GeO2_att0_3sec/img_0001.edf")
        data = np.zeros((tmp.shape[0], tmp.shape[1], 778), dtype=np.float16)
        for i in range(778):
            filename = "./data/GeO2_att0_3sec/img_{0:04d}.edf".format(i + 1)
            data[:, :, i] = loadedf(filename)
            if i % 10 == 0:
                print(i)
        print("Data Loaded.")
        np.save(mf + "tmp_dat.npy", data)
dd = np.load(
    mf + "DropDat.npy"
)  #'DropletizeYuri/DropData/GeO2_att0_3sec/''009_DropDat_YuriModified.npy'
coun = np.load(mf + "coun_nn2.npy")
pix = np.load(mf + "pix_nn2.npy")

# _______Main_______
# mint = np.zeros(778)
# for i in range(data.shape[-1]):
#     tmp = data[:,:,i]*1.-dark
#     tmp[tmp<100] = 0.
#     mint[i] = np.mean(tmp[~mask].astype(np.float32))

# Sum and dropletize image and get photon events
darkblank = np.zeros_like(dark)
nx = ny = 1024
gp = nx * ny - np.sum(mask)
print("Found {0} bad pixels".format(np.sum(mask)))
kbm = []
sp = range(1, 52, 10)
# for ii,i in enumerate(range(10)):
#     for jj,j in enumerate(range(0,data.shape[-1]-i,20)):
#         ind = np.arange(j,j+i+1)
#         if i>0:
#             sim = np.sum(data[:,:,ind], axis=-1)-dark*(i+1)*1.
#         else:
#             sim = np.squeeze(data[:,:,ind])*1.-dark*1.
#         sim = np.ma.array(sim, mask=mask)
#         imd = dropimgood(sim,darkblank,100*(i+1),1500,2155,75000*(i+1),1900,1024,1024)
#         pb = count(imd[1])
#         #pb = [np.sum(sim.astype(np.float32)) ,gp-sum(pb)] + pb
#         pb = [imd[0] ,gp-sum(pb)] + pb
#         print(pb)
#         kbm.append([pb, imd[:2]])
for ii, i in enumerate(range(10)):
    for jj, j in enumerate(range(0, data.shape[-1] - i, 20)):
        ind = np.arange(j, j + i + 1)
        tkb = int(np.sum(coun[ind]))
        pb = count(np.sort(pix[ind].ravel()))
        pb = [tkb, gp - sum(pb)] + pb
        print(pb)
        kbm.append([pb])

np.save(mf + "kbm_nn2_dts.npy", kbm)
# np.save(mf + 'mint.npy', mint)
