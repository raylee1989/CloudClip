import os.path
import numpy as np
import scipy.io as sio
from fft_analysis import fft_analysis
from read_transfrom_distance import read_transfrom_distance
import matplotlib.pyplot as pl
from delinvalid import delinvalid
from split_and_patch_sequence_exlude_presleep import split_and_patch_sequence_exlude_presleep
import pickle

local = 'GZ'
n = 128
maxtfdis = 60.00
maxtflumi = 5.00
mintfdis = 1.0
mintflumi = 0.0
txedges = np.linspace(mintflumi, maxtflumi, n + 1)
tyedges = np.linspace(mintfdis, maxtfdis, n + 1)

xb = np.linspace(mintflumi, maxtflumi, n+1)

yb = np.linspace(mintfdis, maxtfdis, n+1)

H_total = np.zeros((n,n))

linear_mode = read_transfrom_distance()
rootdir = 'D:\data\cloudclipdata'  # 指明被遍历的文件夹

for parent, dirnames, filenames in os.walk(rootdir):
    for i in range(filenames.__len__()):
        if local in filenames[i]:
            print(filenames[i])
            data = sio.loadmat(parent + '\\' + filenames[i])
            d = np.mat(data['a'])
            d_patch = split_and_patch_sequence_exlude_presleep(d)
            ind = np.where(d_patch[:, 1] == -10)
            d_patch = np.delete(d_patch, ind[0], axis=0)

            dis = d_patch[:, 1]
            dis = np.log10(dis)
            # ind = np.where(np.isnan(dis))
            # dis[ind[0]] = -10
            lumi = d_patch[:, 2]
            loglumi = np.log10(lumi)
            # ind = np.where(np.isnan(loglumi))
            # loglumi[ind[0]] = -10

            filtered_dis = fft_analysis(dis, 1 / 5, 0.04 / 60, 0)
            filtered_lumi = fft_analysis(loglumi, 1 / 5, 0.04 / 60, 0)

            # filtered_dis = linear_mode.predict(np.mat(filtered_dis).T).flatten()
            # filtered_dis[filtered_dis > 50] = 50


            # for j in range(filtered_dis.__len__()):
            #     if filtered_dis[j] < 50:
            #         filtered_dis[j] = linear_mode.predict(filtered_dis[j])
            #         print(filtered_dis[j])
            #     else:
            #         filtered_dis[j] = 50

            filtered_dis = linear_mode.predict(np.mat(filtered_dis).T).flatten()
            filtered_dis = 10 ** filtered_dis

            filtered_dis[filtered_dis < 15] = 0

            H, xedges, yedges = np.histogram2d(filtered_lumi, filtered_dis, bins=[xb, yb])
            H_total = H_total + H

log_H = np.log10(H_total)
ind = np.where(np.isinf(log_H))
for i in range(ind[0].__len__()):
    log_H[ind[0][i], ind[1][i]] = 0
#
# for i in range(log_H.shape[0]):
#     for j in range(log_H.shape[0]):
#         if log_H[i, j] != 0:
#             print(log_H[i,j])


output = open('log_H.pkl', 'wb')
pickle.dump(log_H, output)
output.close()

pl.figure(44)
pl.pcolor(txedges, tyedges, log_H.T, cmap='jet')
pl.colorbar()
pl.xlabel('Lg(lux)')
pl.ylabel('Distance(cm)')
fig = pl.gcf()
fig.set_size_inches(10.24, 7.68)

pl.savefig(local+'dis_trans_no15.png', dpi=300)

# pl.show()