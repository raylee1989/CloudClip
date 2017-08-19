from loadOneSample import loadOneSample
from loaddata import loaddata
from load_eye_data import load_eye_data
import numpy as np
from fft_analysis import fft_analysis
from lwlr_2 import lwlr_2
import matplotlib.pyplot as pl
from p_value import p_value
from read_transfrom_distance import read_transfrom_distance
from load_one_sample_no_patch import load_one_sample_no_patch
from mpl_toolkits.mplot3d import Axes3D

n = 40
list_Hp = []
list_dio = []

id_p = load_eye_data()

maxtfdis = 70.00001
maxtflumi = 5.00001
mintfdis = 0.0
mintflumi = 0.00001
txedges = np.linspace(mintflumi, maxtflumi, n + 1)
tyedges = np.linspace(mintfdis, maxtfdis, n + 1)

xb = np.linspace(0.0, 5.0, n+1)
xb = np.insert(xb, 0, -0.00001)
xb = np.append(xb, 5.00001)
yb = np.linspace(0, 70.0, n+1)
yb = np.insert(yb, 0, -0.00001)
yb = np.append(yb, 70.00001)

linear_mode = read_transfrom_distance()
print('70->%d, 50->%d, 15->%d, 0->%d' % (linear_mode.predict(70),linear_mode.predict(50),linear_mode.predict(15),linear_mode.predict(0)))
# y = np.linspace(1,80,80)
# # y = np.arange(0, 90)
# x = np.linspace(1,80,80)
# for i in range(0,x.shape[0]):
#     y[i] = linear_mode.predict(np.log10(x[i]))
# pl.plot(x, 10**y)

# H_total = np.zeros((n,n))

for j in range(id_p.__len__()):
# for j in range(1):
    d, diopter = loadOneSample(id_p, j)

    ind = np.where(d[:, 1] == -10)
    d = np.delete(d, ind[0], axis=0)

    dis = d[:, 1]  # matrix: n,1

    # 消除异方差性
    dis = np.log10(dis)

    lumi = d[:, 2]  # matrix: n,1
    loglumi = np.log10(lumi)  # matrix: n,1


    # figure = pl.figure(40)
    # ax = figure.add_subplot(111, projection='3d')
    # z = np.linspace(0, 6, 1000)
    # r = 1
    # x = r * np.sin(np.pi * 2 * z)
    # y = r * np.cos(np.pi * 2 * z)
    # ax.plot(x, y, z)
    # ax.plot(dis.getA1(), lumi.getA1(), np.array(hour_minute))
    # pl.show()
    

    # pl.figure(41)
    # pl.grid()
    # pl.scatter(loglumi.getA1(), dis.getA1(), 5)

    filtered_dis = fft_analysis(dis, 1 / 5, 0.04 / 60, 0)  # ndarray:n,
    filtered_lumi = fft_analysis(loglumi, 1 / 5, 0.04 / 60, 0)

    # pl.figure(4)
    # pl.grid()
    # pl.scatter(filtered_lumi, filtered_dis, 5)

    # pl.figure(5+j)
    # pl.xlim(mintflumi, maxtflumi)
    # pl.ylim(mintfdis, maxtfdis)

    filtered_dis = linear_mode.predict(np.mat(filtered_dis).T).flatten()
    filtered_dis = 10 ** filtered_dis

    # pl.figure(42)
    # pl.grid()
    # pl.scatter(filtered_lumi, filtered_dis, 5)

    H, xedges, yedges = np.histogram2d(filtered_lumi, filtered_dis, bins=[xb, yb])
    H = np.delete(H, 0, axis=0)
    H = np.delete(H, -1, axis=0)
    H = np.delete(H, 0, axis=1)
    H = np.delete(H, -1, axis=1)

    print('H.sum():', H.sum())
    # if H.sum() == 0.0:
    #     pl.figure(1)
    #     pl.scatter(filtered_lumi, filtered_dis)

    Hpercent = H / H.sum()
    # Hpercent = H / d.shape[0]
    # pl.pcolor(txedges, tyedges, Hpercent.T, cmap='jet')
        # pl.xlim(mintflumi, maxtflumi)
        # pl.ylim(mintfdis, maxtfdis)
    # pl.colorbar()
        # pl.scatter(filtered_lumi, filtered_dis, 5)
    list_Hp.append(Hpercent)
    list_dio.append(diopter.A1)


k = 5
mat_slope = np.zeros((n, n))
mat_pvalue = np.zeros((n, n))
for p in range(n):
    for q in range(n):

        mat_weights_pq = np.zeros((n, n))
        for i in range(n):
            for j in range(n):  # 点p，q与其他点的权的矩阵
                mat_weights_pq[i, j] = np.exp(-0.5 * ((p - i) ** 2 + (q - j) ** 2) / (k ** 2))

        list_x = []
        list_weight = []
        list_y = []
        arr_ind_p, arr_ind_q = np.where(mat_weights_pq > 0.1)  # 返回符合条件的索引
        for ind_h in range(list_Hp.__len__()):
            for ind_ind in range(arr_ind_p.shape[0]):
                if list_Hp[ind_h][arr_ind_p[ind_ind], arr_ind_q[ind_ind]] > 0.0:
                    list_x.append(list_Hp[ind_h][arr_ind_p[ind_ind], arr_ind_q[ind_ind]])
                    list_weight.append(mat_weights_pq[arr_ind_p[ind_ind], arr_ind_q[ind_ind]])
                    list_y.append(list_dio[ind_h])
        if list_x.__len__() > 30:
            slope, mat_y = lwlr_2(list_weight, list_x, list_y)
            mat_slope[p, q] = slope
            # print('list_x.__len__():, slope:', list_x.__len__(), slope, p, q)
            if (slope > 400) | (slope < -400):
                pl.figure(p*100+q)
                pl.xlim(0.0, 0.004)
                pl.plot(list_x, mat_y)
                pl.scatter(list_x, list_y)
            if slope == 0.0:
                # pl.figure(p * 100 + q)
                # pl.title('slope==0')
                # pl.scatter(list_x, list_y)
                pvalue = 1.0
            else:
                pvalue = p_value(slope, list_x, list_y, mat_y)
            mat_pvalue[p, q] = pvalue

            print('list_x.__len__():, slope:, pvalue:', list_x.__len__(), slope, p, q, pvalue)
            if pvalue > 0.05:
                mat_slope[p, q] = 0.0
        else:
            mat_slope[p, q] = 0.0
            mat_pvalue[p, q] = 1.0
            print('list_x.__len__() < 30')


valid_ind = np.where(tyedges < linear_mode.predict(15))[1][-1]
mat_slope[:, 0:valid_ind] = 0
mat_pvalue[:, 0:valid_ind] = 1


pl.figure(100, figsize=[16, 10])
pl.axes([0.03, 0.2, 0.5, 0.6])
pl.pcolor(txedges, tyedges, mat_pvalue.T, cmap='jet')
pl.xlabel('log(lumi)')
pl.ylabel('distance')
pl.colorbar()
pl.clim(0, 0.05)
pl.axes([0.53, 0.2, 0.5, 0.6])
pl.pcolor(txedges, tyedges, mat_slope.T, cmap='jet')
pl.xlabel('log(lumi)')
pl.ylabel('distance')
pl.colorbar()

pl.figure(101, figsize=[16, 10])
pl.axes([0.03, 0.2, 0.5, 0.6])
pl.pcolor(txedges, tyedges, mat_pvalue.T, cmap='jet')
pl.clim(0, 0.05)
pl.xlabel('log(lumi)')
pl.ylabel('distance')
pl.colorbar()
pl.axes([0.53, 0.2, 0.5, 0.6])
pl.pcolor(txedges, tyedges, mat_slope.T, cmap='jet')
pl.clim(-100, 100)
pl.xlabel('log(lumi)')
pl.ylabel('distance')
pl.colorbar()

pl.figure(102)
pl.axes([0.03, 0.2, 0.5, 0.6])
pl.pcolor(txedges, tyedges, mat_pvalue.T, cmap='jet')
pl.clim(0, 0.05)
pl.xlabel('log(lumi)')
pl.ylabel('distance')
pl.colorbar()
pl.axes([0.53, 0.2, 0.5, 0.6])
pl.pcolor(txedges, tyedges, mat_slope.T, cmap='jet')
pl.clim(-150, 150)
pl.xlabel('log(lumi)')
pl.ylabel('distance')
pl.colorbar()

pl.show()


