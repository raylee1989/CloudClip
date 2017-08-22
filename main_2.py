from loadOneSample import loadOneSample
from load_eye_data import load_eye_data
from load_eye_data_2 import load_eye_data_2
import numpy as np
from fft_analysis import fft_analysis
from lwlr_2 import lwlr_2
import matplotlib.pyplot as pl
from p_value import p_value
from read_transfrom_distance import read_transfrom_distance
from plot_p_and_slope import polt_p_and_slope

n = 40
k = 5

maxtfdis = 70.00001
maxtflumi = 5.00001
mintfdis = 0.0
mintflumi = 0.00001
txedges = np.linspace(mintflumi, maxtflumi, n + 1)
tyedges = np.linspace(mintfdis, maxtfdis, n + 1)

xb = np.linspace(0.0, 5.0, n + 1)
xb = np.insert(xb, 0, -0.00001)
xb = np.append(xb, 5.00001)
yb = np.linspace(0, 70.0, n + 1)
yb = np.insert(yb, 0, -0.00001)
yb = np.append(yb, 70.00001)


def hpercent_and_dio(id_p):
    list_Hp = []
    list_dio = []
    for j in range(id_p.__len__()):
    # for j in range(2):
        d, diopter = loadOneSample(id_p, j)
        ind = np.where(d[:, 1] == -10)
        d = np.delete(d, ind[0], axis=0)
        dis = d[:, 1]  # matrix: n,1
        dis = np.log10(dis)  # 消除异方差性
        lumi = d[:, 2]  # matrix: n,1
        loglumi = np.log10(lumi)  # matrix: n,1
        filtered_dis = fft_analysis(dis, 1 / 5, 0.04 / 60, 0)  # ndarray:n,
        filtered_lumi = fft_analysis(loglumi, 1 / 5, 0.04 / 60, 0)
        filtered_dis = linear_mode.predict(np.mat(filtered_dis).T).flatten()
        filtered_dis = 10 ** filtered_dis
        H, xedges, yedges = np.histogram2d(filtered_lumi, filtered_dis, bins=[xb, yb])
        H = np.delete(H, 0, axis=0)
        H = np.delete(H, -1, axis=0)
        H = np.delete(H, 0, axis=1)
        H = np.delete(H, -1, axis=1)
        print('in hpercent_and_dio, H.sum():', H.sum())
        Hpercent = H / H.sum()
        list_Hp.append(Hpercent)
        list_dio.append(diopter.A1)
    return list_Hp, list_dio


def p_and_slope(list_Hp, list_dio):
    mat_slope = np.zeros((n, n))
    mat_intercept = np.zeros((n, n))
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
                slope, mat_y, intercept = lwlr_2(list_weight, list_x, list_y)
                print('x_len:',list_x.__len__())
                if (slope > 400) | (slope < -400):
                    pl.figure(p * 100 + q)
                    pl.xlim(0.0, 0.004)
                    pl.plot(list_x, mat_y)
                    pl.scatter(list_x, list_y)
                if slope == 0.0:
                    pvalue = 1.0
                    intercept = 0.0
                else:
                    pvalue = p_value(slope, list_x, list_y, mat_y)
                mat_pvalue[p, q] = pvalue
                mat_intercept[p, q] = intercept
                mat_slope[p, q] = slope
                print('list_x.__len__():, slope:, pvalue:, intercept', list_x.__len__(), slope, p, q, pvalue, intercept)
                if pvalue > 0.05:
                    mat_slope[p, q] = 0.0
                    mat_intercept[p, q] = 0.0
            else:
                mat_slope[p, q] = 0.0
                mat_intercept[p, q] = 0.0
                mat_pvalue[p, q] = 1.0
                print('list_x.__len__() < 30')

    valid_ind = np.where(tyedges < linear_mode.predict(15))[1][-1]
    mat_slope[:, 0:valid_ind] = 0
    mat_intercept[:, 0:valid_ind] = 0
    mat_pvalue[:, 0:valid_ind] = 1
    return mat_slope, mat_pvalue, mat_intercept


if __name__ == "__main__":
    linear_mode = read_transfrom_distance()
    print('70->%d, 50->%d, 15->%d, 0->%d' % (
        linear_mode.predict(70), linear_mode.predict(50), linear_mode.predict(15), linear_mode.predict(0)))

    ndarr_mse_model = np.zeros(10)  # 第j个元素存第j组的mse
    for j in range(1):
        test_id_p, id_p = load_eye_data_2(j)
        list_Hp, list_dio = hpercent_and_dio(id_p)
        mat_slope, mat_pvalue, mat_intercept = p_and_slope(list_Hp, list_dio)
        list_Hp_test, list_dio_test = hpercent_and_dio(test_id_p)  # 得到了测试集的x（百分比）和真实值y（屈光度）

        ndarr_one_fold = np.zeros(list_Hp_test.__len__())  # 每个元素存第j组中每个人的平均误差
        for i in range(list_Hp_test.__len__()):  # 使用第j组测试集中的第i个人的数据
            mat_test_dio = np.zeros((n, n))  # 把测试集代入拟合出的斜率和截距得到的屈光度，也就是预测值
            mat_error = np.zeros((n, n))  # 预测值减真实值，即第i个人的误差
            for p in range(n):
                for q in range(n):
                    if (list_Hp_test[i][p, q] > 0.0) & (mat_slope[p, q] != 0):
                        mat_test_dio[p, q] = list_Hp_test[i][p, q]*mat_slope[p, q]+mat_intercept[p, q]
                        mat_error[p, q] = np.abs(list_dio_test[i][0] - mat_test_dio[p, q])
            ndarr_one_fold[i] = mat_error.sum() / mat_error.nonzero()[0].__len__()  # 第i个人的平均误差，存在数组onefold的第i个位置
        ndarr_mse_model[j] = (ndarr_one_fold**2).sum() / list_Hp_test.__len__()
    print(ndarr_mse_model.mean())


    # polt_p_and_slope(txedges, tyedges, mat_pvalue, mat_slope)
