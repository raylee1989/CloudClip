import numpy as np
from submat import submat


def cutmat(x, n, xmin, xmax, ymin, ymax, sigma, radius):  #把n’X2型的矩阵x分割为n*n的格子
    # n = 5

    # x = np.arange(1, 21).reshape(10, 2)
    listmat = []
    # xstart = np.min(x[:, 0])
    # xend = np.max(x[:, 0])
    xitv = np.linspace(xmin, xmax, n + 1)

    # ystart = np.min(x[:, 1])
    # yend = np.max(x[:, 1])
    yitv = np.linspace(ymax, ymin, n + 1)  #从大到小分，对应画子图

    # print('in cutmat(),x:', x, '\n', 'xitv:', xitv, '\n', 'yitv:', yitv)
    xlen = xitv.__len__()
    ylen = yitv.__len__()
    m = 0
    tm = 0
    for i in range(ylen - 1):
        for j in range(xlen - 1):
            grid = submat(x, xitv[j], xitv[j + 1], yitv[i + 1], yitv[i])
            m = m + 1  #矩阵个数
            tm = tm + grid.shape[0]  #所有矩阵的总长，看是不是等于样本数
            print('in cutmat(),m,totalm,grid.len：', m, tm, grid.__len__())

            '''对所有矩阵找周围的点'''
            midpoint = np.mat([xitv[j] + (xitv[j + 1] - xitv[j]) / 2, yitv[i + 1] + (yitv[i] - yitv[i + 1]) / 2])
            for o in x:
                diffmat = np.mat(midpoint - o)
                wdis = np.exp(diffmat * diffmat.T / (-2.0 * sigma ** 2))
                if wdis > radius:
                    # print('in if')
                    # print('wdis:', wdis)
                    grid = np.row_stack((grid, o))
            listmat.append(grid)
            print('extend grid.len:', grid.__len__())

            ''' 只对空矩阵借周围的点，非空矩阵不管
            if grid.shape[0] == 0: #空矩阵，找midpoint周围的点
                print(xitv[j], xitv[j + 1], yitv[i], yitv[i + 1])
                print(j, j + 1, i, i + 1)
                midpoint = np.mat([xitv[j] + (xitv[j + 1] - xitv[j])/2, yitv[i + 1] + (yitv[i] - yitv[i + 1])/2])
                print('midpoint:', midpoint)
                for o in x:
                    diffMat = np.mat(midpoint - o)
                    wdis = np.exp(diffMat * diffMat.T / (-2.0 * sigma ** 2))
                    if wdis > radius:
                        # print('in if')
                        # print('wdis:', wdis)
                        grid = np.row_stack((grid, o))
                listmat.append(grid)
                print('grid.len:', grid.__len__())
            else: #非空的矩阵
                listmat.append(grid)
            '''
    return listmat

# return xitv, yitv
