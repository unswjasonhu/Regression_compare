import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
import colormaps as cmps
from matplotlib import cm
from matplotlib import style
style.use("ggplot")
import pandas as pd
from math import sqrt


def scipy_idw(x, y, z, xi, yi):
    interp = Rbf(x, y, z, function='linear')
    return interp(xi, yi)

def read_data(model):
    df = pd.read_csv('%s_populated_estimates.csv'%model, sep=',', header= None)
    return df

def cal_mae(model):
    mae = 0
    mat = read_data(model).as_matrix()
    for i in xrange(100):
        for j in xrange(100):
            mae += abs(mat[i][j]-grid_original[i,j])
    return mae/10000

def cal_rmse(model):
    rmse = 0
    mse = 0
    mat = read_data(model).as_matrix()
    for i in xrange(100):
        for j in xrange(100):
            mse += (mat[i][j]-grid_original[i,j]) ** 2
    rmse = sqrt(mse / 10000)
    return rmse


def main():
    #Draw original
    df_original = pd.read_csv('original.csv',delimiter=',')
    row_original = df_original['grid_location_row']
    col_original = df_original['grid_location_col']
    val_original = df_original['co_original']
    xi = np.linspace(row_original.min(), row_original.max(), 100)
    yi = np.linspace(col_original.min(), col_original.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    xi, yi = xi.flatten(), yi.flatten()
    grid = scipy_idw(row_original,col_original ,val_original, xi, yi)
    grid_original = grid.reshape((100, 100))
    global grid_original

    vmax = 25

    fig, axes = plt.subplots(nrows=2,ncols=4)
    fig.set_size_inches(20,10)
    #fig.subplots_adjust(right =0.8)

    ax1 = plt.subplot(2,4,1)
    im1 = plt.imshow(read_data('SVR'),cmap=cm.jet,vmin=0,vmax=vmax)
    svr_mae = cal_mae('SVR')
    svr_rmse = cal_rmse('SVR')
    print svr_mae, svr_rmse, read_data('SVR').values.max()
    plt.title('(a) Support Vector Regression' ,fontsize = 18,y = -0.15,)
    ax1.annotate('M5 Tunnel', xy=(70,80), xycoords='data',  xytext=(-30, -40), textcoords='offset points', fontsize=16,color='white',
             arrowprops=dict(arrowstyle="->", linewidth = 1,color ='white',connectionstyle="arc3,rad=.2"))
    ax1.annotate('CBD', xy=(80,30), xycoords='data',  xytext=(-70, 20), textcoords='offset points', fontsize=16,color='white',
             arrowprops=dict(arrowstyle="->", linewidth = 1,color ='white',connectionstyle="arc3,rad=.2"))
    ax1.annotate('Eastern Distributor\n Motorway', xy=(90,50), xycoords='data',  xytext=(-150, 0), textcoords='offset points', fontsize=16,color='white',
             arrowprops=dict(arrowstyle="->", linewidth = 1,color ='white',connectionstyle="arc3,rad=.2"))
    plt.axis("off")

    ax2 = plt.subplot(2,4,2)
    im2 = plt.imshow(read_data('DTR'),cmap=cm.jet,vmin=0,vmax=vmax)
    dtr_mae = cal_mae('DTR')
    dtr_rmse = cal_rmse('DTR')
    print dtr_mae, dtr_rmse, read_data('DTR').values.max()
    plt.title('(b) Decision Tree Regression' ,fontsize = 18,y = -0.15)
    plt.axis("off")
    plt.xlabel('(b)')

    ax3 = plt.subplot(2,4,3)
    im3 = plt.imshow(read_data('RFR'),cmap=cm.jet,vmin=0,vmax=vmax)
    rfr_mae = cal_mae('RFR')
    rfr_rmse = cal_rmse('RFR')
    print rfr_mae, rfr_rmse, read_data('RFR').values.max()
    plt.title('(c) Random Forest Regression' ,fontsize = 18,y = -0.15)
    plt.axis("off")
    plt.xlabel('(d)')

    ax4 = plt.subplot(2,4,4)
    im4 = plt.imshow(read_data('XGB'),cmap=cm.jet,vmin=0,vmax=vmax)
    xgb_mae = cal_mae('XGB')
    xgb_rmse = cal_rmse('XGB')
    print xgb_mae, xgb_rmse, read_data('XGB').values.max()
    plt.title('(d) Extreme Gradient Boosting' ,fontsize = 18,y = -0.15)
    plt.axis("off")
    plt.xlabel('(e)')

    ax5 = plt.subplot(2,4,5)
    im5 = plt.imshow(read_data('MLP'),cmap=cm.jet,vmin=0,vmax=vmax)
    mlp_mae = cal_mae('MLP')
    mlp_rmse = cal_rmse('MLP')
    print mlp_mae, mlp_rmse, read_data('MLP').values.max()
    plt.title('(e) Multi-Layer Perceptron' ,fontsize = 18,y = -0.15)
    plt.axis("off")
    plt.xlabel('(f)')

    ax6 = plt.subplot(2,4,6)
    im6 = plt.imshow(read_data('LR'),cmap=cm.jet,vmin=0,vmax=vmax)
    lr_mae = cal_mae('LR')
    lr_rmse = cal_rmse('LR')
    print lr_mae, lr_rmse, read_data('LR').values.max()
    plt.title('(f) Linear Regression' ,fontsize = 18,y = -0.15)
    plt.axis("off")
    plt.xlabel('(h)')

    ax7 = plt.subplot(2,4,7)
    im7 = plt.imshow(read_data('ABR'),cmap=cm.jet,vmin=0,vmax=vmax)
    abr_mae = cal_mae('ABR')
    abr_rmse = cal_rmse('ABR')
    print abr_mae, abr_rmse, read_data('ABR').values.max()
    plt.title('(g) Adaptive Boosting' ,fontsize = 18,y = -0.15)
    plt.axis("off")
    plt.xlabel('(g)')

    ax8 = plt.subplot(2,4,8)
    im8 =plt.imshow(grid_original,cmap=cm.jet,vmin=0,vmax=vmax)
    plt.title('(h) Original CO surface using\n IDW interpolation model',fontsize = 18,y = -0.2)
    print grid_original.max()
    ax8.get_xaxis().set_visible(False)
    ax8.get_yaxis().set_visible(False)




    cbar_ax=fig.add_axes([0.92,0.2,0.03,0.6])
    cbar = fig.colorbar(im1,cax=cbar_ax)
    cbar.ax.tick_params(labelsize=28)
    plt.savefig('surface_compare.png',format='png',dpi=200)
    #plt.show()

if __name__ == "__main__":
    main()