#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation
from sklearn.linear_model import LinearRegression


class Main:
    def __init__(self):
        ###### param ######
        # レコードファイルのパス
        csv_filepath = os.path.join('assets', 'points.csv')

        # 標尺で測定した高さ [m]
        self.height_rod = 0.346

        # 路面領域のy座標 [m]
        self.ground_y = [-3.0, 0.0]

        # 雪堤領域のx座標 [m]
        self.snowbank_x = [0.05, 0.15]
        ###################

        # レコードファイルの読み込み
        df = pd.read_csv(csv_filepath)
        points = df.values    
        
        # y座標マイナスの箇所が地面
        points_ground = points[points[:,1]>self.ground_y[0]]
        points_ground = points[points[:,1]<self.ground_y[1]]

        # データの平均が原点になるように平行移動
        offset = points_ground.mean(axis=0)
        points_ground -= offset
        points -= offset

        # 平面にフィッティング
        fitted_normal_vector = self.fit_plane_to_point_cloud(points_ground)

        # 平面がxy平面に平行になるように点を回転させる (なんかxyの方向が逆になるので反転)
        rotated_points = self.rotate_points_to_xy_plane(points, fitted_normal_vector)
        rotated_points[:, 0] *= -1.0
        rotated_points[:, 1] *= -1.0

        # データのxy座標の平均が(0, 0)になるように平行移動
        offset = rotated_points.mean(axis=0)
        offset[2] = 0
        rotated_points -= offset

        # xの範囲を限定
        limited_points = rotated_points[rotated_points[:,0]>self.snowbank_x[0]]
        limited_points = limited_points[limited_points[:,0]<self.snowbank_x[1]]

        # 点の中央値を計算
        median = np.median(limited_points, axis=0)        
        self.height_lidar = median[2]

        print('Height by leveling rod: {:.3f} m'.format(self.height_rod))
        print('Height by LiDAR:        {:.3f} m'.format(self.height_lidar))
        print('Difference:             {:.3f} m'.format(self.height_rod - self.height_lidar))

        # グラフ描画
        self.plot(rotated_points)


    # 点群から平面をフィッティング
    # @param points: 点群（N×3のnumpy配列）
    # @return 平面の法線ベクトル
    @staticmethod
    def fit_plane_to_point_cloud(points):
        # 線形回帰モデルを作成し、平面の係数をフィット
        model = LinearRegression(fit_intercept=False)
        model.fit(points[:, :2], points[:, 2])

        # 平面の法線ベクトルとバイアス（切片）を取得
        normal_vector = np.append(model.coef_, -1.0)

        return normal_vector / np.linalg.norm(normal_vector)


    # 平面がxy平面に平行になるように点を回転させる
    # @param points: 点群（N×3のnumpy配列）
    # @param rotated_points: 回転後の点群
    @staticmethod
    def rotate_points_to_xy_plane(points, normal_vector):
        # 平面の法線ベクトルから回転行列を生成
        rotation_matrix = Rotation.align_vectors([[0, 0, -1]], [normal_vector])[0].as_matrix()

        # 点を回転させる
        rotated_points = np.dot(points, rotation_matrix.T)

        return rotated_points


    # 3Dプロット
    def plot(self, points):
        fig = plt.figure()

        # 描画設定
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.99)
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.size'] = 14
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.linewidth'] = 0.3
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['legend.handlelength'] = 1.0
        plt.rcParams['legend.labelspacing'] = 0.5
        plt.rcParams['figure.figsize'] = [6.5, 4.8]
        plt.rcParams['figure.dpi'] = 200
        plt.rcParams['figure.subplot.left'] = 0.1
        plt.rcParams['figure.subplot.bottom'] = 0.12
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.top'] = 0.95

        # 3Dプロット        
        ax = fig.add_subplot(111, projection='3d')
        colors = points[:,2]    
        scatter = ax.scatter(points[:,0], points[:,1], points[:,2], s=0.1, c=colors, cmap='viridis')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-1, 3)
        ax.view_init(elev=-45)
        ax.view_init(azim=-90)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Z-axis value')
        plt.show()

        # y軸方向に射影したグラフ
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(points[:,0], points[:,1], s=0.1, c=colors, cmap='viridis')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        plt.gca()
        plt.show()
        
        # z軸方向に射影したグラフ
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(points[:,0], points[:,2], s=0.1, c=colors, cmap='viridis')
        ax.hlines(self.height_rod, -0.5, 1, linestyles='dashed', linewidths=1.5, colors='red', label='by leveling rod')
        ax.hlines(self.height_lidar, -0.5, 1, linewidths=1.5, colors='orange', label='by LiDAR')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Z [m]')
        ax.set_xlim(-0.5, 1)
        ax.set_ylim(-0.1, 0.5)
        plt.legend(fancybox=False, edgecolor='black', facecolor='white', framealpha=1.0)
        plt.gca()
        plt.show()

    
if __name__ == '__main__':
    Main()
