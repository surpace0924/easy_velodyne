#!/usr/bin/env python3

'''
sample code to extract point cloud from rosbag file and save it as csv file.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rosbag
import sensor_msgs.point_cloud2 as pc2

# detastructure
# https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/PointCloud2.html

def main():
    ###### param ######
    # レコードファイルのパス
    bag_filepath = os.path.join('assets', '2024-01-08-11-06-21.bag')

    # 出力ファイルのパス
    csv_filepath = os.path.join('assets', 'points.csv')
    
    # トピック名
    topic_name = '/velodyne_points'

    # 取り出す時刻（レコード開始時刻からの経過秒数）
    extract_time = 30.0

    # 取り出す空間範囲（原点から半径何[m]抽出するか）
    extract_range = 3.0
    ###################

    # velodyne_points トピックのみを抽出し，リスト化
    times = []
    msgs = []
    bag = rosbag.Bag(bag_filepath)
    for topic, msg, t in bag.read_messages():
        if topic == topic_name:
            time = float(f'{t.secs}.{t.nsecs}')
            times.append(time)
            msgs.append(msg)
    times = np.array(times)

    print('レコード時間: ', times[-1] - times[0], '[sec]')
    print('レコード開始時刻: ', times[0])
    print('レコード終了時刻: ', times[-1])

    # レコード開始時刻を基準とした時刻に変換
    times -= times[0]

    # 指定時刻付近のデータを抽出
    idx = np.where(times > extract_time)[0][0]
    print('指定時刻: ', times[idx])
    points = pointcloud2xyz(msgs[idx])

    #範囲内の点群のみを抽出
    points = points[np.linalg.norm(points, axis=1) < extract_range]

    # プロット
    plot_points(points)

    # ファイル出力
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    df.to_csv(csv_filepath, index=False)


# PointCloud2メッセージをnumpy配列に変換
# @param msg: PointCloud2メッセージ
def pointcloud2xyz(msg):
    pc_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    pc_array = np.array(list(pc_data))
    return pc_array


# 点群をプロット
# @param points: 点群（N×3のnumpy配列）
def plot_points(points):
    fig = plt.figure()
    
    # 設定
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
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
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    
    # 視点
    ax.view_init(elev=0)
    ax.view_init(azim=180+45)

    plt.show()

    
if __name__ == '__main__':
    main()
