#'./data4GAN/0fullmaps/50/'のデータ（1mメッシュの都市全体の空間情報）をもとに、pix2pixで用いるサイズに変換


# 3_GAN_makeFullmap4GanDataset.py

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import stats
import os
import gc

padding = 500 #全体マップ作成時と同じ値にする
hb = 12.5   #基地局アンテナ高[m]
hm = 1.5    #移動局アンテナ高[m]
min_xy = np.load("./data/map_padding/min_xy.npy")#座標の位置合わせ用

'''
base_stationなんちゃらの名前付け辺りのためだけにエクセルデータをロード(;o;)
'''
df = pd.read_excel("./data/mapdata.xlsx") #エクセルから建物データを取得
x_map = df['x[m]'].values #建物データの座標x
y_map = df['y[m]'].values #建物データの座標y

'''
高さだけ先行して作成する。whileの中でやるとバグってしまった
'''
map_0 = np.load("./data4GAN/0fullmaps/50/535_582/map.npy")#全体マップのロード
map_0 = map_0[::-1]#処理しやすいように反転

#都市全体の空間情報画像のy軸の座標を定義(10mメッシュ)
y0 = np.arange(padding+181, map_0.shape[0]-padding-181, 10)
#都市全体の空間情報画像のx軸の座標を定義(10mメッシュ)
x0 = np.arange(padding+181, map_0.shape[1]-padding-181, 10)

# 高さ78, 大きさ105で作成。(10mメッシュの場合のサイズ)
h_map = np.zeros((78,105, 1)) # 都市全体の建物高の情報を格納する配列(ndarray)
for i in tqdm(range(len(x0))):
            for j in (range(len(y0))):
                # 10mごとの要素を全体マップ(map_0)から取り出して、h_mapに格納
                h_map[j,i] = map_0[y0[j],x0[i],0]

h_max=h_map.max()#建物高の最大値を取り出す。（今後の処理のため）

'''
基地局の位置に応じた空間情報を作成していく
'''
yyy=0 #基地局のy軸座標のカウント
while yyy*50 <= max(y_map)-min(y_map): #50m間隔なのでy軸のカウントを50倍
    xxx=0 #基地局のx軸座標のカウント
    while xxx*50 <= max(x_map)-min(x_map): #50m間隔なのでx軸のカウントを50倍
        
        '''
        保存先の指定部分
        base_position: 基地局の座標の文字列(全体マップのロードや、保存するときのファイル名に用いる。)
        '''
        # 保存ファイル名のために文字列を作成する部分
        base_y = int(max(y_map)-(yyy*50)) # 基地局のy軸座標
        base_x = int(max(x_map)-(xxx*50)) # 基地局のx軸座標
        base_position = str(base_y)+'_'+str(base_x) #フォルダ名、座標(y,x)を名前とする。
        print(base_position)
        new_dir_path = './data4GAN/3fullmaps' # 保存先を指定

        '''
        都市全体の空間情報を格納する配列を作成
        '''
        # 高さ78, 大きさ105で作成。(10mメッシュの場合のサイズ)
        target_map = np.zeros((78,105, 3)) # 都市全体の空間情報を格納する配列(ndarray)

        '''
        基地局座標に合わせた全体マップのロードをする。
        '''
        map_ = np.load("./data4GAN/0fullmaps/50/"+base_position+"/map.npy")#全体マップ
        map_ = map_[::-1]  #処理しやすいように反転
        d_max=map_[:,:,1].max() #距離の最大値を取り出す

        '''
        各メッシュのx軸とy軸の座標要素
        '''
        y = np.arange(padding+181, map_.shape[0]-padding-181, 10)  
        x = np.arange(padding+181, map_.shape[1]-padding-181, 10)

        '''
        空間情報を格納する配列target_mapに、必要な情報を格納
        '''
        for i in tqdm(range(len(x))): #メッシュのx軸要素
            for j in (range(len(y))): #メッシュのy軸要素

                #target_mapの1チャンネルに距離情報を格納
                # 距離情報が1より小さいとき、0を格納
                if map_[y[j],x[i],1] < 1: target_map[j,i,1] = 0
                # その他の場合、距離情報をlogに変換。そして値を[0,255]にスケーリング
                else: target_map[j,i,1] = ((np.log10(map_[y[j],x[i],1]))/np.log10(d_max))*255
                #target_mapの0チャンネルに建物高情報を格納。値は[0,255]にスケーリング
                target_map[j,i,0] = (h_map[j,i]/h_max)*255

        # 画像で保存(確認用)
        # target_map = target_map[::-1]
        # map_img = target_map
        # filename = str(i*10+181)+'_'+str(j*10+181)
        # pilImg = Image.fromarray(np.uint8(map_img[:,:,0]))
        # pilImg.save(new_dir_path+'/'+ base_position + '_0.png')
        # pilImg = Image.fromarray(np.uint8(map_img[:,:,1]))
        # pilImg.save(new_dir_path+'/'+ base_position + '_1.png')
        
        # 都市全体の空間情報をバイナリで保存
        np.save(new_dir_path+'/'+ base_position, map_img)

        del target_map,map_img,map_
        gc.collect()
        xxx+=1
    yyy+=1
