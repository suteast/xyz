#都市全体の各メッシュ（10Mメッシュ）におけるDCNN入力マップを作成する。
#基地局の位置に応じて、各メッシュでのDCNN入力マップを作成します。



# 1_GAN_makeDataset_padding_NotZscore.py
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import stats
import os

size = 256  #一辺の長さ、これをもとに正方形のデータを作る
padding = 500 #全体マップでのpaddingと同じ値にする
hb = 12.5   #基地局アンテナ高[m]
hm = 1.5    #移動局アンテナ高[m]
min_xy = np.load("./data/map_padding/min_xy.npy")#座標の位置合わせ用

'''
base_stationなんちゃらの名前付け辺りのためだけにエクセルデータをロード(;o;)
'''
df = pd.read_excel("./data/mapdata.xlsx")#エクセルから建物データを取得
x_map = df['x[m]'].values #建物データの座標x
y_map = df['y[m]'].values #建物データの座標y

# 入力マップは、建物の高さ/送受信間距離/受信点を中心とした波紋型の距離、の3チャンネル
test_map1 = np.zeros((size, size, 3))#Testマップ1
test_map2 = np.zeros((size, size, 3))#Testマップ2
map_img = np.zeros((size, size, 3))#画像化用


def make_minimap(BS, MS, map_, size):
    '''
    make_minimap(): MSのローカルx座標がBS方向になるように回転させて切り取る関数
    BS: 送信座標ndarray(x,y)
    MS: 受信座標ndarray(x,y)
    map_: 切り取る前の全体マップ
    size: 切り取るサイズ(例: size=128だと、128×128の正方形を切り抜く)
    '''
    minimap_ = np.zeros((size, size, 3))
    level = np.array([MS[0]+1000000, MS[1]])# MSのx成分を+したもの
    a = BS - MS # 送受信間ベクトル
    b = level - MS # MSからx軸プラス側に水平に伸ばしたベクトル
    cos_theta = np.inner(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))
    theta = np.arccos(cos_theta)

    for i in range(size):
        for j in range(size):
            xi = MS[0]-int(size/2)+j
            yi = MS[1]-int(size/2)+i

            if MS[1]>BS[1]:
                x = int(MS[0] + (xi-MS[0])*np.cos(-theta) - (yi-MS[1])*np.sin(-theta))
                y = int(MS[1] + (xi-MS[0])*np.sin(-theta) + (yi-MS[1])*np.cos(-theta))
            else:
                x = int(MS[0] + (xi-MS[0])*np.cos(theta) - (yi-MS[1])*np.sin(theta))
                y = int(MS[1] + (xi-MS[0])*np.sin(theta) + (yi-MS[1])*np.cos(theta))
            # print(theta)

            minimap_[i,j,1] = map_[y,x,1]   #BSからの距離
            
            ripple = MS - np.array([x, y])              #MSからの距離その1
            minimap_[i,j,2] = np.linalg.norm(ripple)    #MSからの距離その2

            h = map_[y,x,0]     #建物高
            d0 = minimap_[i,j,1]    #BSからの距離
            d1 = minimap_[i,j,2]    #MSからの距離
            h_ = (hb-hm)*d1/(d1+d0) #電波の射線の高さの計算その1
            h0 = hm+h_              #電波の射線の高さの計算その2
            delta_h = h - h0        #建物高から射線高を引く、見通しがある場合はマイナス値になるはず
            minimap_[i,j,0] = delta_h
            
            # if map_[y,x,1] < 1:    minimap_[i,j,1] = 0
            # else:   minimap_[i,j,1] = np.log10(map_[y,x,1])     #Log10化
            
    return minimap_

'''
以下メインの処理
* 基地局の位置に応じた全体の環境に応じて、DCNN用のデータセットを作成
* 受信座標を中心に256*256トリミング
* 見て確認するように画像化させたものを保存
* トリミングした行列をリストに追加
* npz形式でファイル保存(learningでデータセットとして使う)
'''
yyy=0#基地局のy軸座標のカウント
while yyy*50 <= max(y_map)-min(y_map):#50m間隔なのでy軸のカウントを50倍
    xxx=0#基地局のx軸座標のカウント
    if xxx==28: xxx=0#右端に到達したらx成分カウントをリセット、50m間隔なのでカウント28で折り返し
    while xxx*50 <= max(x_map)-min(x_map):#50m間隔なのでy軸のカウントを50倍

        '''
        保存先の指定部分
        base_position: 基地局の座標の文字列(全体マップのロードや、保存するときのファイル名に用いる。)
        '''
        base_y = int(max(y_map)-(yyy*50)) # 基地局のy軸座標
        base_x = int(max(x_map)-(xxx*50)) # 基地局のx軸座標
        base_position = str(base_y)+'_'+str(base_x) #フォルダ名、座標(y,x)を名前とする。
        print(base_position)
        new_dir_path = './data4GAN/1dataset/'+base_position # 保存先を指定
        os.mkdir(new_dir_path)# 保存先を作成
        os.mkdir(new_dir_path+'/test_imgs') # 確認用画像の保存先を作成
        os.mkdir(new_dir_path+'/test_imgs/distance') # 確認用画像の保存先を作成

        BS = np.array([base_y-min_xy[0]+padding, base_x-min_xy[1]+padding])#基地局の座標(x,y)
        map_ = np.load("./data4GAN/0fullmaps/50/"+base_position+"/map.npy")#全体マップのロード
        map_ = map_[::-1]  #マップの反転(numpyで使いやすくするため)

        # 各メッシュのx軸とy軸の座標要素
        y = np.arange(padding+181, map_.shape[0]-padding-181, 10)  
        x = np.arange(padding+181, map_.shape[1]-padding-181, 10)
        x1, x2 = np.array_split(x, 2) #デカすぎるのでx成分を分割

        # npzとしてパッケージングする用の配列、デカすぎるので分割
        test_maps1 = []
        test_maps2 = []

        # 前半(メッシュのx軸要素を分割しているので、分割して処理)
        for i in tqdm(range(len(x1))): #メッシュのx軸要素
            for j in tqdm(range(len(y))): #メッシュのy軸要素
                # p_test = np.append([x0[i], x1[j]])

                #受信座標(x, y)を定義
                MS = np.array([x1[i], y[j]])
                #入力マップ作成
                minimap = make_minimap(BS, MS, map_, size)
                test_map = minimap[::-1]

                # 画像保存(確認用)
                map_img = test_map
                filename = str(i*10+181)+'_'+str(j*10+181)
                # pilImg = Image.fromarray(np.uint8(map_img[:,:,0]))
                # pilImg.save('./dataset/te/'+str(size)+'/確認用/test_imgs/buildings/'+ filename + '.png')
                pilImg = Image.fromarray(np.uint8(map_img[:,:,1]))
                pilImg.save(new_dir_path+'/test_imgs/distance/'+ filename + '.png')
                # pilImg = Image.fromarray(np.uint8(map_img[:,:,2]))
                # pilImg.save(new_dir_path+'/test_imgs/distance/'+ filename + '.png')
                # pilImg = Image.fromarray(np.uint8(map_img[:,:,:2]))
                # pilImg.save('./data/'+str(size)+'/確認用/test_imgs/'+ filename + '.png')

                test_maps1.append(test_map.tolist()) #入力マップをリストに追加

        # ndarray変換
        # test_maps1は入力マップの一覧が入ったリスト
        test_maps1 = np.array(test_maps1)
        # 保存
        np.savez(new_dir_path+'/test_1.npz', test_maps1)

        # 後半(同じ処理)
        for i in tqdm(range(len(x2))):
            for j in tqdm(range(len(y))):
                # p_test = np.append([x0[i], x1[j]])
                #受信座標(x, y)
                MS = np.array([x2[i], y[j]])

                #ミニマップ作成
                minimap = make_minimap(BS, MS, map_, size)
                test_map = minimap[::-1]

                # 画像保存
                # map_img = test_map
                # filename = str(i*10+181)+'_'+str(j*10+181)
                # pilImg = Image.fromarray(np.uint8(map_img[:,:,0]))
                # pilImg.save('./dataset/te/'+str(size)+'/確認用/test_imgs/buildings/'+ filename + '.png')
                # pilImg = Image.fromarray(np.uint8(map_img[:,:,1]))
                # pilImg.save('./dataset/te/'+str(size)+'/確認用/test_imgs/distance/'+ filename + '.png')
                # pilImg = Image.fromarray(np.uint8(map_img[:,:,2]))
                # pilImg.save('./dataset/te/'+str(size)+'/確認用/test_imgs/ripple/'+ filename + '.png')
                # pilImg = Image.fromarray(np.uint8(map_img[:,:,:2]))
                # pilImg.save('./data/'+str(size)+'/確認用/test_imgs/'+ filename + '.png')

                test_maps2.append(test_map.tolist())

        test_maps2 = np.array(test_maps2)

        # 保存
        np.savez(new_dir_path+'/test_2.npz', test_maps2)

        xxx+=1
    yyy+=1
