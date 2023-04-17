#基地局に応じた空間情報を作成


# 0_GAN_table2map_auto_100.py
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

padding = 500#paddingの範囲(マップの周りに空白を追加)[m]
df = pd.read_excel("./data/mapdata.xlsx")#エクセルから建物データを取得
x = df['x[m]'].values#建物データの座標x
y = df['y[m]'].values#建物データの座標y
h = df['h[m]'].values#建物の高さ
resolution = 1#解像度(m)、メッシュサイズ

#xyの最小値(dataset作るときの位置合わせ用)
min_xy = np.array([(min(x))/resolution, (min(y))/resolution])
#各チャンネルの最大値を取り出す(NN入力で使う)
max_num = np.array([0,0])
#行列の列数(全体マップのx軸のサイズ)
x_size = int(np.floor(((max(x)+padding - min(x)+padding + 1) / resolution)))
#行列の行数(全体マップのy軸のサイズ)
y_size = int(np.floor(((max(y)+padding - min(y)+padding + 1) / resolution)))
#全体マップを表す行列
map_ = np.zeros((y_size, x_size, 2), dtype=np.int64) # np.int ここかえた

#基地局からの最大距離(スケール合わせ用)
# distance_scale = np.linalg.norm(base - np.array([y_size,x_size]))
print(max(y))
print(max(x))

def discriminant(corners,target):
    '''
    discriminant(): 点が建物内部に存在するかを判別する関数
    corners: 建物の4頂点の座標 ndarray([y1,x1],[y2,x2],[y3,x3],[y4,x4])
    target: 目的座標 ndarray([y,x])
    '''

    # ベクトル化する。
    vector_a = np.array([corners[0][1], corners[0][0]])
    vector_b = np.array([corners[1][1], corners[1][0]])
    vector_c = np.array([corners[3][1], corners[3][0]])
    vector_d = np.array([corners[2][1], corners[2][0]])
    vector_e = np.array([target[1], target[0]])
    
    # 点から点へのベクトル
    vector_ab = vector_b - vector_a
    vector_ae = vector_e - vector_a
    vector_bc = vector_c - vector_b
    vector_be = vector_e - vector_b
    vector_cd = vector_d - vector_c
    vector_ce = vector_e - vector_c
    vector_da = vector_a - vector_d 
    vector_de = vector_e - vector_d

    # 外積
    vector_cross_ab_ae = np.cross(vector_ab, vector_ae)
    vector_cross_bc_be = np.cross(vector_bc, vector_be)
    vector_cross_cd_ce = np.cross(vector_cd, vector_ce)
    vector_cross_da_de = np.cross(vector_da, vector_de)
    return (vector_cross_ab_ae < 0) and (vector_cross_bc_be < 0) and (vector_cross_cd_ce < 0) and (vector_cross_da_de < 0)

def distance(base, target):
    '''
    distance(): 基地局との距離を計算する関数
    base: 基地局の座標 ndarray([y,x])
    target: 目的座標 ndarray([y,x])
    # distance_scale: スケール(255以内の値で揃えるため)
    '''
    return np.linalg.norm(base - target)

# 行列を作る
# 要素を4つずつ取得して、1件ずつ建物内部に高さの値を埋めていく。
# 建物情報を作成
for i in tqdm(range(0,len(y),4)):

    # 建物の頂点を取得
    corners = np.array([
            [int(np.floor((y[i]-min(y)+padding)/resolution)), int(np.floor((x[i]-min(x)+padding)/resolution))],
            [int(np.floor((y[i+1]-min(y)+padding)/resolution)), int(np.floor((x[i+1]-min(x)+padding)/resolution))],
            [int(np.floor((y[i+2]-min(y)+padding)/resolution)), int(np.floor((x[i+2]-min(x)+padding)/resolution))],
            [int(np.floor((y[i+3]-min(y)+padding)/resolution)), int(np.floor((x[i+3]-min(x)+padding)/resolution))]
        ], dtype=int)

    # 四角形内部を塗りつぶすところ
    for j in range(corners[2][0]-int(np.floor(60/resolution)), corners[1][0]+int(np.floor(60/resolution))):
        for k in range(corners[2][1]-int(np.floor(60/resolution)), corners[1][1]+int(np.floor(60/resolution))):
            target = np.array([j,k])
            if discriminant(corners,target)==True and map_[j,k,0] < int(h[i]):
                map_[j,k,0] = int(h[i])# 全体マップを表す行列map_の0チャンネルに高さを入れる

'''
基地局の位置を変化させる。
基地局の位置に応じて距離情報を計算する。
基地局の座標をフォルダ名にして、そこを保存先にする。
'''
yyy=0 #基地局のy軸座標のカウント
xxx=0 #基地局のx軸座標のカウント
while yyy*50 <= max(y)-min(y): #50m間隔なのでy軸のカウントを50倍
    xxx=0
    while xxx*50 <= max(x)-min(x): #50m間隔なのでy軸のカウントを50倍
        #基地局の座標(y,x),位置合わせで(y,x)の最小値を引く
        base = np.array([max(y)-(yyy*50)-min(y)+padding, max(x)-(xxx*50)-min(x)+padding])                          

	# フォルダ分けのために文字列を作成する部分
        base_y = int(max(y_map)-(yyy*50)) # 基地局のy軸座標
        base_x = int(max(x_map)-(xxx*50)) # 基地局のx軸座標
        base_position = str(base_y)+'_'+str(base_x) #フォルダ名、座標(y,x)を名前とする。
        print(base_position)

	# 保存先を作成
        new_dir_path = './data4GAN/0fullmaps/50/'+base_position
        os.mkdir(new_dir_path)

        # 送受信間距離を全体マップの1チャンネルに書いていく
        for l in range(y_size): # 全体マップのy軸のサイズをループ
            for m in range(x_size): # 全体マップのx軸のサイズをループ
                target = np.array([l,m]) # 対象メッシュの座標
                map_[l,m,1] = distance(base, target) # 対象メッシュとBSの距離を、# 全体マップを表す行列map_の1チャンネルに追記する

        # 保存
        map_ = map_[::-1]	# 反転（実際の見た目に合わせるため）
        np.save(new_dir_path+"/map", map_) # 全体マップをnpyファイルで保存
        pilImg = Image.fromarray(np.uint8(map_)) # 画像に変換
        pilImg.save(new_dir_path+'/map.png') # 画像を保存
        map_ = map_[::-1]	# 反転（次の処理のため、念のため）
        xxx+=1 # x軸のカウント追加
    yyy+=1 # y軸のカウント追加
