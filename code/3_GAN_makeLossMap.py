#算出された損失値をもとに、分布画像を作成。
#'./data4GAN/2predOutputs/'をもとに、損失値の分布画像を作成します。
#各メッシュに、損失値を与えることで分布を得ます。各損失値の大きさに応じて、疑似的なカラーを与えて保存



# 3_GAN_makeLossMap.py
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from scipy import stats
from sklearn import preprocessing
import os

'''
base_stationなんちゃらの名前付け辺りのためだけにエクセルデータをロード(;o;)
'''
df = pd.read_excel("./data/mapdata.xlsx")#エクセルから建物データを取得
x_map = df['x[m]'].values #建物データの座標x
y_map = df['y[m]'].values #建物データの座標y

map_ = np.load("./data/map_padding/map_padding.npy")#全体マップ
map_ = map_[::-1]#扱いやすいようにy軸を反転
padding = 500#全体マップ作成時と同じ値にしてください

#都市全体の空間情報画像のy軸の座標を定義(10mメッシュ)
y = np.arange(padding+181, map_.shape[0]-padding-181, 10)
#都市全体の空間情報画像のx軸の座標を定義(10mメッシュ)
x = np.arange(padding+181, map_.shape[1]-padding-181, 10)

'''
基地局の位置に応じた損失値分布画像を作成していく
'''
yyy=0 #基地局のy軸座標のカウント
while yyy*50 <= max(y_map)-min(y_map): #50m間隔なのでy軸のカウントを50倍
    xxx=0 #基地局のx軸座標のカウント
    while xxx*50 <= max(x_map)-min(x_map): #50m間隔なのでx軸のカウントを50倍
        
        # base = np.array([max(y)-(yyy*50)-min(y)+padding, max(x)-(xxx*50)-min(x)+padding])                          #基地局の座標(y,x),位置合わせで(y,x)の最小値を引く
        '''
        保存先の指定部分
        base_position: 基地局の座標の文字列(全体マップのロードや、保存するときのファイル名に用いる。)
        '''
        base_y = int(max(y_map)-(yyy*50)) # 基地局のy軸座標
        base_x = int(max(x_map)-(xxx*50)) # 基地局のx軸座標
        base_position = str(base_y)+'_'+str(base_x) #フォルダ名、座標(y,x)を名前とする。
        print(base_position)

        '''
        DCNNによって算出された損失値をロード、正規化
        等
        '''
        pred_path = './data4GAN/2predOutputs/'+base_position + '/' + base_position + '.xlsx'#損失値のデータの保存場所を指定
        df = pd.read_excel(pred_path)#エクセルからデータを取得
        preds = df['推定値'].values#DCNNによる推定値
        # preds = stats.zscore(preds)
        preds = preprocessing.minmax_scale(preds)#最小値0、最大値1に正規化

        # 分布画像の保存先を指定
        new_dir_path = './data4GAN/3lossmaps/'+base_position
        os.mkdir(new_dir_path)

        # 分布画像を格納する配列
        pred_map = []

        '''
        損失値を、都市全体の各メッシュに与えていく。
        都市全体を行列としたとき、1行ごとに値を与えていく。
        '''
        num=0#predsに対するインデックス指定に用いる
        for i in tqdm(range(len(x))): #都市全体の空間情報画像のx成分をループ
            pred_list=[]    #1行ごとの推定損失値一覧を格納する配列
            for j in range(len(y)): #都市全体の空間情報画像のy成分をループ
                # p_test = np.append([x0[i], x1[j]])
                #受信座標(x, y)
                # MS = np.array([x[i], y[j]])
                flag = map_[y[j], x[i], 0]#全体マップ上の、対象メッシュにおける建物高を取り出す
                if flag != 0:   #建物高が0じゃないとき、つまり建物があるメッシュには
                    pred_list.append(0) #損失値は0とする。
                else:   #その他、つまち建物がないメッシュには
                    pred_list.append(255*preds[num+j])#DCNNによる損失値を入れる
            pred_map.append(pred_list)#1行ごとの推定損失値一覧を分布配列に与えていく。
            num = (i+1)*len(y) #predsに対するインデックス指定に用いる

        pred_map = np.array(pred_map) #numpy配列に変換
        pred_map = np.rot90(pred_map) #90度反時計方向回転
        # pred_map = pred_map.repeat(10, axis=0).repeat(10, axis=1)

        #一旦、保存
        filename = base_position
        pilImg = Image.fromarray(np.uint8(pred_map))#画像化
        imgpath = new_dir_path +'/'+ filename + '.png'#保存さきの指定
        pilImg.save(imgpath)#保存

        # 保存した画像をグレースケールでロード
        pic=cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)

        #疑似カラー化(JET)
        # RGBとして扱うために色を付ける。
        pseudo_color = cv2.applyColorMap(pic, cv2.COLORMAP_JET)
        # imgpath = './outputs/allpoints/256/heatmaps/' + filename + '_c.png'
        # 保存
        cv2.imwrite(imgpath, np.array(pseudo_color))

        # img = img[:,:,0].reshape(img.shape[:2])
        # print(img.shape) # (240, 320)
        print('-----------------done----------------------')
        xxx+=1
    yyy+=1
