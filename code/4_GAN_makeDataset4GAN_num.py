#データセットの作成
#ここまでで、都市全体の空間情報と損失値の分布画像が用意できました。
#これから、基地局の位置が同じとなる各ペアを対応させて、データセットとして整理



# 4_GAN_makeDataset4GAN_num.py
# 画像2つを横に結合
import pandas as pd
from PIL import Image
import numpy as np
# from tqdm import tqdm

'''
base_stationなんちゃらの名前付け辺りのためだけにエクセルデータをロード(;o;)
'''
df = pd.read_excel("./data/mapdata.xlsx")#エクセルから建物データを取得
x_map = df['x[m]'].values #建物データの座標x
y_map = df['y[m]'].values #建物データの座標y

# データセットを格納する配列
datamaps = []

'''
画像の周りに空白を追加する関数
https://note.nkmk.me/python-pillow-add-margin-expand-canvas/
'''
def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

'''
2つの画像を並べる関数
im1, im2: 対象とする2枚の画像
https://note.nkmk.me/python-pillow-concat-images/
'''
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

'''
基地局の位置に応じて、対応するペアを作成
'''
num=1
yyy=0 #基地局のy軸座標のカウント
while yyy*50 <= max(y_map)-min(y_map): #50m間隔なのでy軸のカウントを50倍
    xxx=0 #基地局のx軸座標のカウント
    while xxx*50 <= max(x_map)-min(x_map): #50m間隔なのでx軸のカウントを50倍
        '''
        保存先の指定部分
        base_position: 基地局の座標の文字列(全体マップのロードや、保存するときのファイル名に用いる。)
        '''
        base_y = int(max(y_map)-(yyy*50))
        base_x = int(max(x_map)-(xxx*50))
        base_position = str(base_y)+'_'+str(base_x)
        print(base_position)

        '''
        都市全体の空間情報の処理
        '''
        # 都市全体の空間情報のnpyファイルをロード
        fullmap = np.load('./data4GAN/3fullmaps/'+base_position+'.npy')
        # 画像化
        im = Image.fromarray(np.uint8(fullmap)) # 要素を整数値に変換
        im_new = add_margin(im, 0, 151, 178, 0, (0, 0, 0)) # 画像のサイズを256に合わせるため、空白を追加
        fullmap = np.array(im_new) # fullmapに上書き

        '''
        損失値の分布画像の処理
        '''
        # 損失値の分布画像をロード
        lossmap = './data4GAN/3lossmaps/'+base_position + '/' + base_position + '.png'
        lossmap = np.array(Image.open(lossmap))
        # 画像化
        im = Image.fromarray(np.uint8(lossmap)) # 要素を整数値に変換
        im_new = add_margin(im, 0, 151, 178, 0, (0, 0, 0)) # 画像のサイズを256に合わせるため、空白を追加
        lossmap = np.array(im_new) # lossmapに上書き

        '''
        ペアを作成
        '''
        # print(fullmap)
        numpy_image = np.hstack((lossmap, fullmap)) #2つの画像を結合(ペア画像を作成)
        # ペア画像を保存
        pilImg = Image.fromarray(np.uint8(numpy_image)) #ペア画像の要素を整数に変換
        pilImg.save('./data4GAN/4dataset/'+ str(num) + '.png') #画像として保存

        datamaps.append(numpy_image) #配列に追加
        print(num)
        print('-----------------done----------------------')

        xxx+=1
        num+=1
    yyy+=1

# データセットを格納している配列をndarray変換
datamaps = np.array(datamaps)
# 保存
np.savez('./data4GAN/4dataset/dataset4GAN.npz', datamaps)
