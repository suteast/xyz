#'./data4GAN/1dataset/'に保存されたデータセットを用いてDCNNに入力


# 2_GAN_pred_allpoints.py
import os
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin") 
import datetime
from scipy import stats
import tensorflow.keras
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras import backend as K 
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from contextlib import redirect_stdout

# GPUの制限
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

df = pd.read_excel("./data/mapdata.xlsx") #エクセルから建物データを取得
x = df['x[m]'].values #建物データの座標x
y = df['y[m]'].values #建物データの座標y

# 対象のチャンネル
padding = 500
ch = 'all'
# ch = '0ch'
# ch = '1ch'
# ch = '2ch'
# epochs = '10epochs'
epochs = '200epochs(cp)'
# epochs = '500epochs'

# モデルのパス(この重みは研究室のPCにあります。)
model_path = './outputs/256/all/256_202104240119mse32batchallch200epoch/256_202104240119mse32batchallch200epoch_Checkpoint.h5'
# model_path = './outputs/256/all/256_202104230224mse32batchallch500epoch/256_202104230224mse32batchallch500epoch.h5'

# rmse
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred), axis = -1))

#--------------------main---------------------------
# モデルをロード
model=load_model(
    model_path,
    custom_objects={'rmse': rmse}
    )
# model_cp.compile(loss=model_cp.loss, optimizer=model_cp.optimizer, metrics=[rmse])

'''
基地局の位置に応じたデータセットで損失値を算出していく
'''
yyy=0 #基地局のy軸座標のカウント
while yyy*50 <= max(y)-min(y): #50m間隔なのでy軸のカウントを50倍
    xxx=0 #基地局のx軸座標のカウント
    if xxx==28: xxx=0 #右端に到達したらx成分カウントをリセット、50m間隔なのでカウント28で折り返し
    while xxx*50 <= max(x)-min(x): #50m間隔なのでx軸のカウントを50倍
        
        #基地局の座標(y,x),位置合わせで(y,x)の最小値を引く
        # base = np.array([max(y)-(yyy*50)-min(y)+padding, max(x)-(xxx*50)-min(x)+padding])

        '''
        保存先の指定部分
        base_position: 基地局の座標の文字列(全体マップのロードや、保存するときのファイル名に用いる。)
        '''
        base_y = int(max(y)-(yyy*50))
        base_x = int(max(x)-(xxx*50))
        base_position = str(base_y)+'_'+str(base_x)
        print(base_position)

        # データセットのロード
        dataset_path = './data4GAN/1dataset/'+base_position
        test_data_1 = np.load(dataset_path+"/test_1.npz")
        test_data_2 = np.load(dataset_path+"/test_2.npz")
        # print(test_data_1)
        X1 = test_data_1['arr_0']
        X2 = test_data_2['arr_0']

        # ガッチャンコ
        X = np.concatenate([X1, X2], 0)
        # X = X.astype(np.float32)

        # ガッチャンコしたデータセットを標準化
        # X = stats.zscore(X,axis=None)
        X[:,:,:,0] = stats.zscore(X[:,:,:,0],axis=None)
        X[:,:,:,1] = stats.zscore(X[:,:,:,1],axis=None)
        X[:,:,:,2] = stats.zscore(X[:,:,:,2],axis=None)

        # 保存先の設定
        new_dir_path = './data4GAN/2predOutputs/'+base_position
        os.mkdir(new_dir_path)

        # 損失値を計算
        pred_ = model.predict(X)
        pred_ = np.sum(pred_, axis=1)

        print(pred_.shape)
        # (8190,)
        # (8268,)
 
        # 推定された損失値をエクセルファイルで保存する。
        df_cp = pd.DataFrame({
            '推定値': pred_
        })
        df_cp.to_excel(new_dir_path+'/'+base_position+'.xlsx')
        print(base_position)
        print("done")
        print("---------------------------------------------")

        xxx+=1
    yyy+=1
