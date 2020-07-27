
"""
tensorflow
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import metrics,preprocessing

def run(fold):
    df=pd.read_csv('../inputs/train_folds.csv')
    
    df['new']=df.AP*df.RH/df.AT
    
    df_train=df[df.kfold!=fold].reset_index(drop=True)
    df_valid=df[df.kfold==fold].reset_index(drop=True)
    
    features=[f for f in df.columns if f not in ['PE','kfold']]
    x_train=df_train[features].values
    y_train=df_train.PE.values
    x_valid=df_valid[features].values
    y_valid=df_valid.PE.values
    
    sc_x=preprocessing.StandardScaler()
    sc_y=preprocessing.StandardScaler()
    
    x_train=sc_x.fit_transform(x_train)
    x_valid=sc_x.transform(x_valid)
    y_train=sc_y.fit_transform(y_train.reshape(-1,1))
    y_valid=sc_y.fit_transform(y_valid.reshape(-1,1))
    
    
    ann=tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=32,activation='relu'))
    ann.add(tf.keras.layers.Dropout(0.20))
    ann.add(tf.keras.layers.Dense(units=64,activation='relu'))
    ann.add(tf.keras.layers.Dropout(0.20))
    ann.add(tf.keras.layers.Dense(units=64,activation='relu'))
    
    ann.add(tf.keras.layers.Dense(units=16,activation='relu'))
    
    ann.add(tf.keras.layers.Dense(units=1,activation='linear'))
    
    ann.compile(optimizer='adam',loss='mse',metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    ann.fit(x_train,y_train,batch_size=32,epochs=300)
    
    y_pred_sc=ann.predict(x_valid)
    
    y_pred=sc_y.inverse_transform(y_pred_sc.reshape(-1,1))
    
    rmse=np.sqrt(np.mean(y_pred-y_valid))
    print(rmse)

    
    test=pd.read_csv('../inputs/Test.csv')
    sample=pd.read_csv('../inputs/sample_submission.csv')
    test['new']=test.AP*test.RH/test.AT
    test=test[features]
    test=sc_x.fit_transform(test.values)
    
    pred_sc=ann.predict(test)
    pred=sc_y.inverse_transform(pred_sc.reshape(-1,1))
    
    sample['PE']=pred

    sample.to_csv("../inputs/sub_tf.csv",index=False)
    
if __name__=="__main__":
    run(3)  