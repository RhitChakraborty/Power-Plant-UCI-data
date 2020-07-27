
"""
Training models
"""

import pandas as pd
import numpy as np
from sklearn import metrics,preprocessing
import dispatcher
import warnings
#import joblib
warnings.filterwarnings('ignore')

def run(fold,model):
    df=pd.read_csv('../inputs/train_folds.csv')

    features=[f for f in df.columns if f not in ['PE','kfold']]
    
    df_train=df[df.kfold!=fold].reset_index(drop=True)
    df_valid=df[df.kfold==fold].reset_index(drop=True)
    
    x_train=df_train[features].values
    y_train=df_train.PE.values
    x_valid=df_valid[features].values
    y_valid=df_valid.PE.values
    
    if model not in ['rf','dt']:
        sc_x=preprocessing.StandardScaler()
        sc_y=preprocessing.StandardScaler()
        x_train=sc_x.fit_transform(x_train)
        x_valid=sc_x.transform(x_valid)
        y_train=sc_y.fit_transform(y_train.reshape(-1,1))    
    
    reg=dispatcher.models[model]
    reg.fit(x_train,y_train)
    
    
    
    #joblib.dump(model,f"../models/{model}_{fold}.bin")
    
    if model not in ['rf','dt']:
        y_pred_sc=reg.predict(x_valid)
        y_pred=sc_y.inverse_transform(y_pred_sc.reshape(-1,1))
    else:
        y_pred=reg.predict(x_valid)
    rmse=metrics.mean_squared_error(y_valid, y_pred,squared=False)
    print(f"fold= {fold} and RMSE={rmse}")
    
    
if __name__=="__main__":
    for model in ['lm','dt','rf','svr','xgb']:
   
        print(model)
        for f in range(5):
            run(f,model)