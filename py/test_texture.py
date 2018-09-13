#!/usr/bin/env python -W ignore::DeprecationWarning

from texture_features_RC import *
import numpy as np
import skimage.feature as ft
import pandas as pd
import statistics
import time
from operator import itemgetter
from datetime import datetime as dt
from networkx import algorithms

#classificacao
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, KFold,GroupKFold
from sklearn.svm import SVC


from _thread import start_new_thread, allocate_lock
import threading


logs_metricas = []


def write_csv(data,file):
		df = pd.DataFrame(data)
		df.to_csv('../logs/'+file+'.csv', mode='a', sep=';',index=False, header=False)


def read_csv(file):

		df1 = pd.DataFrame.from_csv('../dataset/%s'%(file),sep=';',index_col=0,encoding ='ISO-8859-1')

		df1 = df1.reset_index()

		return df1

def write_txt(data):

    with open('features.log', 'w') as out:
       np.savetxt(out, data, '%4.6f')


def clf_randomForest(X,target):

    clf_rf = RandomForestClassifier(n_estimators=500)
    #clf_svm = SVC(C=(2**7), kernel='rbf')
    kf = KFold(10, shuffle=True, random_state=1)
    
    ac_v = []
    cm_v = []
    p_v = []
    r_v = []
    f1_v = []
    e_v = []
    predicts = []

    
    for train_index,teste_index in kf.split(X,target):
        
        
        X_train, X_test = list(itemgetter(*train_index)(X)), list(itemgetter(*teste_index)(X))
        y_train, y_test = list(itemgetter(*train_index)(target)), list(itemgetter(*teste_index)(target))
        clf_rf.fit(X_train,y_train)
        pred = clf_rf.predict(X_test)
        ac = accuracy_score(y_test, pred)
        p = precision_score(y_test, pred,average='weighted')
        r = recall_score(y_test, pred,average='weighted')
        f1 = (2*p*r)/(p+r)
        e = mean_squared_error(y_test, pred)
        cm = confusion_matrix(y_test,pred)
        cm_v.append(cm)
        ac_v.append(ac)
        p_v.append(p)
        r_v.append(r)
        f1_v.append(f1)
        e_v.append(e)
        
        
    ac = statistics.median(ac_v)
    p = statistics.median(p_v)
    f1 = statistics.median(f1_v)
    r = statistics.median(r_v)
    e = statistics.median(e_v)
    

    return (ac,p,f1,r,e)


def convert_df(df):

    new_def = []

    for t in df:

        if t  == True:
            new_def.append(1)
        
        elif t == False:
            new_def.append(0)

    
    return new_def

def crop_slices(imgray):

    array = []

    row = np.size(imgray,0)
    col = np.size(imgray,1)
            
    s = int(row/4)
    s1 = int(col/4)

            
    array.append(imgray)
    array.append(imgray[1:s,1:s1])
    array.append(imgray[s:(2*s),s1:(2*s1)])
    array.append(imgray[(2*s):(3*s),(2*s1):(3*s1)])
    array.append(imgray[(3*s):(4*s),(3*s1):(4*s1)])

    return array

def test_lbp(imgs_dicom,target):

    features_lbp = []

    print('Iniciando extração de caracteristicas com LBP...')

    for im in imgs_dicom:
        #print('Calc LBP')
        lbp = ft.local_binary_pattern(im[0], P, R, METHOD)
        flbp, _ = np.histogram(lbp, normed=True, bins=P + 2, range=(0, P + 2))
        features_lbp.append(flbp.tolist())    
    
    print('Test LBP...')

    ac,p,f1,r,e = clf_randomForest(features_lbp,target)

    l = 'lbp', ac,p,f1,r,e, str(dt.now())

    logs_metricas.append(l)

    print('Acuracia = %f'%(ac))

    print('Precisao = %f'%(p))

    print('Recall = %f'%(r))

    print('Erro = %f'%(e))

    print('Test LBP Finalizado.')

    time.sleep(5)


def test_rc(imgs_dicom, target):

    features_rc = []


    print('Iniciando extração de caracteristicas com RC...')

    count = 0
    for im in imgs_dicom:
        count += 1
        print('%i'%(count))
        #secs = crop_slices(im[0])
        frc = rc.extract_texture([im[0]])
        features_rc.append(frc)
        write_txt(features_rc)

    
    print('Test RC...')

    ac,p,f1,r,e = clf_randomForest(features_rc,target)

    l = 'rc', ac,p,f1,r,e, str(dt.now())

    logs_metricas.append(l)

    print('Acuracia = %f'%(ac))

    print('Precisao = %f'%(p))

    print('Recall = %f'%(r))

    print('Erro = %f'%(e))

    print('Test RC Finalizado.')

def _test_rc(target):

    features_rc = []
    with open("features_.log", "r") as ins:
        for line in ins:
            line = line.strip("\n")
            vals = line.split(' ')
            vals = [float(i) for i in vals]
            features_rc.append(vals)

    print('Test RC...')

    ac,p,f1,r,e = clf_randomForest(features_rc,target)

    l = 'rc', ac,p,f1,r,e, str(dt.now())

    logs_metricas.append(l)

    print('Acuracia = %f'%(ac))

    print('Precisao = %f'%(p))

    print('Recall = %f'%(r))

    print('Erro = %f'%(e))

    print('Test RC Finalizado.')
    

if __name__ == '__main__':
    
    
    print("Iniciando do algoritmo...")

    datas = read_csv('overview.csv')

    df = convert_df(datas['Contrast'])

    datas['Contrast'] =  df
    
    rc = texture_features_RC()

    METHOD = 'uniform'
    P = 16
    R = 2

    imgs_dicom = []  

    print("Carregando as imagens...")
    
    for name in datas['dicom_name']:
        imgs = rc.load_dicompy('../dataset/dicom_dir/%s'%(name))
        imgs_dicom.append(imgs)
    
    #features = 
    
    #test_rc_preLoad(imgs_dicom,datas['Contrast'])

    test_lbp(imgs_dicom,datas['Contrast'])
    
    test_rc(imgs_dicom,datas['Contrast'])
    
    #_test_rc(datas['Contrast'])

    write_csv(logs_metricas,'metricas')

    #extracao de caracteristicas com o LBP e RC
    #for im in imgs_dicom:
    #    print('Calc RC')
    #    secs = crop_slices(im[0])
    #    frc = feature_vec = rc.extract_texture(secs)
    #    print('Calc LBP')
    #    lbp = ft.local_binary_pattern(im[0], P, R, METHOD)
    #    flbp, _ = np.histogram(lbp, normed=True, bins=P + 2, range=(0, P + 2))
    #    features_lbp.append(flbp)
    #    features_rc.append(frc)

    
    


    



    