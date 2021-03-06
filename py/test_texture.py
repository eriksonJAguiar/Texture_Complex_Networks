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
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model


from _thread import start_new_thread, allocate_lock
import threading


logs_metricas = []
log_pred = pd.DataFrame()


def write_csv(data,file):
		df = pd.DataFrame(data)
		df.to_csv('../logs/'+file+'.csv', mode='a', sep=';',index=False, header=True)


def read_csv(file):

		df1 = pd.DataFrame.from_csv('../dataset/%s'%(file),sep=';',index_col=0,encoding ='ISO-8859-1')

		df1 = df1.reset_index()

		return df1

def write_txt(data):

    with open('features.log', 'w') as out:
       np.savetxt(out, data, '%4.6f')


def clf_mensure(clf,X,target,name,tec):

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
        clf.fit(X_train,y_train)
        pred = clf.predict(X_test)
        predicts += pred.tolist()
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
        
        

    log_pred[tec+'_'+name] = predicts
    
    
    
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

    ilbp = time.time()

    METHOD = 'nri_uniform'
    P = 5
    R = 3
    
    
    print('Iniciando extração de caracteristicas com LBP...')

    for im in imgs_dicom:
        #print('Calc LBP')
        lbp = ft.local_binary_pattern(im[0], P, R, METHOD)
        flbp, _ = np.histogram(lbp, normed=True, bins=P + 2, range=(0, P + 2))
        hist, bin_edges = np.histogram(im, density=True)
        f = flbp.tolist() + hist.tolist()
        features_lbp.append(f)    
    
    
    flbp = time.time()

    print("LBP Time - %f"%(flbp - ilbp))
    
    print('Test LBP...')

    clfs = []
    name_clf = ['rf','svm','nv','sgdc']

    clfs.append(RandomForestClassifier(n_estimators=500,n_jobs=2,criterion='gini'))
    clfs.append(SVC(C=2 , kernel='poly'))
    clfs.append(MultinomialNB())
    clfs.append(linear_model.SGDClassifier(loss='log',alpha=0.00001,learning_rate='optimal'))
    

    
    cnt = 0
    for clf in clfs:
        
        print("Teste com algoritmos %s"%(name_clf[cnt]))
        
        ac,p,f1,r,e = clf_mensure(clf,features_lbp,target,name_clf[cnt],'lbp')
        
        l = 'lbp',name_clf[cnt],ac,p,f1,r,e, str(dt.now())

        logs_metricas.append(l)

        print('Acuracia = %f'%(ac))

        print('Precisao = %f'%(p))

        print('Recall = %f'%(r))

        print('Erro = %f'%(e))

        cnt += 1

    
    print('Test LBP Finalizado.')



def test_rc(imgs_dicom, target):

    features_rc = []


    print('Iniciando extração de caracteristicas com RC...')

    irc = time.time()

    for im in imgs_dicom:
        #secs = crop_slices(im[0])
        frc = rc.extract_texture([im[0]])
        hist, bin_edges = np.histogram(im, density=True)
        f = frc + hist.tolist()
        features_rc.append(f)
        write_txt(features_rc)

    
    frc = time.time()

    print("RC Time - %f"%(frc-irc))

    
    print('Test RC...')

    clfs = []
    name_clf = ['rf','svm','nv']

    clfs.append(RandomForestClassifier(n_estimators=500))
    clfs.append(SVC(C=(2**7), kernel='rbf'))
    clfs.append(MultinomialNB())

    
    cnt = 0
    for clf in clfs:
        
        print("Teste com algoritmo %s"%(name_clf[cnt]))
        
        ac,p,f1,r,e = clf_mensure(clf,features_rc,target,name_clf[cnt],'rc')
        
        l = 'rc', name_clf[cnt], ac,p,f1,r,e, str(dt.now())

        logs_metricas.append(l)

        print('Acuracia = %f'%(ac))

        print('Precisao = %f'%(p))

        print('Recall = %f'%(r))

        print('Erro = %f'%(e))

        cnt += 1

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

    clfs = []

    name_clf = ['rf','svm','nv','sgdc']

    clfs.append(RandomForestClassifier(n_estimators=500,n_jobs=2,criterion='gini'))
    clfs.append(SVC(C=2 , kernel='poly'))
    clfs.append(MultinomialNB())
    clfs.append(linear_model.SGDClassifier(loss='log',alpha=0.00001,learning_rate='optimal'))

    
    cnt = 0
    for clf in clfs:
        
        print("Teste com algoritmo %s"%(name_clf[cnt]))
        
        ac,p,f1,r,e = clf_mensure(clf,features_rc,target,name_clf[cnt],'rc')
        
        l = 'rc', name_clf[cnt], ac,p,f1,r,e, str(dt.now())

        logs_metricas.append(l)

        print('Acuracia = %f'%(ac))

        print('Precisao = %f'%(p))

        print('Recall = %f'%(r))

        print('Erro = %f'%(e))

        cnt += 1

    print('Test RC Finalizado.')
    

if __name__ == '__main__':
    
    
    print("Iniciando do algoritmo...")

    datas = read_csv('overview.csv')

    df = convert_df(datas['Contrast'])

    datas['Contrast'] =  df
    
    rc = texture_features_RC()


    imgs_dicom = []  

    print("Carregando as imagens...")
    
    for name in datas['dicom_name']:
        imgs = rc.load_dicompy('../dataset/dicom_dir/%s'%(name))
        imgs_dicom.append(imgs)
    
    #features = 
    
    #test_rc_preLoad(imgs_dicom,datas['Contrast'])

    
    test_lbp(imgs_dicom,datas['Contrast'])
    
    irc = time.time()

    #test_rc(imgs_dicom,datas['Contrast'])
    _test_rc(datas['Contrast'])

    frc = time.time()

    print("RC Time - %f"%(frc-irc))
    
    #_test_rc(datas['Contrast'])

    write_csv(logs_metricas,'metricas')

    write_csv(log_pred,'predicoes')

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

    
    


    



    