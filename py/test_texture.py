from texture_features_RC import *
import numpy as np
import skimage.feature as ft

if __name__ == '__main__':

    rc = texture_features_RC()

    METHOD = 'uniform'
    P = 16
    R = 2

    imgs = rc.load_dicompy('/home/erikson/Documentos/Dataset/LUNG1-001/09-18-2008-StudyID-69331/0-82046/000035.dcm',slices=True)
    

    feature_vec = rc.extract_texture(imgs)

    lbp = ft.local_binary_pattern(imgs[0], P, R, METHOD)
    
    hist, _ = np.histogram(lbp, normed=True, bins=P + 2, range=(0, P + 2))

    


    



    