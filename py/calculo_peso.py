import numpy as np
#import scipy as sp
# import cv2
import networkx as nx
import matplotlib.pyplot as plt
#from sklearn.feature_extraction import image
from PIL import Image
#from scipy.spatial import distance
import pydicom
from oct2py import octave
import math
import collections
from sklearn.feature_extraction import image
import gc

#params:
## path: local da imagem
## opc: opção para utilizar as seções da imagem ou imagem inteira
def load_dicompy(path):
   ds = pydicom.read_file(path)

   array = np.array(ds.pixel_array, dtype=np.uint8)

   return [array]

def load_dicom_script(path, opc):

    octave.addpath('/home/erikson/Documentos/Texture_Complex_Networks/Matlab')

    sections = []
    
    if opc == 1:
        img  = octave.dicom2gray(path)
        sections = [img]
    
    if opc == 2:
        img, img2, img3, img4 = octave.dicom2grayMulti(path, nout=4)
        sections = [img, img2, img3, img4]
    
    return sections

def load_img(path):

    im = Image.open(path).convert('L')

    im2 = np.asarray(im)

    section = [im2]

    return section

def convert2graph(imgs):
    
    grs = []
    
    for im in imgs:
        gr = image.img_to_graph(im)
        grs.append(gr)
    
    return grs 

#params:
## sections: Seções da imagem
## opc: quando verdadeiro salva a imagem do grafo
def calc_weights(sections, opc=False):
    s = 1
    gs = []
    log_w = []
    for sec in sections:
        print('Iniciando o grafo da secção %i ...'%(s))
        col = sec[0].size
        row = int(sec.size/col)
        #col, row = sec.size
    
        i = 0
        j = 0
    
        G = nx.Graph()
        
        np.seterr(over="ignore")
        
        r = 2
        t = 0.9
    
        # pospx = [i,j] e intespx. representando o pixel
        pxdic = dict()
    
    
        print("Iniciando calculo dos pesos...")
    
        cont = 0
        for i in range(1, row - 2):
            for j in range(1, col - 2):
                #ind = [i+1, j, i, j+1, i+1, j+1]
                ind = [i+1, j, i, j+1, i+1, j+1, i-1, j-1, i-1, j, i+1, j+1, i, j-1, i-1, j+1]
                #ind = r * [i-1, j, i,j-1, i+1, j, i, j+1]
                base = cont
                pxdic[cont] = dict()
                pxdic[cont]['pospx'] = [i, j]
                pxdic[cont]['intespx'] = sec[i][j]
                G.add_node(cont)
                for k in range(0, int(len(ind)/2)):
                    d = 0
                    #d = math.sqrt(((ind[k] - i) ** 2) + ((ind[k+1] - j) ** 2))
                    #w = (255 - math.fabs(sec[i][j] - sec[ind[k]][ind[k+1]]))/255
                    #w = ((((ind[k] - i)**2) + ((ind[k+1] - j)**2)) + ((r**2)*((math.fabs(sec[i][j] - sec[ind[k]][ind[k+1]]))/255))/(2*(r**2)))
                    d = (((ind[k] - i) ** 2) + ((ind[k+1] - j) ** 2)) + ((sec[i][j] - sec[ind[k]][ind[k+1]]) ** 2)
                    w = ((d/(255)**2)-(r ** 2))
                    log_w.append(w)
                    if d <= r and w <= 0.9:
                        cont += 1
                        G.add_node(cont)
                        pxdic[cont] = dict()
                        pxdic[cont]['pospx'] = [k, k+1]
                        pxdic[cont]['intespx'] = sec[k][k+1]
                        #w = ((((ind[k] - i)**2) + ((ind[k+1] - j)**2)) + ((r**2)*((math.fabs(sec[i][j] - sec[ind[k]][ind[k+1]]))/255))/(2*(r**2))) 
                        G.add_edge(base, cont, weight=w)
                
                
            
        
        print("Calculo dos pesos finalizado.")
         
        #gc.collect()
        
        gs.append(G)
        
        if opc == True:
            print("Iniciando a o desenho do grafo...")
            f = plt.figure()
            nx.draw(G)
            f.savefig('grafo_sec%i.png'%(s))
        #plt.show()
        
        s += 1
    
    with open('pesos.log', 'a') as out:
        np.savetxt(out, log_w, fmt='%4.1f')

    
    return gs

#params:
## G: Grafo gerado
## opc: quando verdadeiro exibe o grafico do histograma 
def calc_histDeg(G, opc=False):
    
    #degree_sequence = [d for n, d in G.degree()]
    #degreeCount = collections.Counter(degree_sequence)
    #deg, cnt = zip(*degreeCount.items())
    
    #if opc == True:
    #    fig, ax = plt.subplots()
    #    plt.bar(deg, cnt, width=0.80, color='b')
    #    plt.title("Degree Histogram")
    #    plt.ylabel("Count")
    #    plt.xlabel("Degree")
    #    ax.set_xticks([d + 0.4 for d in deg])
    #    ax.set_xticklabels(deg)
    #    plt.show()
    
    #centDeg = nx.degree_centrality(G)
    #degreeCount = collections.Counter(centDeg)
    #deg, cnt = zip(*degreeCount.items())

    #print(deg)

    #print(cnt)
    
    hdeg = nx.degree_histogram(G)

    return hdeg

def dens_prob(hst):
    
    dens = []
    
    for i in hst:
        d = round((i/sum(hst)),4)
        if d > 0:
            dens.append(d)
    
    return dens


def metrics_rc(d_prob):
    
    mean = 0
    entropy = 0
    energy = 0
    contrast = 0
    
    index = 0
    for k in d_prob:
        mean += index * k
        contrast += k * (index**2)  
        energy += (k ** 2)
        entropy += k * math.log2(k)
        
        index += 1
    
    entropy *= -1
    
    return (mean,entropy,energy,contrast)
    

if __name__ == '__main__':
    
    print("Iniciando o algoritmo...")
    
    sections = load_dicom_script('/home/erikson/Documentos/Dataset/LUNG1-010/01-01-2014-StudyID-54264/1-08510/000011.dcm', opc=2)

    #sections = load_dicompy('/home/erikson/Documentos/Dataset/LUNG1-003/01-01-2014-StudyID-34270/1-28595/000035.dcm')
    
    #sections = load_img("/home/erikson/Documentos/Texture_Complex_Networks/laranja.png")

    print("Script matlab processado.")
    
    #gfs = convert2graph(sections)
    
    gfs = calc_weights(sections)
    
    g_metric = []
    
    index = 0
    for g in gfs:
        d_prob = []
        d = calc_histDeg(g)
        #print(d)
        d_prob = dens_prob(d)
        #print(d_prob)
        me,etr,enr,ctr = metrics_rc(d_prob)
        gm = dict()
        gm['mean'] = me
        gm['entropy'] = etr
        gm['energy'] = enr
        gm['contrast'] = ctr
        g_metric.append(gm)
      
    
    print(g_metric)

    gc.collect()
    
    print("Algoritmo finalizado!!!")
    
#    print(log_dist[len(log_dist)-1])
#    print(log_i[len(log_i)-1])
#    print(log_j[len(log_j)-1])
#    print(log_w[len(log_w)-1])
#    
#    with open('dist.log', 'a') as out:
#        np.savetxt(out, log_dist, fmt='%4.1f')
#    
#    
#    with open('i.log', 'a') as out:
#        np.savetxt(out, log_i, '%4.1f')
#    
#    with open('j.log', 'a') as out:
#        np.savetxt(out, log_j, '%4.1f')
#    
#    with open('pesos.log', 'a') as out:
#        np.savetxt(out, log_w, fmt='%4.1f')
    #print(log)
    


    # print(m_adjacencia)
