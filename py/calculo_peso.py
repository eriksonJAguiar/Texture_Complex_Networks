import numpy as np
#import scipy as sp
# import cv2
import networkx as nx
import matplotlib.pyplot as plt
#from sklearn.feature_extraction import image
#from PIL import Image
#from scipy.spatial import distance
#import pydicom
from oct2py import octave
import math
import collections

#params:
## path: local da imagem
## opc: opção para utilizar as seções da imagem ou imagem inteira
def load_dicom(path, opc):

    octave.addpath('/home/erikson/Documentos/Texture_Complex_Networks/Matlab')

    sections = []
    
    if opc == 1:
        img  = octave.dicom2gray(path)
        sections = [img]
    
    if opc == 2:
        img, img2, img3, img4 = octave.dicom2grayMulti(path, nout=4)
        sections = [img, img2, img3, img4]
    
    return sections
    
#params:
## sections: Seções da imagem
## opc: quando verdadeiro salva a imagem do grafo
def calc_weights(sections, opc=False):
    s = 1
    gs = []
    for sec in sections:
        print('Iniciando o grafo da secção %i ...'%(s))
        col = sec[0].size
        row = int(sec.size/col)
    
        i = 0
        j = 0
    
        G = nx.Graph()
        
        np.seterr(over="ignore")
        
        r = 3
    
        # pospx = [i,j] e intespx. representando o pixel
        pxdic = dict()
    
    
        print("Iniciando calculo dos pesos...")
    
        cont = 0
        for i in range(0, row - 1):
            for j in range(0, col - 1):
                ind = [i+1, j, i, j+1, i+1, j+1]
                base = cont
                pxdic[cont] = dict()
                pxdic[cont]['pospx'] = [i, j]
                pxdic[cont]['intespx'] = sec[i][j]
                G.add_node(cont)
                for k in range(0, int(len(ind)/2)):
                    d = 0
                    d = math.sqrt(((ind[k] - i) ** 2) + ((ind[k+1] - j) ** 2))
                    if d <= r:
                        cont += 1
                        G.add_node(cont)
                        pxdic[cont] = dict()
                        pxdic[cont]['pospx'] = [k, k+1]
                        pxdic[cont]['intespx'] = sec[k][k+1]
                        w = round(((ind[k] - i)**2 + (ind[k+1] - j)**2 + ((r**2)*((math.fabs(sec[i][j] - sec[ind[k]][ind[k+1]]))/255))/(2*(r**2))), 4) 
                        G.add_edge(base, cont, weight=w)
         
        
        print("Calculo do pesos finalizado.")
        
        gs.append(G)
        
        if opc == True:
            print("Iniciando a o desenho do grafo...")
            f = plt.figure()
            nx.draw(G)
            f.savefig('grafo_sec%i.png'%(s))
        #plt.show()
        
        s += 1
    
    return gs

#params:
## G: Grafo gerado
## opc: quando verdadeiro exibe o grafico do histograma 
def calc_histDeg(G, opc=False):
    
    degree_sequence = sorted([d for n, d in G.degree()], reverse=False)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    
    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')
    
    if opc == True:
        plt.title("Degree Histogram")
        plt.ylabel("Count")
        plt.xlabel("Degree")
        ax.set_xticks([d + 0.4 for d in deg])
        ax.set_xticklabels(deg)
    
        plt.show()
    
    return (deg, cnt)

def dens_prob(hi):
    
    dens = []
    
    for i in hi:
        d = (i/sum(hi))
        dens.append(d)
    
    return dens


def metrics_rc(pi):
    
    mean = 0
    entropy = 0
    energy = 0
    contrast = 0
    
    index = 0
    for i in pi:
        mean += index * i
        contrast += i * (index**2)  
        energy += (i**2)
        entropy += i * math.log2(i)
        
        index += 1
    
    entropy *= -1
    
    return (mean,entropy,energy,contrast)
    

if __name__ == '__main__':
    
    print("Iniciando o algoritmo...")
    
    sections = load_dicom('/home/erikson/Documentos/Dataset/LUNG1-001/09-18-2008-StudyID-69331/0-82046/000001.dcm', opc=1)
    
    print("Script matlab processado.")
    
    graphs = calc_weights(sections)
    
    g_metric = []
    
    index = 0
    for g in graphs:
        d_prob = []
        d, c = calc_histDeg(g)
        d_prob = dens_prob(d)
        me,etr,enr,ctr = metrics_rc(d_prob)
        gm = dict()
        gm['mean'] = me
        gm['entropy'] = etr
        gm['energy'] = enr
        gm['contrast'] = ctr
        g_metric.append(gm)
      
    
    print(g_metric)
    
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
