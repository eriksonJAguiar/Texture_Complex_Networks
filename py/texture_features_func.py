import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import pydicom
from oct2py import octave
import math
import collections
from sklearn.feature_extraction import image
import gc


# carrega a imagem dicom a partir da biblioteca pydicom
# params:
# path: local da imagem
# opc: opção para utilizar as seções da imagem ou imagem inteira
def load_dicompy(path, slices=False):
    ds = pydicom.read_file(path)

    array = []

    imgray = np.array(ds.pixel_array, dtype=np.uint8)

    if slices == True:

        row = np.size(imgray, 0)
        col = np.size(imgray, 1)

        s = int(row/4)
        s1 = int(col/4)

        array.append(imgray)
        array.append(imgray[1:s, 1:s1])
        array.append(imgray[s:(2*s), s1:(2*s1)])
        array.append(imgray[(2*s):(3*s), (2*s1):(3*s1)])
        array.append(imgray[(3*s):(4*s), (3*s1):(4*s1)])

    else:
        array.append(imgray)

        print("Script de aquisicao da imagem processado.")

    return array

# funcao carrega uma imagem dicom a partir do matlab
# params:
# path: caminho da imagem
# opc: opcao 1 carrega a imagem inteira
def load_dicom_script(path, opc):

    octave.addpath('../Matlab')

    sections = []

    if opc == 1:
        img = octave.dicom2gray(path)
        sections = [img]

    if opc == 2:
        img, img2, img3, img4 = octave.dicom2grayMulti(path, nout=4)
        sections = [img, img2, img3, img4]

    print("Script matlab processado.")

    return sections

# funcao carrega uma imagem png
# params:
# path: caminho da imagem
def load_img(path):

   im = Image.open(path).convert('L')
   im2 = np.asarray(im)

   section = [im2]

   return section

 # params:
 # sections: Seções da imagem
 # opc: quando verdadeiro salva a imagem do grafo
def calc_weights_default(sections, opc=False, r=2, t=0.9):
     s = 1
     gs = []
     for sec in sections:
         print('Iniciando o grafo da secção %i ...' % (s))
         row = np.size(sec, 0)
         col = np.size(sec, 1)

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
                 # ind = [i+1, j, i, j+1, i+1, j+1]
                 ind = [i+1, j, i, j+1, i+1, j+1, i-1, j-1, i-1, j, i+1, j+1, i, j-1, i-1, j+1]
                 #ind = r * [i-1, j, i,j-1, i+1, j, i, j+1]
                 base = cont
                 pxdic[cont] = dict()
                 pxdic[cont]['pospx'] = [i, j]
                 pxdic[cont]['intespx'] = sec[i][j]
                 G.add_node(cont)
                 for k in range(0, int(len(ind)/2)):
                        d = 0
                        d = (((ind[k] - i) ** 2) + ((ind[k+1] - j) ** 2)
                             ) + ((sec[i][j] - sec[ind[k]][ind[k+1]]) ** 2)
                        w = ((d/(255)**2)-(r ** 2))
                        if d <= r and w <= t:
                            cont += 1
                            G.add_node(cont)
                            pxdic[cont] = dict()
                            pxdic[cont]['pospx'] = [k, k+1]
                            pxdic[cont]['intespx'] = sec[k][k+1]
                            G.add_edge(base, cont, weight=w)

         print("Calculo dos pesos finalizado.")

         gs.append(G)

         if opc == True:
             print("Iniciando a o desenho do grafo...")
             f = plt.figure()
             nx.draw(G)
             f.savefig('grafo_sec%i.png' % (s))
             # plt.show()

         s += 1

     return gs

# utiliza a distancia euclidiana e o peso 1 para calcular
# params:
# sections: Seções da imagem
# opc: quando verdadeiro salva a imagem do grafo
def weights_euclidian_p1(sections, opc=False, r=2, t=0.9):
        s = 1
        gs = []
        for sec in sections:
            print('Iniciando o grafo da secção %i ...' % (s))
            row = np.size(sec, 0)
            col = np.size(sec, 1)

            i = 0
            j = 0

            G = nx.Graph()

            np.seterr(over="ignore")

            # pospx = [i,j] e intespx. representando o pixel
            pxdic = dict()

            print("Iniciando calculo dos pesos...")

            cont = 0
            for i in range(1, row - 2):
                for j in range(1, col - 2):
                    # ind = [i+1, j, i, j+1, i+1, j+1]
                    ind = [i+1, j, i, j+1, i+1, j+1, i-1, j -
                           1, i-1, j, i+1, j+1, i, j-1, i-1, j+1]
                    # ind = r * [i-1, j, i,j-1, i+1, j, i, j+1]
                    base = cont
                    pxdic[cont] = dict()
                    pxdic[cont]['pospx'] = [i, j]
                    pxdic[cont]['intespx'] = sec[i][j]
                    G.add_node(cont)
                    for k in range(0, int(len(ind)/2)):
                        d = 0
                        d = math.sqrt(((ind[k] - i) ** 2) +
                                      ((ind[k+1] - j) ** 2))
                        w = (
                            255 - math.fabs(sec[i][j] - sec[ind[k]][ind[k+1]]))/255
                        if d <= r and w <= t:
                            cont += 1
                            G.add_node(cont)
                            pxdic[cont] = dict()
                            pxdic[cont]['pospx'] = [k, k+1]
                            pxdic[cont]['intespx'] = sec[k][k+1]
                            G.add_edge(base, cont, weight=w)

            print("Calculo dos pesos finalizado.")

            gs.append(G)

            if opc == True:
                print("Iniciando a o desenho do grafo...")
                f = plt.figure()
                nx.draw(G)
                f.savefig('grafo_sec%i.png' % (s))
                # plt.show()

            s += 1

        return gs

# utiliza a distancia euclidiana e o peso 1 para calcular
# params:
# sections: Seções da imagem
# opc: quando verdadeiro salva a imagem do grafo
def weights_euclidian_p2(sections, opc=False, r=2, t=0.9):
        s = 1
        gs = []
        for sec in sections:
            print('Iniciando o grafo da secção %i ...' % (s))
            row = np.size(sec, 0)
            col = np.size(sec, 1)

            i = 0
            j = 0

            G = nx.Graph()

            np.seterr(over="ignore")

            # pospx = [i,j] e intespx. representando o pixel
            pxdic = dict()

            print("Iniciando calculo dos pesos...")

            cont = 0
            for i in range(1, row - 2):
                for j in range(1, col - 2):
                    # ind = [i+1, j, i, j+1, i+1, j+1]
                    ind = [i+1, j, i, j+1, i+1, j+1, i-1, j -
                           1, i-1, j, i+1, j+1, i, j-1, i-1, j+1]
                    # ind = r * [i-1, j, i,j-1, i+1, j, i, j+1]
                    base = cont
                    pxdic[cont] = dict()
                    pxdic[cont]['pospx'] = [i, j]
                    pxdic[cont]['intespx'] = sec[i][j]
                    G.add_node(cont)
                    for k in range(0, int(len(ind)/2)):
                        d = 0
                        d = math.sqrt(((ind[k] - i) ** 2) +
                                      ((ind[k+1] - j) ** 2))
                        w = ((((ind[k] - i)**2) + ((ind[k+1] - j)**2)) + ((r**2) * ((math.fabs(sec[i][j] - sec[ind[k]][ind[k+1]]))/255))/(2*(r**2)))
                        if d <= r and w <= t:
                            cont += 1
                            G.add_node(cont)
                            pxdic[cont] = dict()
                            pxdic[cont]['pospx'] = [k, k+1]
                            pxdic[cont]['intespx'] = sec[k][k+1]
                            G.add_edge(base, cont, weight=w)

            print("Calculo dos pesos finalizado.")

            gs.append(G)

            if opc == True:
                print("Iniciando a o desenho do grafo...")
                f = plt.figure()
                nx.draw(G)
                f.savefig('grafo_sec%i.png' % (s))
                # plt.show()

            s += 1

        return gs

# params:
# G: Grafo gerado
# opc: quando verdadeiro exibe o grafico do histograma
def calc_histDeg(G, opc=False):

        hdeg = nx.degree_histogram(G)

        if opc == True:
            degree_sequence = [d for n, d in G.degree()]
            degreeCount = collections.Counter(degree_sequence)
            deg, cnt = zip(*degreeCount.items())
            fig, ax = plt.subplots()
            plt.bar(deg, cnt, width=0.80, color='b')
            plt.title("Degree Histogram")
            plt.ylabel("Count")
            plt.xlabel("Degree")
            ax.set_xticks([d + 0.4 for d in deg])
            ax.set_xticklabels(deg)
            fig.savefig("hisgram_degree.png")

        return hdeg

def dens_prob(hst):

       dens = []

       for i in hst:
           d = round((i/sum(hst)), 4)
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

        entropy *= (-1)

        return (mean, entropy, energy, contrast)

# params:
# path: caminho da imagem
# type_img: (1) carrega imagem biblioteca dicom python
# (2) carrega a imagem a partir de um script matlab
# (3) carrega aimagem a partir de uma imagem conversional

def extract_texture(path, type_img=1):

        print("Iniciando o algoritmo...")

        # /Users/erjulioaguiar/Documents/siim-medical-image-analysis-tutorial/dicom_dir/ID_0069_AGE_0074_CONTRAST_0_CT.dcm

        # /home/erikson/Documentos/Dataset/LUNG1-001/09-18-2008-StudyID-69331/0-82046/000035.dcm

        if type_img == 1:
            sections = load_dicompy(path, slices=True)

        elif type_img == 1:
            sections = load_dicom_script(path, opc=2)

        else:
            sections = load_img(path)

        gfs = calc_weights_default(sections)

        g_metric = []

        for g in gfs:
            d_prob = []
            d = calc_histDeg(g)
            d_prob = dens_prob(d)
            me, etr, enr, ctr = metrics_rc(d_prob)
            m_aux = [me, etr, enr, ctr]
            g_metric += m_aux

        #print(g_metric)

        gc.collect()

        print("Algoritmo finalizado!!!")

        return g_metric
