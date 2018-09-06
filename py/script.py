import numpy as np
import scipy as sp
#import cv2
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
from PIL import Image
from oct2py import octave


#lena = cv2.imread("lena.jpeg", 0)
#im = Image.open("laranja.png").convert('L')

#im.show()

#array = np.asarray(im)


#gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)

#graph = image.img_to_graph(im)

#print(graph)

#G = nx.from_scipy_sparse_matrix(graph,create_using=nx.MultiGraph())


#nx.draw(G)
#plt.show()


octave.addpath('/home/erikson/Documentos/Dataset')

X = octave.dicom2gray('/home/erikson/Documentos/Dataset/LUNG1-001/09-18-2008-StudyID-69331/0-82046/000001.dcm')

m_adj = octave.calc_graph(X)


G = nx.Graph(m_adj)

nx.draw(G)
plt.show()
