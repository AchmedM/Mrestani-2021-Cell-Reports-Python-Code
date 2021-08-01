import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import label, regionprops
from scipy.ndimage.morphology import binary_fill_holes

def cms(loc):
    return loc[['x','y']].mean(axis=0).values

def get_eigenvalues(sel):
    X = sel.data.x
    Y = sel.data.y
    M = len(X)
    xm = (X).sum() / M
    ym = (Y).sum() / M

    x0 = (X - xm).ravel()
    y0 = (Y - ym).ravel()

    rr = [x0, y0]
    CC = np.zeros([2, 2])
    # pdb.set_trace()
    for m in range(2):
        for n in range(2):
            CC[m, n] = np.dot(rr[m], rr[n])

    CC /= M
    E = np.linalg.eig(CC)
    E = [E[0].tolist(), E[1].T.tolist()]
    EE = list(zip(*sorted(zip(*E), key=lambda pair: pair[0])))

    E1 = list(EE[1])
    E0 = list(EE[0])
    e1, e2 = np.sqrt(E0)
    return e1, e2

def get_labeled_boutons(I):
    c=I[:,:,0]>I[:,:,1]
    l=label(c==False,8)
    m=(l==1)+(I[:,:,0]==255)*(I[:,:,1]==0)*(I[:,:,2]==0)
    l=label(m==0)
    return l