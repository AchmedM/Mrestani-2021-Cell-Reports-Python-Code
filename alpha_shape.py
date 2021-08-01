from CGAL.CGAL_Kernel import Point_2
from CGAL.CGAL_Alpha_shape_2 import Alpha_shape_2
from CGAL.CGAL_Alpha_shape_2 import GENERAL, REGULAR
import numpy as np
def get_alpha_shape(coords,alpha=1000):

    points=[]
    for p in coords:
        points.append(Point_2(*p))

    als = Alpha_shape_2(points,alpha,GENERAL)
    als.make_alpha_shape(points)
    return als

def get_alpha_edges(coords,alpha):
    lines=[]
    alpha_shape=get_alpha_shape(coords,alpha=alpha)
    alpha_shape.set_alpha(alpha)
    alpha_shape.set_mode(1)
    for i,h in enumerate(alpha_shape.alpha_shape_edges()):
        if i==0:
            done = h
        elif h==done:
            break
        s=alpha_shape.segment(h)
        lines.append([[s.max().x(),s.max().y()],[s.min().x(),s.min().y()]])
        
       
    return lines

def get_alpha_triangles(coords,alpha):
    alpha_shape=get_alpha_shape(coords,alpha=alpha)
    alpha_shape.set_alpha(alpha)
    alpha_shape.set_mode(1)
    triangles=[]
    for i,h in enumerate(alpha_shape.finite_faces()):
        if i==0:
            done = h
        elif h==done:
            break
        
        if not alpha_shape.classify(h):
            continue
        triangle=[]
        for i in range (3):
            
            p=h.vertex(i).point()
            triangle.append([p.x(),p.y()])
        triangles.append(triangle)
    return triangles

def get_alpha_area(coords,alpha):
    triangles = get_alpha_triangles(coords,alpha)
    if len(triangles)>0:
        ntriangles=np.ones((len(triangles),3,3))
        ntriangles[:,:,1:]=triangles
        
        A=(sum(np.linalg.det(ntriangles))/2)
    else: 
        A=0
    return A


        
