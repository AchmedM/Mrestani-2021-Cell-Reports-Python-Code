import numpy as np
from collections import OrderedDict
import pdb
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import morphology as morph
from scipy.ndimage import label
import scipy.ndimage as nd
from scipy.stats import binned_statistic_dd
#from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.kernel_density import KDEMultivariate

import copy   
#try:
  

def get_bins(data,keys=['x','y'], spacing=(10,10), origin=None, maxv=None, pad_width=0):
    d=[]
    for k in keys:
        d.append(data[k])
      
    if maxv is None:
        maxv=[] 
        for k in keys:
            maxv.append(data[k].max())
        
    if origin is None:
        origin=[] 
        for k in keys:
            origin.append(data[k].min())
    maxv = np.array(maxv)     
    origin = np.array(origin)     
    if pad_width:
            origin -= spacing*pad_width
            maxv += spacing*pad_width
    edges = []#OrderedDict()
    for i,k in enumerate(keys):
        edges.append(np.arange(origin[i], maxv[i] + spacing[i], spacing[i]) )
    return edges

def get_spatial_statistic(data,bins = None, kdims=['x','y'],vdim='Amplitude', statistic='count', spacing=(10,10), origin=None, maxv=None, 
                           pad_width=0,soft=False):
    """
    perfoms binning and aggregation with the idea that it makes an image of the pointcloud in a rapidstorm fashion . 
    when non bins are provided, the bins will be created for columns *kdims* with given *spacing* when *statistic*=='count', a
    normal histogram is calculated. If vdim=None, an array with ones is used as weights
    
    for aggregation methods check th documentation of scipy.stats.binned_statistic_dd.
    when *soft* is true, aich weight will be destibuted between two neighboring bins. (like convolution with a triangular kernel)
    for this the amount of data will temporarly increase by afactor of two for each dimension 
    
    """
    #if soft:
    #    pad_width+=1
    if bins is None:
        bins = get_bins(data,keys=kdims, spacing=spacing,origin=origin,maxv=maxv,pad_width=pad_width)
        
    if soft:
        if statistic not in ['sum','count']:
            print('be aware agrregation might not make sense for soft edges')
        #pad_width+=1
        data_copy  =data[kdims].as_matrix()
        if statistic == 'count':
             vdim = None
             statistics = 'sum'
        if not vdim is None:
            weights =data[vdim].values
        else: 
            weights = np.ones_like(data_copy[:,0])
        
        for i,ei in enumerate(bins):
           
            ci = (ei[:-1]+ei[1:])/2 # center of bin
            wi = (ei[1]-ei[0])  #width of bin
            
            dl = data_copy[:,i]-wi/2 # data contributing to lower bin
            #relative_position
            lm2 =data_copy[:,i]-ci[0] 
            frac = (lm2 % wi)/wi
           
            dh = dl + wi #data contibuting to higher bin
            data_copy = np.r_[data_copy,data_copy] #create space
            data_copy[:,i] = np.r_[dl,dh]  
           
            # calculate weights
            
            weights= np.r_[weights*frac,weights*(1-frac)]
            #print weights.sum()
            #print lm.min(), lm.max()
            #print ei[0], ei[-1]
            #pdb.set_trace()
    else:
        data_copy =data[kdims].values
        if not vdim is None:
            weights =data[vdim].values
        else: 
            weights = np.ones_like(data_copy[:,0])
       
            
        #bins = get_bins(data,keys=kdims, spacing=spacing,origin=origin,maxv=maxv,pad_width=pad_width)
    if soft and statistic=='count':
        statistic='sum' #to account for weights     
    return binned_statistic_dd(data_copy,weights,statistic=statistic,bins=bins)

def get_binned_statistic(data, kdims=['x','y'],bins = None, vdim=None, axlog=False, statistic='count', ):
    """
    perfoms binning and aggregation. 
    when non bins are provided, the bins will be created for columns *kdims* with given *spacing* when *statistic*=='count', a
    normal histogram is calculated. If vdim=None, an array with ones is used as weights
    
    
    for aggregation methods check th documentation of scipy.stats.binned_statistic_dd.
    """

    data_copy =data[kdims].values
    if not vdim is None:
        weights =data[vdim].values
    else: 
        weights = np.ones_like(data_copy[:,0])
    if not axlog is False:
        nb=[]
        for i, l in enumerate(axlog):
            if l == True:
                if bins is None:
                    nb.append(np.logspace(data_copy[:,i].min(),data_copy[:,i].max()))
                elif not type(bins) == type([]):
                     nb.append(np.logspace(np.log10(data_copy[:,i].min()),np.log10(data_copy[:,i].max()),bins))
            
                else: 
                    nb.append(np.logspace(np.log10(data_copy[:,i].min()),np.log10(data_copy[:,i].max()),bins[i]))
            else:
                if bins is None:
                    nb.append(np.linspace(data_copy[:,i].min(),data_copy[:,i].max(),bins[i]))
                elif not type(bins)== type([]):
                    nb.append(np.linspace(data_copy[:,i].min(),data_copy[:,i].max(),bins))
                else:
                    if np.size(bins[i])==1:                  
                        nb.append(np.linspace(data_copy[:,i].min(),data_copy[:,i].max(),bins[i]))
                    else:
                        nb.append(bins[i])
        #stop        
        bins = nb
            
        #bins = get_bins(data,keys=kdims, spacing=spacing,origin=origin,maxv=maxv,pad_width=pad_width)
        
    return binned_statistic_dd(data_copy,weights,statistic=statistic,bins=bins)

def show_statistic(data,kdims=['x','y'],bins = None,vdim=None, log=False, axlog=False, statistic='count', ax=None, im='imshow', label=None):
        """
        plots binned statistics. for 2d plotting can be done with imshow or pcolor. use the later for correct labels
        pcolor does not deal well with nan. so the nan values will be set to the minimum
        """
        stats=get_binned_statistic(data=data,  kdims=kdims,bins = bins, vdim=vdim,  axlog=axlog, statistic=statistic )
        if ax is None:
            ax = plt.gca()
        pos=[]
        for i ,e in enumerate(stats.bin_edges):
            if axlog is False or axlog[i]==0:
                pos.append((e[:-1]+e[1:])/2.)
            else: 
                e=np.log(e)
                m=(e[:-1]+e[1:])/2.
                pos.append(np.exp(m))
        
        if len(kdims)==1:
            plt.plot(pos[0],stats.statistic, label=label)
            plt.xlabel(kdims[0])
            lab = statistic
            if not vdim is None:
                lab=' '.join ([lab,vdim]) 
            plt.ylabel(lab)
            if log:
                plt.semilogy()
            if not axlog is False and axlog[0]==True and log==False:
                plt.semilogx()
            elif not axlog is False and axlog[0]==True and log==True:
                plt.loglog()
        if len(kdims)==2:
            if im == 'imshow':
                if log == False:
                    plt.imshow(stats.statistic)
                else:
                    plt.imshow(np.log10(stats.statistic))
            if im == 'pcolor':
                Y,X=np.meshgrid(*stats.bin_edges[::-1])
                s=stats.statistic.copy()
                s[np.isnan(s)]=np.min(s[np.isnan(s)==0])
                if log == False:
                    plt.pcolor(X,Y,s)

                else:
                    s[s<=0]=np.min(s[s>0])
                    plt.pcolor(X,Y,s)
                if not axlog is False :
                    if axlog[0]==True and axlog[1]==False:
                         plt.semilogx()
                    elif axlog[1]==True and axlog[0]==False:
                         plt.semilogy()
                    elif axlog[1]==True and axlog[1]==True:
                         plt.loglog()
            plt.xlabel(kdims[0])
            plt.ylabel(kdims[1])

        return stats




def get_index(data, bins,kdims=['x','y']):
    return [np.digitize(data[k].values, b,right=True)-1 for k,b in zip(kdims,bins)]
    
    

def get_masked_data(data, mask, bins,kdims=['x','y'],return_type='data'):
    """
    applies a binary masked on the data
    if *return_type* == 'selection' the indeces will be returned, if return_type == 'data', a copy of the selected data is returned if return_type == 'mask', the mask is returned
    """
    idx = get_index(data, bins, kdims=kdims)
    in_mask = mask[idx]
    mask=np.ones(len(data),dtype='bool')
  
    if return_type=='mask':
       return in_mask
    if return_type=='data':
        return data.loc[in_mask]
    if return_type=='selection':
       return data.index[in_mask]
    #return data[in_mask]    

def select_between(data, kdims, roi, return_type='data'):
    """rectangular selection
    *data is pandas dataframe, *kdims* the names of the column, *roi* is list of tuples containin min,max values, may be None
    if *return_type* == 'selection' the indeces will be returned, if return_type == 'data', a copy of the selected data is returned if return_type == 'mask', the mask is returned"""
    mask=np.ones(len(data),dtype='bool')
    for i,k in enumerate(kdims):
        roii= list(roi[i])
        if roii[0] is None:
            roii[0]=data[k].min()-1
        if roii[1] is None:
            roii[1]=data[k].max()+1
        mask*=data[k].between(*roii)
        
    if return_type=='mask':
       return mask
    if return_type=='data':
        return data.loc[mask]
    if return_type=='selection':
       return data.index[mask]


def get_kde(data, kdims, bandwidth=1, scale_key=None, transform=None,bins=None):
    if bins is None:
         bins= get_bins(data,keys=kdims)
    assert np.all([bins.keys()[i] == kdims[i] for i in range(len(kdims))])

    if scale_key is not None:

        radiusk=data[scale_key].values
        if trasform is not none:
            radiusk=transform(radiusk)
        bandwidth*=radiusk
    kde= KDEMultivariate(data[kdims].values,'cc',bw=bandwidth)
    linc=[]
    for e in bins.values():
        linc.append((e[1:]+e[:-1])/2)
    c=np.meshgrid(*linc)
    shape=c[0].shape
    coords = [v.ravel() for v in c]
    res=kde.pdf(coords)
    return bins, res.reshape(shape)


    
    
    
