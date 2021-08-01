import selection_and_binning as sb
from collections import OrderedDict
import sr_io
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
import pdb


class WorkflowException(Exception):
    pass


class Selection(object):
    """
        keeps the indeces of a selection and wraps useful functions

    """

    def __init__(self, dataframe=None,  indices=None, filename=None, selection=None, bins=None, metadata=None, name=None):
        """
           make a selection on the dataframes by specifying indices, if no indeces are provided, all indices are used.
           in case
           sometimes it is useful to to work in an image space. If you want to be able to overlap images of two selection, you should use the same bins
        """
        self._name = None
        if name is None:
            name = 0
        self._attributes = pd.DataFrame(index=[name])
        self.series = {}
        self.name = name
        self.filename = filename
        if metadata is None:
            metadata = {}
        self.metadata = metadata
        if not selection is None:
            self._dataframe = selection._dataframe
            if indices is None:
                self.indices = selection.indices
            else:
                self.indices = indices
            self.metadata = copy.deepcopy(selection.metadata)
            if not metadata is None:
                self.metadata.update(metadata)
        else:
            if not dataframe is None:
                self._dataframe = dataframe

            if not dataframe is None:
                if indices is None:
                    indices = dataframe.index
                self.indices = indices
            self.metadata = metadata

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name
        self._attributes.index = [name]

    @property
    def data(self):
        return self._dataframe.loc[self.indices]

    @data.setter
    def data(self, data):
        self._dataframe = data
        self.indices = data.index

    def load_data(self):
        self.data = sr_io.load_rapidstorm_file(self.filename)

    def select_between(self, kdims, roi):
        indices = sb.select_between(self.data, kdims, roi, 'selection')
        return Selection(self._dataframe, indices=indices, \
                         metadata=copy.deepcopy(self.metadata))

    def set_spatial_bins(self, keys=['x', 'y'], spacing=(10, 10), origin=None, \
                         maxv=None, pad_width=0):
        self.spatial_bins = sb.get_bins(
            self.data, keys=keys, spacing=spacing, origin=origin, maxv=maxv, \
            pad_width=pad_width)

    def get_spatial_statistic(self, kdims=['x', 'y'], vdim='Amplitude',\
                              statistic='count', spacing=None, origin=None, pad_width=0, soft=False):
        """
            calculate a historamm style image, or do an aggregation.
            if no spacing is provided self.spacial_bins will be used set by self.set_bins first
        """
        if spacing is None and origin is None:
            bins = self.spatial_bins
        else:
            bins = sb.get_bins(self.data, keys=['x', 'y'],\
                               spacing=spacing, origin=origin)
        return sb.get_spatial_statistic(self.data, bins, kdims=kdims, vdim=vdim,\
                            statistic=statistic, spacing=spacing, origin=origin,\
                            pad_width=pad_width, soft=soft)

    def show_statistic(self, kdims=['x', 'y'], bins=None, vdim=None, axlog=False,\
                       log=False, statistic='count', ax=None, im='imshow', label=None):
        """
        plots binned statistics. for 2d plotting can be done with imshow or pcolor.\
         use the later for correct labels
        """
        return sb.show_statistic(self.data, kdims=kdims, bins=bins, vdim=vdim,\
                                 axlog=axlog, log=log, statistic=statistic, \
                                 ax=ax, im=im, label=label, )

    def get_binned_statistic(self, kdims=['x', 'y'], bins=None, vdim='Amplitude'\
                     , axlog=False, statistic='count', to_series=True, name=None,\
                      set_series=True):
        stats = sb.get_binned_statistic(self.data,  kdims=kdims,
                                        bins=bins, vdim=vdim, axlog=axlog, statistic=statistic)
        if len(kdims) == 1 and to_series:
            a = stats

            ind = (a.bin_edges[0][1:]+a.bin_edges[0][:-1])/2.

            name = '_'.join([kdims[0], 'vs', vdim, statistic])
            if set_series:

                self.series[name] = pd.Series(a.statistic, index=ind)
            else:
                return pd.Series(a.statistic, index=ind)
        else:
            return stats

    def get_bins_rapidstorm(self, spacing=(10, 10)):
        fi = self.filename
        xmin = float(sr_io.get_rapidstorm_attrib(fi, 'min')[0][:-2])*1e9
        xmax = float(sr_io.get_rapidstorm_attrib(fi, 'max')[0][:-2])*1e9
        ymin = float(sr_io.get_rapidstorm_attrib(fi, 'min')[1][:-2])*1e9
        ymax = float(sr_io.get_rapidstorm_attrib(fi, 'max')[1][:-2])*1e9
        self.set_spatial_bins(origin=(xmin, ymin), spacing=spacing, maxv=(xmax, ymax))

    def get_masked_data(self, mask, kdims=['x', 'y']):
        """
        returns a selection based on a mask which need to have the same bins as this selection

        """
        indices = sb.get_masked_data(self.data, mask, self.spatial_bins,
                                     kdims=kdims, return_type='selection')
        return Selection(selection=self, indices=indices)

    def get_cluster(self, kdims=['x', 'y'], scale=1, eps=20, min_samples=10,\
                    kind='DBSCAN', **kwargs):
        """
        get a collection from a clustering algorithmn. scale can be a list to
         change the weights of the dimensions.
        """
        if kind == 'DBSCAN':
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, **
                               kwargs).fit(self.data[kdims].values*scale)
            labels = clusterer.labels_

        if kind == 'HDBSCAN':
            clusterer = HDBSCAN(min_samples=min_samples, **
                                kwargs).fit(self.data[kdims].values*scale)
            labels = clusterer.labels_
        nc = Groups(parent=self, grouper=labels)
        nc.metadata['cluster_params'] = {'kdims': kdims, 'scale': scale,
                                         'eps': eps, 'min_samples': min_samples,\
                                          'kind': kind, 'kwargs': kwargs}
        nc.clusterer = clusterer
        return nc

    def append_data(self, appendframe):
        self.data = pd.merge(self.data, appendframe, how='outer')

    def filter_time_cluster(self, kdims=['x', 'y', 'ImageNumber'], \
                            scale=[1, 1, 30], eps=150, min_samples=2):
        time_clusters = self.get_cluster(kdims, scale=scale, eps=eps, min_samples=min_samples)

        quant = time_clusters.agg('mean', suffix=False)

        count = time_clusters.count()
        count.loc[-1] = 1
        time_clusters.data = pd.concat([quant, count], 1).iloc[0:]
        # nc2=Selection(col2.group_to_parent(col2.data.keys()[2:]))

        single = time_clusters.get_group(-1).data
        single['Count'] = 1
        new = Selection(pd.concat([time_clusters.data, single]).reset_index(drop=True))

        return new


class Collection(Selection):
    """
    holds a dictionary of indices
    """

    def __init__(self, dataframe=None, bins=None):
        super(Collection, self).__init__(dataframe, indices=None, bins=None)
        self.selections = OrderedDict()
        self.dataframes = {}

    def length(self):
        return len(self.selections.keys())

    def addSelection(self, selection, name=None):

        if name is None:
            if selection._attributes.index == 0:
                name = self.length()
                while name in self.selections.keys():
                    name += 1
            else:
                name = selection._attribuses.index

        if name in self.selections.keys():
            raise WorkflowException('selection name already in collection')
        self.selections[name] = selection

    def apply(self, func, kdims=None, axis=0, suffix=True, *args, **kwargs):

        result = {}
        for name, selection in self.selections.items():
            dg = selection.data
            if kdims is None:
                data = dg
            else:
                data = dg[kdims]
            result[name] = data.apply(func, axis, args, kwargs)
        df = pd.DataFrame(result).transpose()
        if suffix:
            if suffix is True and isinstance(func, type('')):
                suffix = func
            elif not isinstance(func, type('')):
                suffix = '_'
            df = df.rename(columns={old: '_'.join([old, suffix]) for old in kdims})
        return df

    def count(self):
        result = {}
        for name, selection in self.selections.items():

            result[name] = len(selection.indices)
        return pd.Dataframe({'Count': pd.Series(result)})  # .transpose()

    def col_to_parent(self, kdims=[]):

        alld = []
        for name, selection in self.selections.items():
            if kdims is None:
                kdims = self.data.keys()
            dg = selection.data
            for k in kdims:
                dg[k] = self.data[k].loc[name]
            alld.append(dg)
        return pd.concat(alld, axis=0)

    def retrieve_attributes(self, kdims=None, reindex=True, set_data=False):
        results = OrderedDict()
        for name, selection in self.selections.items():
            if kdims is None:
                result[name] = selection._attributes
            else:
                result[name] = selection._attributes[kdims]
        if reindex:
            data = pd.concat(result.values(), ignore_index=True)
            data.index = result.keys()
        else:
            data = pd.concat(result.values())
        if set_data:
            self.dataframe = data
        else:
            return data

    def retrieve_series(self, kdims=None):
        results = OrderedDict()
        for name, selection in self.selections.items():

            if kdims is None:
                kdims = selection.series.keys()
            for key in kdims:
                if not key in results.keys():
                    results[key] = pd.DataFrame()
                # if name in results[key].keys():
                results[key][name] = selection.series[key]
        return results

    def retrieve_metadata(self, kdims=None, set_data=False, reindex=False):
        results = OrderedDict()
        for name, selection in self.selections.items():
            if kdims is None:
                results[name] = pd.DataFrame(selection.metadata, index=[selection.name])
            else:
                results[name] = pd.DataFrame(selection.metadata[kdims], index=[selection.name])
        if reindex:
            data = pd.concat(results.values(), ignore_index=True)
            data.index = results.keys()
        else:
            data = pd.concat(results.values())
        if set_data:
            self.dataframe = data
        else:
            return data

    def plot_series(self, kdims=None, errorbar=False, legend=True):
        d = self.retrieve_series()
        if kdims is None:
            kdims = d.keys()
        for k in kdims:
            plt.figure()
            df = d[k]
            if errorbar:
                plt.errorbar(x=a.index, y=a.mean(axis=1), yerr=a.std(axis=1))
            else:

                lineobs = plt.plot(df)
                if legend:
                    plt.legend(iter(lineobs), df.keys())
            xl, blub, yl = k.split('_', 2)
            if yl[-5:] == 'count':
                yl = 'count'
            plt.xlabel(xl)
            plt.ylabel(yl)

    def join(self):
        ad = []
        for sel in self.selections.values():
            ad.append(sel.data)
        ad = pd.concat(ad).reset_index(drop=True)
        return Selection(ad)


class Groups(Collection):
    """
    creates an Collection using groupby alowing optimized aggregation.
    """
    """
    def __init__(self,parent=None, selectionIndices=None, dataframe=None, bins= None, ):
        super(Collection,self).__init__(dataframe, indices=None, bins = None)

        self.parent=parent
        if parent is None:
            self._parentdata=None
        else:
            self._parentdata=parent.data
        if selectionIndices is None:
            self.selectionIndices={}
        else:
            self.selectionIndices=selectionIndices
    """

    def __init__(self, parent, grouper, dataframe=None, bins=None, clusterer=None):
        super(Groups, self).__init__(dataframe, bins=None)

        self.parent = parent
        if parent is None:
            self._parentdata = None
        else:
            self._parentdata = parent.data

        self.grouper = grouper
        self.selectionIndices = self.groupbyiterator().groups

    def get_group(self, name):

        # .get_group(name),bins=self.bins)
        return Selection(self.parent.data.loc[self.selectionIndices[name]])

    def length(self):
        return len(self.groupbyiterator().keys)

    def groupbyiterator(self):
        return self._parentdata.groupby(self.grouper)

    def agg(self, func, kdims=None, suffix=True, ravel=True):
        if kdims == None:
            d = self.groupbyiterator()
        else:
            d = self.groupbyiterator()[kdims]
        quant = d.agg(func)
        mi = quant.columns
        ncol = None
        if type(mi) == pd.MultiIndex:
            if ravel == True:
                ncol = pd.Index(['_'.join([e[0], e[1]]) for e in mi.tolist()])
        else:
            if suffix:
                ncol = pd.Index(['_'.join([e, func]) for e in mi.tolist()])
        if not ncol is None:
            quant.columns = ncol
        return quant

    def count(self):
        return pd.DataFrame({'Count': self.groupbyiterator().size()})

    def get_parent_labels(self):
        """
        returns
        """
        return np.array([name for name, unused_df in self.groupbyiterator()])[self.groupbyiterator().ngroup()]  # ngroup provides the group of each line in parentdata

    def group_to_parent(self, kdims=[], ignore_noise=False, ret_label=True):
        label = self.get_parent_labels()
        d = pd.DataFrame({'label': label})
        d.index = self.parent.indices
        if len(kdims) == 0:
            d2 = d
        else:
            dc = self.data
            if ignore_noise:
                dc = dc.loc[0:]

            d2 = pd.merge(pd.DataFrame(d.label), dc[kdims], right_index=True, \
                          left_on=['label'])

        if not ret_label:
            d2 = d2['keys']
        """
        d=self._parentdata
        if 'label' in kdims:
            d['label']= self.groupbyiterator().ngroup()-1
            print 'warning using ngroup -1'
            kdims.remove('label')
        if len(kdims):
            for k in kdims:
                d=d.assign(**{k:np.nan})
            for label, groupindex in self.groupbyiterator().groups.items():
                if ignore_noise and label==-1:
                    continue
                #x=self.data[kdims].loc[label].transpo
                d[kdims].loc[groupindex] = pd.DataFrame(self.data[kdims].loc[label]).transpose()
        """
        return pd.concat([self._parentdata, d2], axis=1)

    def fast_analysis(self, set_data=False, reduce_noise=True):
        quant = self.agg(['mean', 'std'])
        count = self.count()
        if reduce_noise:
            if -1 in count.index:
                count.loc[-1] = 1
        d = pd.concat([quant, count], 1)
        if set_data:
            self.data = d
        else:
            return d

    def return_nonoise(self):
        sel = Selection(selection=self.parent,
                        indices=self.parent.indices[self.clusterer.labels_ >= 0])
        sel.metadata['noise_filter'] = self.metadata['cluster_params']
        return sel


"""
TODO:
    + metadata
    + copy()
"""
