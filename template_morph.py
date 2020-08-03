from scipy.interpolate import interp1d
import numpy as np


class AffineMorphTemplate(object):
    def __init__(self, hist):
        '''
        hist: a numpy-histogram-like tuple of (sumw, edges, name)
        '''
        self.sumw, self.edges, self.varname = hist
        self.norm = self.sumw.sum()
        self.mean = (self.sumw*(self.edges[:-1] + self.edges[1:])/2).sum() / self.norm
        self.cdf = interp1d(x=self.edges,
                            y=np.r_[0, np.cumsum(self.sumw / self.norm)],
                            kind='linear',
                            assume_sorted=True,
                            bounds_error=False,
                            fill_value=(0, 1),
                           )
        
    def get(self, shift=0., scale=1.):
        '''
        Return a shifted and scaled histogram
        i.e. new edges = edges * scale + shift
        '''
        scaled_edges = (self.edges - shift) / scale
        return np.diff(self.cdf(scaled_edges)) * self.norm, self.edges, self.varname
