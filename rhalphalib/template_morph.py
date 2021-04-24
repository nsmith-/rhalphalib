from scipy.interpolate import interp1d
import numpy as np


class AffineMorphTemplate(object):
    def __init__(self, hist):
        '''
        hist: a numpy-histogram-like tuple of (sumw, edges, name)
        '''
        self.sumw = hist[0]
        self.edges = hist[1]
        self.varname = hist[2]
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
        values = np.diff(self.cdf(scaled_edges)) * self.norm
        return values.clip(min=0), self.edges, self.varname
