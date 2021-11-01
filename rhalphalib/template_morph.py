from scipy.interpolate import interp1d
import numpy as np


# class AffineMorphTemplate(object):
#     def __init__(self, hist):
#         '''
#         hist: a numpy-histogram-like tuple of (sumw, edges, name)
#         '''
#         self.sumw = hist[0]
#         self.edges = hist[1]
#         self.varname = hist[2]
#         self.norm = self.sumw.sum()
#         self.mean = (self.sumw*(self.edges[:-1] + self.edges[1:])/2).sum() / self.norm
#         self.cdf = interp1d(
#             x=self.edges,
#             y=np.r_[0, np.cumsum(self.sumw / self.norm)],
#             kind='linear',
#             assume_sorted=True,
#             bounds_error=False,
#             fill_value=(0, 1),
#         )

#     def get(self, shift=0., smear=1.):
#         '''
#         Return a shifted and smeard histogram
#         i.e. new edges = edges * smear + shift
#         '''
#         smeard_edges = (self.edges - shift) / smear
#         values = np.diff(self.cdf(smeard_edges)) * self.norm
#         return values.clip(min=0), self.edges, self.varname


class AffineMorphTemplate(object):
    def __init__(self, hist):
        '''
        hist: a numpy-histogram-like tuple of (sumw, edges)
        '''
        self.sumw = hist[0]
        self.edges = hist[1]
        self.varname = hist[2]
        self.centers = self.edges[:-1] + np.diff(self.edges)/2
        self.norm = self.sumw.sum()
        self.mean = (self.sumw*self.centers).sum() / self.norm
        self.cdf = interp1d(x=self.edges,
                            y=np.r_[0, np.cumsum(self.sumw / self.norm)],
                            kind='linear',
                            assume_sorted=True,
                            bounds_error=False,
                            fill_value=(0, 1),
                           )
        
    def get(self, shift=0., smear=1.):
        '''
        Return a shifted and smeard histogram
        i.e. new edges = edges * smear + shift
        '''
        if not np.isclose(smear, 1.):
            shift += self.mean * (1 - smear)
        smeard_edges = (self.edges - shift) / smear
        return np.diff(self.cdf(smeard_edges)) * self.norm, self.edges, self.varname
     

class MorphHistW2(object):
    def __init__(self, hist):
        '''
        hist: uproot/UHI histogram or a tuple (values, edges, variances)
        '''
        try:
            self.sumw = hist.values
            self.edges = hist.edges
            self.variances = hist.variances
        except:
            self.sumw = hist[0]
            self.edges = hist[1]
            self.varname = hist[2]
            self.variances = hist[3]

        from mplhep.error_estimation import poisson_interval
        down, up = np.nan_to_num(np.abs(poisson_interval(self.sumw, self.variances)), 0.)

        self.nominal = AffineMorphTemplate((self.sumw, self.edges, self.varname))
        self.w2s = AffineMorphTemplate((self.variances, self.edges, self.varname))
        
    def get(self, shift=0., smear=1.):
        nom, edges, _ = self.nominal.get(shift, smear)
        w2s, edges, _ = self.w2s.get(shift, smear)       
        return nom, edges, self.varname, w2s