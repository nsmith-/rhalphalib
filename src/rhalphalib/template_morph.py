from typing import Tuple, Union
from scipy.interpolate import interp1d
import numpy as np
import hist


class AffineMorphTemplate(object):
    """Affine morphing of a histogram

    Parameters:
    hist: a numpy-histogram-like tuple of (sumw, edges, varname)
    """

    def __init__(self, hist: tuple[np.ndarray, np.ndarray, str]):
        self.sumw = hist[0]
        self.edges = hist[1]
        self.varname = hist[2]
        self.centers = self.edges[:-1] + np.diff(self.edges) / 2
        self.norm = self.sumw.sum()
        self.mean = (self.sumw * self.centers).sum() / self.norm
        self.cdf = interp1d(
            x=self.edges,
            y=np.r_[0, np.cumsum(self.sumw / self.norm)],
            kind="linear",
            assume_sorted=True,
            bounds_error=False,
            fill_value=(0, 1),
        )

    def get(self, shift=0.0, smear=1.0):
        """
        Return a shifted and smeard histogram
        i.e. new edges = edges * smear + shift
        """
        if not np.isclose(smear, 1.0):
            shift += self.mean * (1 - smear)
        smeard_edges = (self.edges - shift) / smear
        return np.diff(self.cdf(smeard_edges)) * self.norm, self.edges, self.varname


_HistType = Union[hist.Hist, Tuple[np.ndarray, np.ndarray, str, np.ndarray]]


class MorphHistW2(object):
    """
    Extends AffineMorphTemplate to shift variances as well

    Parameters:
        hist: uproot/UHI histogram or a tuple (values, edges, variances)
    """

    def __init__(self, hist: _HistType):
        try:  # hist object
            self.sumw = hist.values()
            self.edges = hist.axes[0].edges
            self.varname = hist.axes[0].name
            self.variances = hist.variances()
            self.return_hist = True
        except:  # tuple  # noqa
            self.sumw = hist[0]
            self.edges = hist[1]
            self.varname = hist[2]
            self.variances = hist[3]
            self.return_hist = False

        self.nominal = AffineMorphTemplate((self.sumw, self.edges, self.varname))
        self.w2s = AffineMorphTemplate((self.variances, self.edges, self.varname))

    def get(self, shift=0.0, smear=1.0) -> _HistType:
        nom, edges, _ = self.nominal.get(shift, smear)
        w2s, edges, _ = self.w2s.get(shift, smear)
        if self.return_hist:  # return hist object
            h = hist.Hist(
                hist.axis.Variable(edges, name=self.varname),
                storage="weight",
            )
            h.view(flow=False).value = nom
            h.view(flow=False).variance = w2s
            return h
        else:  # return tuple
            return nom, edges, self.varname, w2s
