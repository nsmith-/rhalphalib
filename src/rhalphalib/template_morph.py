from scipy.interpolate import interp1d
import numpy as np


class AffineMorphTemplate(object):
    def __init__(self, hist):
        """
        hist: a numpy-histogram-like tuple of (sumw, edges)
        """
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


class MorphHistW2(object):
    """
    Extends AffineMorphTemplate to shift variances as well

    Parameters
    ----------
    object : hist object or tuple
    """

    def __init__(self, hist):
        """
        hist: uproot/UHI histogram or a tuple (values, edges, variances)
        """
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

    def get(self, shift=0.0, smear=1.0):
        nom, edges, _ = self.nominal.get(shift, smear)
        w2s, edges, _ = self.w2s.get(shift, smear)
        if self.return_hist:  # return hist object
            import hist

            h = hist.Hist(
                hist.axis.Variable(edges, name=self.varname),
                storage="weight",
            )
            h.view(flow=False).value = nom
            h.view(flow=False).variance = w2s
            return h
        else:  # return tuple
            return nom, edges, self.varname, w2s
