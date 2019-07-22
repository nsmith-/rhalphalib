import ROOT
from coffea import hist
import numpy as np


def _to_numpy(hinput):
    if isinstance(hinput, ROOT.TH1):
        sumw = np.zeros(hinput.GetNBinsX())
        binning = np.zeros(sumw.size + 1)
        name = hinput.GetName()
        for i in range(1, sumw.size + 1):
            sumw[i] = hinput.GetBinContent(i)
            binning[i] = hinput.GetXaxis().GetBinLowEdge(i)
        binning[i+1] = hinput.GetXaxis().GetBinUpEdge(i)
        return (sumw, binning, name)
    elif isinstance(hinput, hist.Hist):
        sumw = hinput.values()[()]
        binning = hinput.axes()[0].edges()
        name = hinput.axes()[0].name
        return (sumw, binning, name)
    elif isinstance(hinput, tuple) and len(hinput) == 3:
        return hinput
    else:
        raise ValueError


def _to_TH1(sumw, binning, name):
    h = ROOT.TH1D(name, "template;%s;Counts" % name, binning.size - 1, binning)
    h.SetDirectory(0)
    for i, w in enumerate(sumw):
        h.SetBinContent(i + 1, w)
    return h
