import matplotlib.pyplot as plt
import hist
from typing import List, Union
import fnmatch
import itertools


def plot_cov(
    fitDiagnostics_file="fitDiagnostics.root",
    out=None,
    include: Union[str, List[str], None] = None,
    data=False,
    year=2017,
):
    import ROOT as r

    rf = r.TFile.Open(fitDiagnostics_file)
    h2 = rf.Get("fit_s").correlationHist()

    x_bins = h2.GetXaxis().GetNbins()
    y_bins = h2.GetYaxis().GetNbins()
    y_labels = [h2.GetYaxis().GetBinLabel(i) for i in range(1, y_bins + 1)]
    x_labels = [h2.GetXaxis().GetBinLabel(i) for i in range(1, x_bins + 1)]
    hist_2d = hist.new.StrCat(x_labels, label="").StrCat(y_labels, label="").Double()
    for i in range(0, x_bins):
        for j in range(0, y_bins):
            hist_2d.view()[i, j] = h2.GetBinContent(i + 1, j + 1)

    if include == "all":
        keys = [lab for lab in x_labels]
    elif include == "tf":
        keys = [lab for lab in x_labels if not (lab.startswith("qcdparam") or "mcstat" in lab)]
    elif include is None:
        keys = [lab for lab in x_labels if not (lab.startswith("qcdparam") or "mcstat" in lab or lab.startswith("tf"))]
    elif isinstance(include, str) or isinstance(include, list):
        # check for fnmatch wildcards
        if not isinstance(include, list):
            include = [include]
        if any(any(special in pattern for special in ["*", "?"]) for pattern in include):
            keys = []
            for pattern in include:
                keys.append([k for k in x_labels if fnmatch.fnmatch(k, pattern)])
            keys = list(dict.fromkeys(list(itertools.chain.from_iterable(keys))))
        else:
            keys = include
    else:
        keys = x_labels

    # Plot it
    fig, ax = plt.subplots()
    hist_2d[keys, keys].plot2d(cmap="RdBu", cmin=-1, cmax=1, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=30, horizontalalignment="right")
    ax.minorticks_off()
    ax.set_ylabel("")
    ax.set_xlabel("")

    if out is None:
        return fig
    else:
        fig.savefig(out, bbox_inches="tight")
