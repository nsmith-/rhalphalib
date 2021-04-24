import ROOT as r
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep
import root_numpy as rnp


def plot_cov(fitDiagnostics_file='fitDiagnostics.root',
             out='covariance_matrix.png', include=None,
             data=False, year=2017):
    assert include in [None, 'all', 'tf']
    rf = r.TFile.Open(fitDiagnostics_file)
    h2 = rf.Get('fit_s').correlationHist()
    TH2 = rnp.hist2array(h2)

    labs = []
    for i in range(h2.GetXaxis().GetNbins() + 2):
        lab = h2.GetXaxis().GetBinLabel(i)
        labs.append(lab)
    labs = labs[1:-1]  # Remove over/under flows

    if include == 'all':
        sel_labs = [lab for lab in labs]
    elif include == 'tf':
        sel_labs = [lab for lab in labs if not (lab.startswith('qcdparam') or 'mcstat' in lab)]
    else:
        sel_labs = [lab for lab in labs if not (lab.startswith('qcdparam') or 'mcstat' in lab or lab.startswith('tf'))]
    sel_ixes = [labs.index(lab) for lab in sel_labs]

    # Get only values we want
    def extract(arr2d, ix):
        x, y = np.meshgrid(ix, ix)
        return arr2d[x, y]

    cov_mat = extract(np.flip(TH2, axis=1), sel_ixes)

    # Plot it
    fig, ax = plt.subplots(figsize=(12, 10))
    g = sns.heatmap(cov_mat,
                    xticklabels=sel_labs,
                    yticklabels=sel_labs,
                    cmap='RdBu',
                    vmin=-1, vmax=1,
                    ax=ax
                    )
    hep.cms.label(fontsize=23, year=year, data=data)
    g.set_xticklabels(g.get_xticklabels(), rotation=30, horizontalalignment='right')
    g.set_yticklabels(g.get_yticklabels(), rotation=30, horizontalalignment='right')
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-1, -.75, -.5, -.25, 0, .25, .5, .75, 1])
    plt.minorticks_off()

    fig.savefig(out, bbox_inches='tight')
