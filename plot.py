import os, errno
import argparse
from collections import OrderedDict

#import ROOT as r
import uproot

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

import mplhep as hep
plt.style.use([hep.cms.style.ROOT, {'font.size': 24}])
plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input-file",
                    default='hxxModel/fitDiagnostics.root',
                    help="Input fitDiagnostics file")
parser.add_argument("--space",
                    default='prefit',
                    dest='namespace',
                    help="fitDiagnostics namespace to plot")
args = parser.parse_args()


def make_dirs(dirname):
    """
    Ensure that a named directory exists; if it does not, attempt to create it.
    """
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


make_dirs('plots')

#rf = r.TFile.Open(args.i)

cdict = {
    'hqq': 'blue',
    'hcc': 'darkred',
    'wqq': 'lightgreen',
    'wcq': 'green',
    'qcd': 'gray',
    'tqq': 'plum',
    'zbb': 'dodgerblue',
    'zcc': 'red',
    'zqq': 'turquoise',
}

label_dict = OrderedDict({
    'Data': 'Data',
    'MC': 'MC',
    'zbb': "$\mathrm{Z(b\\bar{b})}$",
    'zcc': "$\mathrm{Z(c\\bar{c})}$",
    'zqq': "$\mathrm{Z(q\\bar{q})}$",
    'wcq': "$\mathrm{W(c\\bar{q})}$",
    'wqq': "$\mathrm{W(q\\bar{q})}$",
    'hbb': "$\mathrm{H(b\\bar{b})}$",
    'hqq': "$\mathrm{H(b\\bar{b})}$",
    'hcc': "$\mathrm{H(c\\bar{c})}$",
    'qcd': "QCD",
    'tqq': "$\mathrm{t\\bar{t}}$",
})


def full_plot(cats, pseudo=True):
    class Ugh():
        def __init__(self):
            self.plot_bins = None

    ugh = Ugh()

    def tgasym_to_err(tgasym):
        # https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/wiki/nonstandard
        # Rescale density by binwidth for actual value
        _binwidth = tgasym._fEXlow + tgasym._fEXhigh
        _x = tgasym._fX
        _y = tgasym._fY * _binwidth
        _xerrlo, _xerrhi = tgasym._fEXlow, tgasym._fEXhigh
        _yerrlo, _yerrhi = tgasym._fEYlow * _binwidth, tgasym._fEYhigh * _binwidth
        return _x, _y, _yerrlo, _yerrhi, _xerrlo, _xerrhi

    def plot_data(x, y, yerr, xerr, ax=None, pseudo=pseudo, ugh=None):
        if ugh is None:
            ugh = Ugh()
        data_err_opts = {
            'linestyle': 'none',
            'marker': '.',
            'markersize': 12.,
            'color': 'k',
            'elinewidth': 2,
        }
        if np.sum([y != 0][0]) > 0:
            if ugh.plot_bins is None:
                ugh.plot_bins = [y != 0][0]
            else:
                ugh.plot_bins = (ugh.plot_bins & [y != 0][0])
        x = np.array(x)[ugh.plot_bins]
        y = np.array(y)[ugh.plot_bins]
        yerr = [
            np.array(yerr[0])[ugh.plot_bins],
            np.array(yerr[1])[ugh.plot_bins]
        ]
        xerr = [
            np.array(xerr)[0][ugh.plot_bins],
            np.array(xerr)[1][ugh.plot_bins]
        ]
        ax.errorbar(x,
                    y,
                    yerr,
                    xerr,
                    fmt='+',
                    label="MC" if pseudo else "Data",
                    **data_err_opts)

    def th1_to_step(th1):
        _h, _bins = th1.numpy()
        _binwidths = [_bins[i + 1] - _bins[i] for i in range(len(_bins[:-1]))]
        _h *= _binwidths
        return _bins, np.r_[_h, _h[-1]]

    def plot_step(bins, h, ax=None, label=None, nozeros=True, **kwargs):
        ax.step(bins, h, where='post', label=label, c=cdict[label], **kwargs)

    avail_samples = [
        k.decode(encoding="utf-8").split(';')[0] for k in cats[0].keys()
        if b'total' not in k
    ]

    # Plotting
    fig, (ax, rax) = plt.subplots(2,
                                  1,
                                  gridspec_kw={'height_ratios': (3, 1)},
                                  sharex=True)
    plt.subplots_adjust(hspace=0)

    ## Main
    res = np.array(list(map(tgasym_to_err, [cat['data'] for cat in cats])))
    ### Sum along y, keep only first one for x
    _x, _y = res[:, 0][0], np.sum(res[:, 1], axis=0),
    _yerrlo, _yerrhi = np.sum(res[:, 2], axis=0), np.sum(res[:, 3], axis=0)
    _xerrlo, _xerrhi = res[:, 4][0], res[:, 5][0]
    plot_data(_x, _y, [_yerrlo, _yerrhi], [_xerrlo, _xerrhi], ax=ax, ugh=ugh)

    ### Stack qcd/ttbar
    tot_h, bins = None, None
    for mc, zo in zip(['qcd', 'tqq'], [1, 0]):
        if mc not in avail_samples: continue
        res = np.array(list(map(th1_to_step, [cat[mc] for cat in cats])))
        bins, h = res[:, 0][0], np.sum(res[:, 1], axis=0)
        if tot_h is None:
            plot_step(bins, h, ax=ax, label=mc, zorder=zo)
            tot_h = h
        else:
            plot_step(bins, h + tot_h, label=mc, ax=ax, zorder=zo)
            tot_h += h

    ### Stack plots
    tot_h, bins = None, None
    for mc in ['hqq', 'hcc', 'zbb', 'zcc', 'zqq', 'wcq', 'wqq']:
        if mc not in avail_samples: continue
        res = np.array(list(map(th1_to_step, [cat[mc] for cat in cats])))
        bins, h = res[:, 0][0], np.sum(res[:, 1], axis=0)
        if tot_h is None:
            plot_step(bins, h, ax=ax, label=mc)
            tot_h = h
        else:
            plot_step(bins, h + tot_h, label=mc, ax=ax)
            tot_h += h

    #######
    ## Ratio plot
    rax.axhline(0, c='gray', ls='--')

    ### Caculate diff
    res = np.array(list(map(tgasym_to_err, [cat['data'] for cat in cats])))
    _x, _y = res[:, 0][0], np.sum(res[:, 1], axis=0),
    _yerrlo, _yerrhi = np.sum(res[:, 2], axis=0), np.sum(res[:, 3], axis=0)
    _xerrlo, _xerrhi = res[:, 4][0], res[:, 5][0]
    #### Subtract MC
    y = np.copy(_y)
    for mc in ['qcd', 'tqq']:
        if mc not in avail_samples: continue
        res = np.array(list(map(th1_to_step, [cat[mc] for cat in cats])))
        bins, h = res[:, 0][0], np.sum(res[:, 1], axis=0)
        y -= h[:-1]  # Last step duplicate is not needed
    #### Scale by data uncertainty
    y /= (_yerrlo + _yerrhi)
    plot_data(_x,
              y, [_yerrlo / _yerrlo, _yerrhi / _yerrhi], [_xerrlo, _xerrhi],
              ax=rax,
              ugh=ugh)
    _scale_for_mc = np.r_[(_yerrlo + _yerrhi), (_yerrlo + _yerrhi)[-1]]

    ### Stack plots
    tot_h, bins = None, None
    for mc in ['hqq', 'hcc', 'zbb', 'zcc', 'zqq', 'wcq', 'wqq']:
        if mc not in avail_samples: continue
        res = np.array(list(map(th1_to_step, [cat[mc] for cat in cats])))
        bins, h = res[:, 0][0], np.sum(res[:, 1], axis=0)
        if tot_h is None:
            plot_step(bins, h / _scale_for_mc, ax=rax, label=mc)
            tot_h = h
        else:
            plot_step(bins, (h + tot_h) / _scale_for_mc, label=mc, ax=rax)
            tot_h += h

    ############
    # Style
    ax = hep.cms.cmslabel(ax, data=(not pseudo))
    ax.legend(ncol=2)

    ax.set_ylabel('Events / 7GeV', ha='right', y=1)
    rax.set_xlabel('jet $\mathrm{m_{SD}}$ [GeV]', ha='right', x=1)
    rax.set_ylabel(
        r'$\mathrm{\frac{Data-(MultiJet+t\bar{t})}{\sigma_{Data}}}$')

    ax.set_xlim(40, 200)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.4)
    #ax.ticklabel_format(axis='y', style='sci', scilimits=(0,3), useOffset=False)
    #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))
    f = mtick.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.10e' % x))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(g))
    rax.set_ylim(rax.get_ylim()[0] * 1.3, rax.get_ylim()[1] * 1.3)

    ipt = int(str(cats[0].name, 'utf-8').split('ptbin')[1][0]) if b'ptbin' in cats[0].name else 0
    if len(cats) == 1 or b'muon' not in cats[0].name:
        pt_range = str(pbins[ipt]) + "$< \mathrm{p_T} <$" + str(
            pbins[ipt + 1]) + " GeV"
    else:
        pt_range = str(pbins[0]) + "$< \mathrm{p_T} <$" + str(
            pbins[-1]) + " GeV"

    annot = pt_range \
            +'\nDeepDoubleX{}'.format(", MuonCR" if b'muon' in cats[0].name else "") \
            +'\n{} Region'.format("Passing" if "pass" in str(cats[0].name, 'utf-8') else "Failing")

    ax.annotate(annot,
                linespacing=1.7,
                xy=(0.04, 0.94),
                xycoords='axes fraction',
                ha='left',
                va='top',
                ma='center',
                fontsize='small',
                bbox={
                    'facecolor': 'white',
                    'edgecolor': 'white',
                    'alpha': 0,
                    'pad': 13
                },
                annotation_clip=False)

    # Leg sort
    ax.legend(*hep.plot.sort_legend(ax, label_dict), ncol=2, columnspacing=0.8)

    if b'muon' in cats[0].name:
        _iptname = "MuonCR"
    else:
        _iptname = str(str(ipt) if len(cats) == 1 else "")
    name = str("pass" if "pass" in str(cats[0].name, 'utf-8') else "fail"
               ) + _iptname

    fig.savefig('plots/{}.png'.format(args.namespace + "_" + name),
                #transparent=True,
                bbox_inches="tight")


shape_type = 'shapes_' + args.namespace

f = uproot.open(args.input_file)
pbins = [450, 500, 550, 600, 675, 800, 1200]
axees = []
for region in ['pass', 'fail']:
    print("Plotting {} region".format(region))
    for i in range(0, 6):
        cat_name = 'ptbin{}{};1'.format(i, region)
        try:
            cat = f[shape_type][cat_name]
        except:
            raise ValueError(
                "Namespace {} is not available, only following namespaces were found in the file: {}"
                .format(args.namespace, f.keys()))
        full_plot([cat])
    full_plot([f[shape_type]['ptbin{}{};1'.format(i, region)] for i in range(0, 6)])
    # MuonCR if included
    try:
        cat = f[shape_type]['muonCR{};1'.format(region)]
        full_plot([cat])
        print("Plotted muCR")
    except:
        pass
