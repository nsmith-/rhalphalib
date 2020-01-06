import argparse
from collections import OrderedDict

import uproot

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from utils import make_dirs

import mplhep as hep
plt.style.use([hep.cms.style.ROOT, {'font.size': 24}])
plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input-file",
                    default='hxxModel/shapes.root',
                    help="Input shapes file")
parser.add_argument("--fit",
                    default=None,
                    choices={"prefit", "postfit"},
                    dest='fit',
                    help="Shapes to plot")
parser.add_argument("--3reg",
                    action='store_true',
                    dest='three_regions',
                    help="By default plots pass/fail region. Set to plot pqq/pcc/pbb")
parser.add_argument("-o", "--output-folder",
                    default='plots',
                    dest='output_folder',
                    help="Folder to store plots - will be created if it doesn't exist.")

pseudo = parser.add_mutually_exclusive_group(required=True)
pseudo.add_argument('--data', action='store_false', dest='pseudo')
pseudo.add_argument('--MC', action='store_true', dest='pseudo')

args = parser.parse_args()

make_dirs(args.output_folder)

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

sdict = {
    'hqq': '-',
    'hcc': '-',
    'wqq': '-',
    'wcq': '-',
    'qcd': '-',
    'tqq': '-',
    'zbb': '-',
    'zcc': '-',
    'zqq': '-',
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


def full_plot(cats, pseudo=True, fittype=""):

    # Determine:
    if "pass" in str(cats[0].name) or "fail" in str(cats[0].name):
        regs = "pf"
    elif "pqq" in str(cats[0].name) or "pcc" in str(cats[0].name) or "pbb" in str(
            cats[0].name):
        regs = "3reg"
    else:
        print("Unknown regions")
        return

    # For masking 0 bins (don't want to show them)
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
        return _bins, np.r_[_h, _h[-1]]

    def th1_to_err(th1):
        _h, _bins = th1.numpy()
        _x = _bins[:-1] + np.diff(_bins)/2
        _xerr = [abs(_bins[:-1] - _x), _bins[1:] - _x]
        _var = th1.variances

        return _x, _h, _var, [_xerr[0], _xerr[1]]

    def plot_step(bins, h, ax=None, label=None, nozeros=True, **kwargs):
        ax.step(bins, h, where='post', label=label, c=cdict[label], **kwargs)

    # Sample proofing
    by_cat_samples = []
    for _cat in cats:
        cat_samples = [
            k.decode(encoding="utf-8").split(';')[0] for k in _cat.keys()
            if b'total' not in k
        ]
        by_cat_samples.append(cat_samples)

    from collections import Counter
    count = Counter(sum(by_cat_samples, []))
    k, v = list(count.keys()), list(count.values())
    for _sample in np.array(k)[np.array(v) != max(v)]:
        print("Sample {} is partially or entirely missing and won't be plotted".format(
            _sample))

    avail_samples = list(np.array(k)[np.array(v) == max(v)])

    # Plotting
    fig, (ax, rax) = plt.subplots(2, 1,
                                  gridspec_kw={'height_ratios': (3, 1)},
                                  sharex=True)
    plt.subplots_adjust(hspace=0)

    #  Main
    res = np.array(list(map(th1_to_err, [cat['data_obs'] for cat in cats])))
    _x, _h = res[:, 0][0], np.sum(res[:, 1], axis=0)
    _xerr = res[:, -1][0]
    _yerr = np.sqrt(np.sum(res[:, 2], axis=0))
    plot_data(_x, _h, yerr=[_yerr, _yerr], xerr=_xerr, ax=ax, ugh=ugh)

    # Stack qcd/ttbar
    tot_h, bins = None, None
    for mc, zo in zip(['qcd', 'tqq'], [1, 0]):
        if mc not in avail_samples:
            continue
        res = np.array(list(map(th1_to_step, [cat[mc] for cat in cats])))
        bins, h = res[:, 0][0], np.sum(res[:, 1], axis=0)
        if tot_h is None:
            plot_step(bins, h, ax=ax, label=mc, zorder=zo)
            tot_h = h
        else:
            plot_step(bins, h + tot_h, label=mc, ax=ax, zorder=zo)
            tot_h += h

    # Stack plots
    tot_h, bins = None, None
    for mc in ['hqq', 'hcc', 'zbb', 'zcc', 'zqq', 'wcq', 'wqq']:
        if mc not in avail_samples:
            continue
        res = np.array(list(map(th1_to_step, [cat[mc] for cat in cats])))
        bins, h = res[:, 0][0], np.sum(res[:, 1], axis=0)
        if tot_h is None:
            plot_step(bins, h, ax=ax, label=mc)
            tot_h = h
        else:
            plot_step(bins, h + tot_h, label=mc, ax=ax)
            tot_h += h

    #######
    # Ratio plot
    rax.axhline(0, c='gray', ls='--')

    # Caculate diff
    res = np.array(list(map(th1_to_err, [cat['data_obs'] for cat in cats])))
    _x, _y = res[:, 0][0], np.sum(res[:, 1], axis=0)
    _xerr = res[:, -1][0]
    _yerr = np.sqrt(np.sum(res[:, 2], axis=0))

    y = np.copy(_y)
    for mc in ['qcd', 'tqq']:
        if mc not in avail_samples:
            continue
        res = np.array(list(map(th1_to_step, [cat[mc] for cat in cats])))
        bins, h = res[:, 0][0], np.sum(res[:, 1], axis=0)
        y -= h[:-1]

    y /= _yerr
    _scale_for_mc = np.r_[_yerr,  _yerr[-1]]

    def prop_err(A, B, C, a, b, c):
        # Error propagation for (Data - Bkg)/Sigma_{Data} plot
        e = C**2 * (a**2 + b**2) + c**2 * (A - B)**2
        e /= C**4
        e = np.sqrt(e)
        return e

    # Error propagation, not sensitive to args[-1]
    err = prop_err(_y, _y-y, np.sqrt(_y), np.sqrt(_y), np.sqrt(_y-y), 1)

    plot_data(_x, y, yerr=[err, err], xerr=_xerr, ax=rax, ugh=ugh)

    # Stack plots
    tot_h, bins = None, None
    for mc in ['hqq', 'hcc', 'zbb', 'zcc', 'zqq', 'wcq', 'wqq']:
        if mc not in avail_samples:
            continue
        res = np.array(list(map(th1_to_step, [cat[mc] for cat in cats])))
        bins, h = res[:, 0][0], np.sum(res[:, 1], axis=0)
        if tot_h is None:
            plot_step(bins, h / _scale_for_mc, ax=rax, label=mc)
            tot_h = h
        else:
            plot_step(bins, (h + tot_h)/_scale_for_mc, label=mc, ax=rax)
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
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(0,3), useOffset=False)
    # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))
    f = mtick.ScalarFormatter(useOffset=False, useMathText=True)
    # g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.10e' % x))

    def g(x, pos):
        return "${}$".format(f._formatSciNotation('%1.10e' % x))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(g))
    rax.set_ylim(rax.get_ylim()[0] * 1.3, rax.get_ylim()[1] * 1.3)

    ipt = int(str(cats[0].name
                  ).split('ptbin')[1][0]) if b'ptbin' in cats[0].name else 0
    if len(cats) == 1:
        pt_range = str(pbins[ipt]) + "$< \mathrm{p_T} <$" + str(
            pbins[ipt + 1]) + " GeV"
    else:
        pt_range = str(pbins[0]) + "$< \mathrm{p_T} <$" + str(
            pbins[-1]) + " GeV"
    if b'muon' in cats[0].name:
        pt_range = str(pbins[0]) + "$< \mathrm{p_T} <$" + str(
            pbins[-1]) + " GeV"

    lab_mu = ", MuonCR" if b'muon' in cats[0].name else ""
    if regs == "pf":
        lab_reg = "Passing" if "pass" in str(cats[0].name) else "Failing"
    else:
        if "pqq" in str(cats[0].name):
            lab_reg = "Light"
        elif "pcc" in str(cats[0].name):
            lab_reg = "Charm"
        elif "pbb" in str(cats[0].name):
            lab_reg = "Bottom"
    
    annot = pt_range + '\nDeepDoubleX{}'.format(lab_mu) + '\n{} Region'.format(lab_reg)

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
    leg = ax.legend(*hep.plot.sort_legend(ax, label_dict), ncol=2, columnspacing=0.8)
    leg.set_title(title=fittype.capitalize(),prop={'size':"smaller"})

    if b'muon' in cats[0].name:
        _iptname = "MuonCR"
    else:
        _iptname = str(str(ipt) if len(cats) == 1 else "")
    # name = str("pass" if "pass" in str(cats[0].name) else "fail"
    #            ) + _iptname
    name = str(lab_reg) + _iptname

    fig.savefig('{}/{}.png'.format(args.output_folder, fittype + "_" + name),
                bbox_inches="tight")


if args.fit is None:
    shape_types = ['prefit', 'postfit']
else:
    shape_types = [args.fit]
if args.three_regions:
    regions = ['pqq', 'pcc', 'pbb']
else:
    regions = ['pass', 'fail']

f = uproot.open(args.input_file)
for shape_type in shape_types:
    pbins = [450, 500, 550, 600, 675, 800, 1200]
    for region in regions:
        print("Plotting {} region".format(region))
        for i in range(0, 6):
            continue
            cat_name = 'ptbin{}{}_{};1'.format(i, region, shape_type)
            try:
                cat = f[cat_name]
            except Exception:
                raise ValueError("Namespace {} is not available, only following"
                                "namespaces were found in the file: {}".format(
                                    args.fit, f.keys()))

            fig = full_plot([cat], pseudo=args.pseudo, fittype=shape_type)
        full_plot([f['ptbin{}{}_{};1'.format(i, region, shape_type)] for i in range(0, 6)],
                   pseudo=args.pseudo, fittype=shape_type)
        # MuonCR if included
        try:
            cat = f['muonCR{}_{};1'.format(region, shape_type)]
            full_plot([cat], args.pseudo, fittype=shape_type)
            print("Plotted muCR")
        except Exception:
            pass
