import ROOT as r
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

from matplotlib.colors import to_rgba
from collections import OrderedDict

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


def calphas(boolar, c='#455D8C'):
    c = list(to_rgba(c))
    cs = []
    for b in boolar:
        _c = c
        if b:
            _c[-1] = 1
        else:
            _c[-1] = .3
        cs.append(tuple(_c))
    return cs


def hatches(boolar):
    hs = []
    for b in boolar:
        if b:
            hs.append('')
        else:
            hs.append('.')
    return hs
    

def plot_fractions(fitDiagnostics_file='fitDiagnostics.root',
                   model_file='model_combined.root',
                   out='fractions.png',
                   data=False, year=2017,):
    rf = r.TFile.Open(fitDiagnostics_file)
    rm = r.TFile.Open(model_file)

    # Get veff names
    # par_names = rf.Get('fit_s').floatParsFinal().contentsString().split(',')
    # par_names = [p for p in par_names if 'veff' in p]
    par_names = [
        'veff_wcq_pbb', 'veff_wcq_pcc', 'veff_wqq_pbb', 'veff_wqq_pcc', 'veff_zbb_pbb',
        'veff_zbb_pcc', 'veff_zcc_pbb', 'veff_zcc_pcc', 'veff_zqq_pbb', 'veff_zqq_pcc'
    ]

    # Get initial values
    nom_dict = {}
    for pn in par_names:
        try:
            nom_dict[pn] = rm.Get('w').allVars().find(pn).getValV()
        except Exception:
            print("Missing {}".format(pn))

    # Get postfit values
    res_dict = {}
    unc_dict = {}
    for pn in par_names:
        try:
            _v = round(rf.Get('fit_s').floatParsFinal().find(pn).getVal(), 4)
            _u = round(rf.Get('fit_s').floatParsFinal().find(pn).getError(), 4)
            res_dict[pn] = _v
            unc_dict[pn] = _u
        except:
            continue


    # Plotting
    plt.style.use(hep.style.ROOT)
    samples = ["zbb", "zcc", "zqq", "wcq", "wqq"]

    def get_dval(vname, nominal=nom_dict, actual=res_dict):
        try:
            return actual[vname]
        except:
            return nominal[vname]

    def tobars(dic):
        pbb, pcc, pqq = [], [], []
        for samp in samples:
            ec = get_dval("veff_{}_pcc".format(samp), actual=dic)
            eb = get_dval("veff_{}_pbb".format(samp), actual=dic) * (1-ec)
            pbb.append(eb)
            pcc.append(ec)
            pqq.append(1-eb-ec)
        return pbb, pcc, pqq

    def wasfitted(dic=res_dict):
        yaynayb = []
        yaynayc = []
        yaynayq = []
        for samp in samples:
            if "veff_{}_pcc".format(samp) in dic.keys():
                yaynayc.append(True)
            else:
                yaynayc.append(False)
            if "veff_{}_pbb".format(samp) in dic.keys():
                yaynayb.append(True)
            else:
                yaynayb.append(False)
            if "veff_{}_pbb".format(samp) in dic.keys() or "veff_{}_pcc".format(samp) in dic.keys():
                yaynayq.append(True)
            else:
                yaynayq.append(False)

        return np.array(yaynayb), np.array(yaynayc), np.array(yaynayq)

    c = ['#455D8C', '#F23C05', '#6AAEB9']
    width = 0.35
    gap = 0.03
    _hatch = "//"
    nvec = len(samples)

    fig, ax = plt.subplots(figsize=(12.5, 10))
    pbb, pcc, pqq = tobars(res_dict)
    ynpbb, ynpcc, ynpqq = wasfitted(res_dict)

    ind = np.arange(nvec)-gap-width
    p1 = ax.barh(ind, pbb, label="Pbb", align='edge', height=width, color=calphas(ynpbb, c[0]))
    for bar, pattern in zip(p1, hatches(ynpbb)):
        bar.set_hatch(pattern)
    p1 = ax.barh(ind, np.array(pcc), left=np.array(pbb), label="Pcc", align='edge', height=width, color=calphas(ynpcc, c[1]))
    for bar, pattern in zip(p1, hatches(ynpcc)):
        bar.set_hatch(pattern)
    p1 = ax.barh(ind, np.array(pqq), left=np.array(pbb)+np.array(pcc), label="Pqq", align='edge', height=width, color=calphas(ynpqq, c[2]))
    for bar, pattern in zip(p1, hatches(ynpqq)):
        bar.set_hatch(pattern)



    pbb, pcc, pqq = tobars(nom_dict)
    ind = np.arange(nvec)+gap
    p1 = ax.barh(ind, pbb, align='edge',label="Pbb",  height=width, color=c[0], hatch=_hatch, alpha=0.7)
    p1 = ax.barh(ind, np.array(pcc), left=np.array(pbb), align='edge',label="Pcc",  height=width, color=c[1], hatch=_hatch, alpha=0.7)
    p1 = ax.barh(ind, np.array(pqq), left=np.array(pbb)+np.array(pcc), align='edge', label="Pqq",  height=width, color=c[2], hatch=_hatch, alpha=0.7)

    plt.legend(title='Fitted')

    lines = plt.gca().get_legend().legendHandles
    for i in [0,1,2]:
        _col = lines[i].get_facecolor()[:-1]
        _col += (1,)
        lines[i].set_facecolor(_col)
    legend1 = plt.legend([lines[i] for i in [0,1,2]], ["Pbb", "Pcc", "Pqq"], loc=1, title="Fitted",
                        bbox_to_anchor=(1,0.7))
    for i in [0,1,2]:
        _col = lines[i].get_facecolor()[:-1]
        _col += (.3,)
        lines[i].set_facecolor(_col)
        lines[i].set_hatch("..")
    legend2 = plt.legend([lines[i] for i in [0,1,2]], ["Pbb", "Pcc", "Pqq"], loc=1, title="Frozen",
                        bbox_to_anchor=(1,0.4))
    for i in [3,4,5]:
        _col = lines[i].get_facecolor()[:-1]
        _col += (.7,)
        lines[i].set_facecolor(_col)
        lines[i].set_hatch("//")
    legend3 = plt.legend([lines[i] for i in [3,4,5]], ["Pbb", "Pcc", "Pqq"], loc=1, title="Prefit",
                        bbox_to_anchor=(1,1))
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    plt.gca().add_artist(legend3)

    plt.yticks(range(nvec), [label_dict[l] for l in samples])
    plt.vlines(1, -1, 5, lw=2, linestyle='--')
    plt.xlim(0, 1.23)
    plt.ylim(-.7, nvec - 1 + .7)
    plt.xlabel("Vector Fractions", x=1, ha='right')
    ax.tick_params(axis='y', which='minor', left=False)

    hep.cms.label(data=data, year=year)

    fig.savefig(out)
