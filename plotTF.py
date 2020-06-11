import argparse
import os
from operator import methodcaller

import uproot
import ROOT as r

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

import mplhep as hep
from utils import make_dirs, get_fixed_mins_maxs, pad2d

plt.switch_backend('agg')


# Benstein polynomial calculation
def bern_elem(x, v, n):
    # Bernstein element calculation
    normalization = 1. * math.factorial(n) / (math.factorial(v) * math.factorial(n - v))
    Bvn = normalization * (x**v) * (1-x)**(n-v)
    return float(Bvn)


def TF(pT, rho, n_pT=2, n_rho=2, par_map=np.ones((3, 3))):
    # Calculate TF Polynomial for (n_pT, n_rho) degree Bernstein poly
    val = 0
    for i_pT in range(0, n_pT+1):
        for i_rho in range(0, n_rho+1):
            val += (bern_elem(pT, i_pT, n_pT)
                    * bern_elem(rho, i_rho, n_rho)
                    * par_map[i_pT][i_rho])

    return val


# TF Plots
def plotTF(TF, msd, pt, mask=None, MC=False, raw=False, rhodeg=2, ptdeg=2, out=None):
    """
    Parameters:
    TF: Transfer Factor array
    msd: Mass bins array (meshgrid-like)
    pt: pT bins array (meshgrid-like)
    """
    fig, ax = plt.subplots()
    #if mask is not None:
    #    TF = np.ma.array(TF, mask=~mask)

    zmin, zmax = np.floor(10*np.min(TF))/10, np.ceil(10*np.max(TF))/10
    zmin, zmax = zmin + 0.001, zmax - 0.001
    clim = np.max([.3, np.min([abs(zmin - 1), abs(zmax - 1)])])
    levels = np.linspace(1-clim, 1+clim, 500)

    if mask is not None:
        contf = ax.contourf(msd, pt, TF, levels=levels,
                            corner_mask=False, cmap='RdBu_r')
    else:
        contf = ax.contourf(msd, pt, TF, levels=levels, cmap='RdBu_r')
    cax = hep.make_square_add_cbar(ax, pad=0.2, size=0.5)
    if abs(1-zmin) > .3 and abs(1-zmax) > .3:
        c_extend = 'both'
    elif abs(1-zmin) > .3:
        c_extend = 'min'
    elif abs(1-zmax) > .3:
        c_extend = 'max'
    else:
        c_extend = 'neither'
    cbar = fig.colorbar(contf, cax=cax, extend=c_extend)
    cbar.set_ticks([np.arange(1-clim, 1+clim, 0.1)])

    def rho_bound(ms, rho):
        # rho = {-6, -2.1}
        fpt = ms * np.e**(-rho/2)
        return fpt

    x = np.arange(40, 70)
    ax.plot(x, rho_bound(x, -6), 'black', lw=3)
    ax.fill_between(x, rho_bound(x, -6), 1200, facecolor="none", hatch="x",
                    edgecolor="black", linewidth=0.0)
    x = np.arange(150, 201)
    ax.plot(x, rho_bound(x, -2.1) + 5, 'black', lw=3)
    ax.fill_between(x, rho_bound(x, -2.1), facecolor="none", hatch="x",
                    edgecolor="black", linewidth=0.0)

    ax.set_xlim(40,201)
    ax.set_ylim(450,1200)
    ax.invert_yaxis()

    tMC = "MC only" if MC else "Data Residual"
    if raw: tMC = "Prefit MC"
    ax.set_title('{} Transfer Factor ({},{})'.format(tMC, rhodeg, ptdeg),
                 pad=15,
                 fontsize=26)
    ax.set_xlabel(r'Jet $\mathrm{m_{SD}}$', ha='right', x=1)
    ax.set_ylabel(r'Jet $\mathrm{p_{T}}$', ha='right', y=1)
    cbar.set_label(r'TF', ha='right', y=1)

    label = "MC" if MC else "Data"
    if raw: 
        label = "MCRaw"
    if out is None:
        fig.savefig('{}/{}{}.png'.format(args.output_folder,
                                         "TFmasked" if mask is not None else "TF",
                                         label),
                    bbox_inches="tight")
    else:
        fig.savefig('TF{}.png'.format(out, bbox_inches="tight"))
        


def plotTF_ratio(in_ratio, mask, region):
    fig, ax = plt.subplots()

    H = np.ma.masked_where(in_ratio * mask <= 0.01, in_ratio * mask)
    zmin, zmax = np.floor(10*np.min(TFres))/10, np.ceil(10*np.max(TFres))/10
    zmin, zmax = zmin + 0.001, zmax - 0.001
    clim = np.max([.3, np.min([abs(zmin - 1), abs(zmax - 1)])])
    ptbins = np.array([450, 500, 550, 600, 675, 800, 1200])
    msdbins = np.linspace(40, 201, 24)
    hep.hist2dplot(H.T, msdbins, ptbins, vmin=1-clim, vmax=1+clim,
                   cmap='RdBu_r', cbar=False)
    cax = hep.make_square_add_cbar(ax, pad=0.2, size=0.5)
    if abs(1-zmin) > .3 and abs(1-zmax) > .3:
        c_extend = 'both'
    elif abs(1-zmin) > .3:
        c_extend = 'min'
    elif abs(1-zmax) > .3:
        c_extend = 'max'
    else:
        c_extend = 'neither'
    cbar = fig.colorbar(ax.get_children()[0], cax=cax, extend=c_extend)

    ax.set_xticks(np.arange(40, 220, 20))
    ax.tick_params(axis='y', which='minor', left=False, right=False)
    ax.invert_yaxis()

    ax.set_title('{} QCD Ratio'.format(region), pad=15, fontsize=26)
    ax.set_xlabel(r'Jet $\mathrm{m_{SD}}$', ha='right', x=1)
    ax.set_ylabel(r'Jet $\mathrm{p_{T}}$', ha='right', y=1)
    cbar.set_label(r'(Pass QCD) / (Fail QCD * eff)', ha='right', y=1)

    fig.savefig('{}/{}{}.png'.format(args.output_folder, "TF_ratio_", region),
                bbox_inches="tight")


def plot_qcd(qcd, fail=False):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')

    ax.set_ylim(get_fixed_mins_maxs(450, 1200))
    ax.set_xlim(get_fixed_mins_maxs(40, 201))
    ax.set_zlim(get_fixed_mins_maxs(0, 6))

    ax.set_title(('Fail' if fail else "Pass") + 'QCD')
    ax.set_xlabel(r'Jet $\mathrm{m_{SD}}$', ha='right', x=1, labelpad=15)
    ax.set_ylabel(r'Jet $\mathrm{p_{T}}$', ha='right', y=1, labelpad=15)
    ax.set_zlabel(r'$\mathrm{log_{10}(N)}$', ha='left', labelpad=15)

    ptbins = np.array([450, 500, 550, 600, 675, 800, 1200])
    msdbins = np.linspace(40, 201, 24)
    ptpts, msdpts = np.meshgrid(ptbins[:-1] + 0.3 * np.diff(ptbins),
                                msdbins[:-1] + 0.5 * np.diff(msdbins),
                                indexing='ij')

    Xi = (msdpts-3.5).flatten()
    Yi = np.array([list(ptbins[:-1])]*23).T.flatten()
    Zi = 0
    dx = 7
    dy = np.array([list(np.diff(ptbins))]*23).T.flatten()
    dz = qcd.ravel()
    dz = np.where(dz > 2, dz, 1)

    ax.bar3d(Xi, Yi, Zi, dx, dy, np.log10(dz), color='w')
    ax.azim = ax.azim + 180
    ax.invert_xaxis()

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    fig.savefig('{}/{}.png'.format(args.output_folder,
                                   ("Fail" if fail else "Pass") + "QCD"),
                bbox_inches="tight")


if __name__ == '__main__':

    plt.style.use([hep.cms.style.ROOT, {'font.size': 24}])
    plt.switch_backend('agg')
    np.seterr(divide='ignore', invalid='ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--dir",
                        default='',
                        help="Model/Fit dir")
    parser.add_argument("-i",
                        "--input-file",
                        default='shapes.root',
                        help="Input shapes file")
    parser.add_argument("-f", "--fit",
                        default='fitDiagnostics.root',
                        dest='fit',
                        help="fitDiagnostics file")
    parser.add_argument("-o", "--output-folder",
                        default='plots',
                        dest='output_folder',
                        help="Folder to store plots - will be created ? doesn't exist.")
    parser.add_argument("--year",
                        default="2017",
                        type=str,
                        help="year label")
    parser.add_argument("--MC",
                        action='store_true',
                        dest='isMC',
                        help="Use 'simulation' label")

    args = parser.parse_args()
    if args.output_folder.split("/")[0] != args.dir:
        args.output_folder = os.path.join(args.dir, args.output_folder)
    make_dirs(args.output_folder)

    # Get fitDiagnostics File
    rf = r.TFile.Open(os.path.join(args.dir, args.fit))

    # Get TF parameters
    hmp = []
    par_names = rf.Get('fit_s').floatParsFinal().contentsString().split(',')
    par_names = [p for p in par_names if 'tf' in p]
    MCTF = []
    for pn in par_names:
        if "deco" not in pn:
            hmp.append(round(rf.Get('fit_s').floatParsFinal().find(pn).getVal(), 4))
        elif "deco" in pn:
            MCTF.append(round(rf.Get('fit_s').floatParsFinal().find(pn).getVal(), 4))

    def _get(s):
        # Helper
        return s[-1][0]

    par_names = [n for n in par_names if "deco" not in n]
    ptdeg = max(
        list(
            map(int, list(map(_get, list(map(methodcaller("split", 'pt_par'),
                                             par_names)))))))
    rhodeg = max(
        list(
            map(int, list(map(_get, list(map(methodcaller("split", 'rho_par'),
                                             par_names)))))))

    parmap = np.array(hmp).reshape(rhodeg+1, ptdeg+1)
    if len(MCTF) > 0:
        MCTF_map = np.array(MCTF).reshape(rhodeg+1, ptdeg+1)\

    ##### Smooth plots
    from plotTF2 import plotTF as plotTFsmooth
    from plotTF2 import TF_smooth_plot, TF_params
    _values = hmp
    # TF Data
    plotTFsmooth(*TF_smooth_plot(*TF_params(_values, nrho=2, npt=2)), MC=False, raw=args.isMC,
                 out='{}/TF_data'.format(args.output_folder), year=args.year)

    # TF MC Postfit
    _vect = np.load('decoVector.npy')
    _MCTF_nominal = np.load('MCTF.npy')
    _values = _values = _vect.dot(np.array(MCTF)) + _MCTF_nominal
    plotTFsmooth(*TF_smooth_plot(*TF_params(_values, nrho=2, npt=2)), MC=True, raw=args.isMC,
                 out='{}/TF_MC'.format(args.output_folder), year=args.year)

    # Effective TF (combination)
    _tf1, _, _, _ = TF_smooth_plot(*TF_params(hmp, nrho=2, npt=2))
    _tf2, bit1, bit2, bit3 = TF_smooth_plot(*TF_params(_values, nrho=2, npt=2))
    plotTFsmooth(_tf1*_tf2, bit1, bit2, bit3, MC=True, raw=args.isMC,
                 out='{}/TF_eff'.format(args.output_folder), year=args.year,
                 label='Effective Transfer Factor')

    # Define bins
    # ptbins = np.array([450, 500, 550, 600, 675, 800, 1200])
    ptbins = np.arange(450, 1205, 5)
    npt = len(ptbins) - 1
    # msdbins = np.linspace(40, 201, 24)
    msdbins = np.arange(40, 201.5, .5)

    # here we derive these all at once with 2D array
    ptpts, msdpts = np.meshgrid(ptbins[:-1] + 0.3 * np.diff(ptbins),
                                msdbins[:-1] + 0.5 * np.diff(msdbins),
                                indexing='ij')
    rhopts = 2*np.log(msdpts/ptpts)
    ptscaled = (ptpts - 450.) / (1200. - 450.)
    rhoscaled = (rhopts - (-6)) / ((-2.1) - (-6))
    validbins = (rhoscaled >= 0) & (rhoscaled <= 1)
    rhoscaled[~validbins] = 1  # we will mask these out later

    def TFwrap(pt, rho):
        return TF(pt, rho, ptdeg, rhodeg, parmap)

    TFres = np.array(list(map(TFwrap, ptscaled.flatten(),
                          rhoscaled.flatten()))).reshape(ptpts.shape)

    if len(MCTF) > 0:
        def TFwrap(pt, rho):
            return TF(pt, rho, ptdeg, rhodeg, MCTF_map)

        MCTFres = np.array(list(map(TFwrap, ptscaled.flatten(),
                                rhoscaled.flatten()))).reshape(ptpts.shape)


    # Pad mass bins
    pmsd = pad2d(msdpts)
    pmsd[:, 0] = 40
    pmsd[:, -1] = 201
    # Pad pT bins
    ppt = pad2d(ptpts)
    ppt[0, :] = 450
    ppt[-1, :] = 1200
    # Pad TF result
    pTF = pad2d(TFres)
    pvb = pad2d(validbins).astype(bool)

    # if len(MCTF) > 0:
    #     pMCTF = pad2d(MCTFres)
    #     plotTF(1+pMCTF, pmsd, ppt, mask=pvb, MC=True)

    # # Plot TF
    # plotTF(pTF, pmsd, ppt)
    # plotTF(pTF, pmsd, ppt, mask=pvb, MC=False)

    # Get Info from Shapes
    # Build 2D
    f = uproot.open(os.path.join(args.dir, args.input_file))
    region = 'prefit'
    fail_qcd, pass_qcd = [], []
    bins = []
    for ipt in range(6):
        fail_qcd.append(f['ptbin{}fail_{}/qcd'.format(ipt, region)].values)
        pass_qcd.append(f['ptbin{}pass_{}/qcd'.format(ipt, region)].values)

    fail_qcd = np.array(fail_qcd)
    pass_qcd = np.array(pass_qcd)

    mask = ~np.isclose(pass_qcd, np.zeros_like(pass_qcd))
    mask *= ~np.isclose(fail_qcd, np.zeros_like(fail_qcd))
    q = np.sum(pass_qcd[mask])/np.sum(fail_qcd[mask])
    in_data_rat = (pass_qcd/(fail_qcd * q))

    plotTF_ratio(in_data_rat, mask, region="Prefit")

    region = 'postfit'
    fail_qcd, pass_qcd = [], []
    bins = []
    for ipt in range(6):
        fail_qcd.append(f['ptbin{}fail_{}/qcd'.format(ipt, region)].values)
        pass_qcd.append(f['ptbin{}pass_{}/qcd'.format(ipt, region)].values)

    fail_qcd = np.array(fail_qcd)
    pass_qcd = np.array(pass_qcd)

    mask = ~np.isclose(pass_qcd, np.zeros_like(pass_qcd))
    mask *= ~np.isclose(fail_qcd, np.zeros_like(fail_qcd))
    q = np.sum(pass_qcd[mask])/np.sum(fail_qcd[mask])
    in_data_rat = (pass_qcd/(fail_qcd * q))

    plotTF_ratio(in_data_rat, mask, region="Postfit")

    # Plot QCD shape
    plot_qcd(pass_qcd, fail=False)
    plot_qcd(fail_qcd, fail=True)
