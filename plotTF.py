import argparse
from operator import methodcaller

import uproot
import ROOT as r

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

import mplhep as hep
from utils import make_dirs, get_fixed_mins_maxs, pad2d


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
def plotTF(TF, msd, pt, mask=None):
    """
    Parameters: 
    TF: Transfer Factor array
    msd: Mass bins array (meshgrid-like)
    pt: pT bins array (meshgrid-like)
    """
    fig, ax = plt.subplots()
    if mask is not None:
        TF = np.ma.array(TF, mask=~pvb)

    zmin, zmax = np.floor(10*np.min(TF))/10, np.ceil(10*np.max(TF))/10
    levels = np.linspace(zmin, zmax, 500)

    if mask is not None:
        contf = ax.contourf(msd, pt, TF, levels=levels,
                            corner_mask=False, cmap='RdBu_r')
    else:
        contf = ax.contourf(msd, pt, TF, levels=levels, cmap='RdBu_r')
    cax = hep.make_square_add_cbar(ax, pad=0.2, size=0.5)
    cbar = fig.colorbar(contf, cax=cax)
    cbar.set_ticks([np.arange(zmin, zmax, 0.1)])

    ax.invert_yaxis()

    ax.set_title('Transfer Factor ({},{})'.format(rhodeg, ptdeg), pad=15, fontsize=26)
    ax.set_xlabel(r'Jet $\mathrm{m_{SD}}$', ha='right', x=1)
    ax.set_ylabel(r'Jet $\mathrm{p_{T}}$', ha='right', y=1)
    cbar.set_label(r'TF', ha='right', y=1)

    fig.savefig('{}/{}.png'.format(args.output_folder,
                                   "TFmasked" if mask is not None else "TF"),
                bbox_inches="tight")


def plotTF_ratio(in_ratio, mask):
    fig, ax = plt.subplots()

    H = np.ma.masked_where(in_ratio * mask <= 0.01, in_ratio * mask)
    zmin, zmax = np.floor(10*np.min(TFres))/10, np.ceil(10*np.max(TFres))/10
    hep.hist2dplot(H, msdbins, ptbins, vmin=zmin, vmax=zmax, cmap='RdBu_r', cbar=False)
    cax = hep.make_square_add_cbar(ax, pad=0.2, size=0.5)
    cbar = fig.colorbar(ax.get_children()[0], cax=cax)

    ax.set_xticks(np.arange(40, 220, 20))
    ax.tick_params(axis='y', which='minor', left=False, right=False)
    ax.invert_yaxis()

    ax.set_title('Postfit QCD Ratio', pad=15, fontsize=26)
    ax.set_xlabel(r'Jet $\mathrm{m_{SD}}$', ha='right', x=1)
    ax.set_ylabel(r'Jet $\mathrm{p_{T}}$', ha='right', y=1)
    cbar.set_label(r'(Pass QCD) / (Fail QCD * eff)', ha='right', y=1)

    fig.savefig('{}/{}.png'.format(args.output_folder, "TF_ratio"), bbox_inches="tight")


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
    parser.add_argument("-i",
                        "--input-file",
                        default='tempModel/shapes.root',
                        help="Input shapes file")
    parser.add_argument("-f", "--fit",
                        default='tempModel/fitDiagnostics.root',
                        dest='fit',
                        help="fitDiagnostics file")

    parser.add_argument("-o", "--output-folder",
                        default='plots',
                        dest='output_folder',
                        help="Folder to store plots - will be created ? doesn't exist.")

    args = parser.parse_args()

    make_dirs(args.output_folder)

    # Get fitDiagnostics File
    rf = r.TFile.Open(args.fit)

    # Get TF parameters
    hmp = []
    par_names = rf.Get('fit_s').floatParsFinal().contentsString().split(',')
    par_names = [p for p in par_names if 'tf' in p]
    for pn in par_names:
        hmp.append(round(rf.Get('fit_s').floatParsFinal().find(pn).getVal(), 4))

    def _get(s):
        # Helper
        return s[-1][0]

    ptdeg = max(
        list(
            map(int, list(map(_get, list(map(methodcaller("split", 'pt_par'),
                                             par_names)))))))
    rhodeg = max(
        list(
            map(int, list(map(_get, list(map(methodcaller("split", 'rho_par'),
                                             par_names)))))))

    parmap = np.array(hmp).reshape(rhodeg+1, ptdeg+1)

    # Define bins
    ptbins = np.array([450, 500, 550, 600, 675, 800, 1200])
    npt = len(ptbins) - 1
    msdbins = np.linspace(40, 201, 24)

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

    # Plot TF
    plotTF(pTF, pmsd, ppt)
    plotTF(pTF, pmsd, ppt, mask=pvb)

    # Get Info from Shapes
    # Build 2D
    f = uproot.open(args.input_file)
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

    plotTF_ratio(in_data_rat, validbins)

    # Plot QCD shape
    plot_qcd(pass_qcd, fail=False)
    plot_qcd(fail_qcd, fail=True)
