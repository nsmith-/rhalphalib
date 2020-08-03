import argparse
import os
from operator import methodcaller

import uproot
import ROOT as r

import matplotlib.pyplot as plt
import numpy as np

import json
import mplhep as hep
from utils import make_dirs, get_fixed_mins_maxs, pad2d

plt.switch_backend('agg')

from _plot_TF import plotTF as plotTFsmooth
from _plot_TF import TF_smooth_plot, TF_params, TF, plotTF_ratio, plot_qcd


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

    configs = json.load(open("config.json"))

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

    parmap = np.array(hmp).reshape(ptdeg+1, rhodeg+1)

    if configs['fitTF']:
        degs = tuple([int(s) for s in configs['degs'].split(',')])
    if configs['MCTF']:
        degsMC = tuple([int(s) for s in configs['degsMC'].split(',')])

    if len(MCTF) > 0:
        MCTF_map = np.array(MCTF).reshape(degsMC[0]+1, degsMC[1]+1)

    ##### Smooth plots
    _values = hmp
    # TF Data
    if configs['fitTF']:
        print("Plot TF - data residual")
        plotTFsmooth(*TF_smooth_plot(*TF_params(_values, nrho=rhodeg, npt=ptdeg)), MC=False, raw=args.isMC,
                     rhodeg=rhodeg, ptdeg=ptdeg,
                     out='{}/TF_data'.format(args.output_folder), year=args.year)

    # TF MC Postfit
    if configs['MCTF']:
        print("Plot TF - MC post-fit")
        _vect = np.load('decoVector.npy')
        _MCTF_nominal = np.load('MCTF.npy')
        _values = _values = _vect.dot(np.array(MCTF)) + _MCTF_nominal
        plotTFsmooth(*TF_smooth_plot(*TF_params(_values, npt=degsMC[0], nrho=degsMC[1])), MC=True, raw=args.isMC,
                    ptdeg=degsMC[0], rhodeg=degsMC[1],
                    out='{}/TF_MC'.format(args.output_folder), year=args.year)
    
    # Effective TF (combination)
    if configs['fitTF'] and configs['MCTF']:
        print("Plot TF - effective combination")
        _tf1, _, _, _ = TF_smooth_plot(*TF_params(hmp, npt=degs[0], nrho=degs[1]))
        _tf2, bit1, bit2, bit3 = TF_smooth_plot(*TF_params(_values, npt=degsMC[0], nrho=degsMC[1]))
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
        return TF(pt, rho, n_pT=degs[0], n_rho=degs[1], par_map=parmap)

    TFres = np.array(list(map(TFwrap, ptscaled.flatten(),
                          rhoscaled.flatten()))).reshape(ptpts.shape)

    if len(MCTF) > 0:
        def TFwrap(pt, rho):
            return TF(pt, rho, n_pT=degsMC[0], n_rho =degsMC[1], par_map=MCTF_map)

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

    plotTF_ratio(in_data_rat, mask, region="Prefit", args=args)

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

    plotTF_ratio(in_data_rat, mask, region="Postfit", args=args)

    # Plot QCD shape
    #plot_qcd(pass_qcd, fail=False)
    #plot_qcd(fail_qcd, fail=True)
