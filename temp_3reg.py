from __future__ import print_function, division
import rhalphalib as rl
import numpy as np
import pickle
import uproot
rl.util.install_roofit_helpers()


def get_templ2(f, region, sample, ptbin, syst=None):
    if sample in ["hcc", "hqq"]:
        sample += "125"
    hist_name = '{}_{}'.format(sample, region)
    if syst is not None:
        hist_name += "_" + syst
    hist_name += "_bin" + str(ptbin)
    try:
        h_vals = f[hist_name].values
        h_edges = f[hist_name].edges
    except Exception:
        print("Warning: template {} was not found, replaces with [0, 0, ...0]".format(
            hist_name))
        h_vals = np.zeros_like(f[hist_name.replace('bin5', 'bin4')].values)
        h_edges = f[hist_name.replace('bin5', 'bin4')].edges
    h_key = 'msd'
    return (h_vals, h_edges, h_key)


def get_templ2M(f, region, sample, ptbin, syst=None, read_sumw2=False):
    # With Matched logic
    if sample in ["hcc", "hqq"]:
        sample += "125"
    hist_name = '{}_{}'.format(sample, region)
    if syst is not None:
        hist_name += "_" + syst
    if (sample.startswith("w") or sample.startswith("z")
            or sample.startswith("h")) and syst is None:
        hist_name += "_" + 'matchedUp'
    hist_name += "_bin" + str(ptbin)
    h_vals = f[hist_name].values
    h_edges = f[hist_name].edges
    h_key = 'msd'
    if read_sumw2:
        h_variances = f[hist_name].variances[:, ptbin]
        return (h_vals, h_edges, h_key, h_variances)
    return (h_vals, h_edges, h_key)


def dummy_rhalphabet(pseudo, throwPoisson, MCTF,  use_matched):
    fitTF = True

    # Default lumi (needs at least one systematics for prefit)
    sys_lumi = rl.NuisanceParameter('CMS_lumi', 'lnN')

    # Define Bins
    ptbins = np.array([450, 500, 550, 600, 675, 800, 1200])
    npt = len(ptbins) - 1
    msdbins = np.linspace(40, 201, 24)
    msd = rl.Observable('msd', msdbins)

    # Define pt/msd/rho grids
    ptpts, msdpts = np.meshgrid(ptbins[:-1] + 0.3 * np.diff(ptbins),
                                msdbins[:-1] + 0.5 * np.diff(msdbins),
                                indexing='ij')
    rhopts = 2*np.log(msdpts/ptpts)
    ptscaled = (ptpts - 450.) / (1200. - 450.)
    rhoscaled = (rhopts - (-6)) / ((-2.1) - (-6))
    validbins = (rhoscaled >= 0) & (rhoscaled <= 1)
    rhoscaled[~validbins] = 1  # we will mask these out later

    # Template reading
    # f = uproot.open('hxx/hist_1DZcc_pt_scalesmear.root')
    f = uproot.open('hxx/templates3.root')

    qcdpqq, qcdpcc, qcdpbb = 0., 0., 0.
    for ptbin in range(npt):
        pqqCh = rl.Channel("ptbin%d%s" % (ptbin, 'pqq'))
        pccCh = rl.Channel("ptbin%d%s" % (ptbin, 'pcc'))
        pbbCh = rl.Channel("ptbin%d%s" % (ptbin, 'pbb'))

        pqqTempl = get_templ2(f, "pqq", "qcd", ptbin)
        pccTempl = get_templ2(f, "pcc", "qcd", ptbin)
        pbbTempl = get_templ2(f, "pbb", "qcd", ptbin)

        pqqCh.setObservation(pqqTempl)
        pccCh.setObservation(pccTempl)
        pbbCh.setObservation(pbbTempl)

        qcdpqq += pqqCh.getObservation().sum()
        qcdpcc += pccCh.getObservation().sum()
        qcdpbb += pbbCh.getObservation().sum()

    qcdeff_c = qcdpcc / qcdpqq
    qcdeff_b = qcdpbb / qcdpqq

    # build actual fit model now
    model = rl.Model("temp3Model")

    regions = ['pbb', 'pcc', 'pqq']
    for ptbin in range(npt):
        for region in regions:
            ch = rl.Channel("ptbin%d%s" % (ptbin, region))
            model.addChannel(ch)
            # include_samples = ['zbb', 'zcc', 'zqq', 'wcq', 'wqq', 'hcc', 'tqq', 'hqq']
            include_samples = ['zcc', 'wqq']
            # Define mask
            mask = validbins[ptbin].copy()
            if not pseudo and region in ['pbb', 'pcc']:
                mask[10:14] = False

            if not fitTF:  # Add QCD sample when not running TF fit
                include_samples.append('qcd')
            for sName in include_samples:
                continue
                if use_matched:
                    templ = get_templ2M(f, region, sName, ptbin)
                else:
                    templ = get_templ2(f, region, sName, ptbin)
                stype = rl.Sample.SIGNAL if sName in ['zcc'] else rl.Sample.BACKGROUND
                sample = rl.TemplateSample(ch.name + '_' + sName, stype, templ)

                # Systematics
                sample.setParamEffect(sys_lumi, 1.023)

                ch.addSample(sample)

            if not pseudo:
                data_obs = get_templ2(f, region, 'data_obs', ptbin)
                if ptbin == 0 and region in ['pbb', 'pcc']:
                    print("Reading real data")

            else:
                yields = []
                if "qcd" not in include_samples:
                    include_samples = include_samples + ['qcd']
                for samp in include_samples:
                    if use_matched:
                        yields.append(get_templ2M(f, region, samp, ptbin)[0])
                    else:
                        yields.append(get_templ2(f, region, samp, ptbin)[0])
                yields = np.sum(np.array(yields), axis=0)
                if throwPoisson:
                    yields = np.random.poisson(yields)

                data_obs = (yields, msd.binning, msd.name)
            ch.setObservation(data_obs)

            # drop bins outside rho validity
            ch.mask = mask

    if fitTF:
        tf1_dataResidual = rl.BernsteinPoly("tf_dataResidual_cc", (2, 2), ['pt', 'rho'],
                                            limits=(0, 10))
        tf2_dataResidual = rl.BernsteinPoly("tf_dataResidual_bb", (2, 2), ['pt', 'rho'],
                                            limits=(0, 10))
        tf1_dataResidual_params = tf1_dataResidual(ptscaled, rhoscaled)
        tf2_dataResidual_params = tf2_dataResidual(ptscaled, rhoscaled)

        tf1_params = qcdeff_c * tf1_dataResidual_params
        tf2_params = qcdeff_b * tf2_dataResidual_params

        for ptbin in range(npt):
            pqqCh = model['ptbin%dpqq' % ptbin]
            pccCh = model['ptbin%dpcc' % ptbin]
            pbbCh = model['ptbin%dpbb' % ptbin]

            qcdparams = np.array([
                rl.IndependentParameter('qcdparam_ptbin%d_msdbin%d' % (ptbin, i), 0)
                for i in range(msd.nbins)
            ])

            initial_qcd = pqqCh.getObservation().astype(float)

            for sample in pqqCh:
                initial_qcd -= sample.getExpectation(nominal=True)
            if np.any(initial_qcd < 0.):
                raise ValueError("initial_qcd negative for some bins..", initial_qcd)

            sigmascale = 10  # to scale the deviation from initial
            scaledparams = initial_qcd * (
                1 + sigmascale / np.maximum(1., np.sqrt(initial_qcd)))**qcdparams
            pqq_qcd = rl.ParametericSample('ptbin%dpqq_qcd' % ptbin,
                                           rl.Sample.BACKGROUND, msd, scaledparams)

            pcc_qcd = rl.TransferFactorSample('ptbin%dpcc_qcd' % ptbin,
                                              rl.Sample.BACKGROUND,
                                              tf1_params[ptbin, :],
                                              pqq_qcd)
            pbb_qcd = rl.TransferFactorSample('ptbin%dpbb_qcd' % ptbin,
                                              rl.Sample.BACKGROUND,
                                              tf2_params[ptbin, :],
                                              pqq_qcd)
            pqqCh.addSample(pqq_qcd)
            pccCh.addSample(pcc_qcd)
            pbbCh.addSample(pbb_qcd)

    # vectorparams = []
    vectorparams = {}
    for reg in regions:
        for samp in ["zcc", "wqq"]:
            # vectorparams.append(rl.IndependentParameter('veff_%s_%s' % (samp, region), 1))
            vectorparams['veff_%s_%s' % (samp, reg)] = rl.IndependentParameter('veff_%s_%s' % (samp, reg), 1)

    print(vectorparams)
    if True:
        for ptbin in range(npt):
            for sName in ["zcc", "wqq"]:
                tot_templ = 0
                for reg in regions:
                    if use_matched:
                        tot_templ = np.sum(get_templ2M(f, reg, sName, ptbin)[0])
                    else:
                        tot_templ = np.sum(get_templ2(f, reg, sName, ptbin)[0])

                for region in regions:
                    chid = [ch.name for ch in model.channels].index("ptbin%d%s" % (ptbin, region))
                    ch = model.channels[chid]
                    
                    if use_matched:
                        templ = get_templ2M(f, region, sName, ptbin)
                    else:
                        templ = get_templ2(f, region, sName, ptbin)
                    norm_templ = templ[0].sum() / tot_templ
                    scale_templ = templ[0] * tot_templ / templ[0].sum()
                    templ = (scale_templ, templ[1], templ[2])
                    
                    stype = rl.Sample.SIGNAL if sName in ['zcc'] else rl.Sample.BACKGROUND
                    sample = rl.TemplateSample(ch.name + '_' + sName, stype, templ)
                    
                    sample.setParamEffect(vectorparams['veff_%s_%s' % (sName, region)],
                                          norm_templ * vectorparams['veff_%s_%s' % (sName, region)])
                    
                    ch.addSample(sample)
                
            break
                #stype = rl.Sample.SIGNAL if sName in ['zcc'] else rl.Sample.BACKGROUND
                #sample = rl.TemplateSample(ch.name + '_' + sName, stype, templ)

        

    with open("temp3Model.pkl", "wb") as fout:
        pickle.dump(model, fout)

    model.renderCombine("temp3Model")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument("--throwPoisson",
                        type=str2bool,
                        default='True',
                        choices={True, False},
                        help="If plotting data, redraw from poisson distribution")

    parser.add_argument("--MCTF",
                        type=str2bool,
                        default='False',
                        choices={True, False},
                        help="Fit QCD in MC first")
    parser.add_argument("--matched",
                        type=str2bool,
                        default='False',
                        choices={True, False},
                        help=("Use matched/unmatched templates"
                              "(w/o there is some W/Z/H contamination from QCD)"))

    pseudo = parser.add_mutually_exclusive_group(required=True)
    pseudo.add_argument('--data', action='store_false', dest='pseudo')
    pseudo.add_argument('--MC', action='store_true', dest='pseudo')

    args = parser.parse_args()

    dummy_rhalphabet(pseudo=args.pseudo,
                     throwPoisson=args.throwPoisson,
                     MCTF=args.MCTF,
                     use_matched=args.matched,
                     )
