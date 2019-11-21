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
    except:
        print("Warning: template {} was not found, replaces with [0, 0, ...0]".format(
            hist_name))
        h_vals = np.zeros_like(f[hist_name.replace('bin5', 'bin4')].values)
        h_edges = f[hist_name.replace('bin5', 'bin4')].edges
    h_key = 'msd'
    return (h_vals, h_edges, h_key)


def dummy_rhalphabet(pseudo, throwPoisson, MCTF):
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

    qcdpass, qcdfail = 0., 0.
    for ptbin in range(npt):
        failCh = rl.Channel("ptbin%d%s" % (ptbin, 'fail'))
        passCh = rl.Channel("ptbin%d%s" % (ptbin, 'pass'))

        passTempl = get_templ2(f, "pqq", "qcd", ptbin)
        failTempl = get_templ2(f, "pcc", "qcd", ptbin)

        failCh.setObservation(failTempl)
        passCh.setObservation(passTempl)
        qcdfail += failCh.getObservation().sum()
        qcdpass += passCh.getObservation().sum()

    qcdeff = qcdpass / qcdfail

    # build actual fit model now
    model = rl.Model("temp3Model")

    for ptbin in range(npt):
        for region in ['pbb', 'pcc', 'pqq']:
            ch = rl.Channel("ptbin%d%s" % (ptbin, region))
            model.addChannel(ch)
            include_samples = ['zbb', 'zcc', 'zqq', 'wcq', 'wqq', 'hcc', 'tqq', 'hqq']
            # Define mask
            mask = validbins[ptbin].copy()
            if not pseudo and region in ['pbb', 'pcc']:
                mask[10:14] = False

            if not fitTF:  # Add QCD sample when not running TF fit
                include_samples.append('qcd')
            for sName in include_samples:
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
                for samp in include_samples + ['qcd']:
                    yields.append(get_templ2(f, region, samp, ptbin)[0])
                yields = np.sum(np.array(yields), axis=0)
                if throwPoisson:
                    yields = np.random.poisson(yields)

                data_obs = (yields, msd.binning, msd.name)
            ch.setObservation(data_obs)

            # drop bins outside rho validity
            ch.mask = mask

    if fitTF:
        tf_dataResidual = rl.BernsteinPoly("tf_dataResidual", (2, 2), ['pt', 'rho'],
                                           limits=(0, 10))
        tf_dataResidual_params = tf_dataResidual(ptscaled, rhoscaled)

        tf_params = qcdeff * tf_dataResidual_params

        for ptbin in range(npt):
            pqqCh = model['ptbin%dpqq' % ptbin]
            pccCh = model['ptbin%dpcc' % ptbin]

            qcdparams = np.array([
                rl.IndependentParameter('qcdparam_ptbin%d_msdbin%d' % (ptbin, i), 0)
                for i in range(msd.nbins)
            ])
            initial_qcd = pqqCh.getObservation().astype(float)
            # was integer, and numpy complained about subtracting float from it
            for sample in pqqCh:
                initial_qcd -= sample.getExpectation(nominal=True)
            if np.any(initial_qcd < 0.):
                raise ValueError("initial_qcd negative for some bins..", initial_qcd)
            sigmascale = 10  # to scale the deviation from initial
            scaledparams = initial_qcd * (
                1 + sigmascale / np.maximum(1., np.sqrt(initial_qcd)))**qcdparams
            pqq_qcd = rl.ParametericSample('ptbin%dpqq_qcd' % ptbin,
                                           rl.Sample.BACKGROUND, msd, scaledparams)
            pqqCh.addSample(pqq_qcd)
            pcc_qcd = rl.TransferFactorSample('ptbin%dpcc_qcd' % ptbin,
                                              rl.Sample.BACKGROUND, tf_params[ptbin, :],
                                              pqq_qcd)
            pccCh.addSample(pcc_qcd)

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

    pseudo = parser.add_mutually_exclusive_group(required=True)
    pseudo.add_argument('--data', action='store_false', dest='pseudo')
    pseudo.add_argument('--MC', action='store_true', dest='pseudo')

    args = parser.parse_args()

    dummy_rhalphabet(pseudo=args.pseudo,
                     throwPoisson=args.throwPoisson,
                     MCTF=args.MCTF
                     )
