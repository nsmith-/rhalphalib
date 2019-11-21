from __future__ import print_function, division
import warnings
import rhalphalib as rl
import numpy as np
import pickle
import ROOT
import uproot
from template_morph import AffineMorphTemplate
rl.util.install_roofit_helpers()


warnings.filterwarnings('error')

SF2017 = {  # cristina Jun25
    'shift_SF': 0.979,
    'shift_SF_ERR': 0.012,
    'smear_SF': 1.037,
    'smear_SF_ERR': 0.049,  # prelim SF @26% N2ddt
    'V_SF': 0.92,
    'V_SF_ERR': 0.018,
    'BB_SF': 1.0,
    'BB_SF_ERR': 0.3,  # prelim ddb SF
}


def ddx_SF(f, region, sName, ptbin, syst, mask):
    if region == "pass":
        return 1. + SF2017['BB_SF_ERR']/SF2017['BB_SF']
    else:
        _pass = get_templ(f, "pass", sName, ptbin)
        _pass_rate = np.sum(_pass[0] * mask)
        _fail = get_templ(f, "fail", sName, ptbin)
        _fail_rate = np.sum(_fail[0] * mask)
        if _fail_rate > 0:
            return 1. - SF2017['BB_SF_ERR'] * (_pass_rate/_fail_rate)
        else:
            return 1


def shape_to_num(f, region, sName, ptbin, syst, mask):
    _nom = get_templ(f, region, sName, ptbin)
    _nom_rate = np.sum(_nom[0] * mask)
    if _nom_rate < .1:
        return 1.0
    _up = get_templ(f, region, sName, ptbin, syst=syst+"Up")
    _up_rate = np.sum(_up[0] * mask)
    _down = get_templ(f, region, sName, ptbin, syst=syst+"Up")
    _down_rate = np.sum(_down[0] * mask)
    _diff = np.abs(_up_rate-_nom_rate) + np.abs(_down_rate-_nom_rate)
    return 1.0 + _diff / (2. * _nom_rate)


def get_templ(f, region, sample, ptbin, syst=None, read_sumw2=False):
    if sample in ["hcc", "hqq"]:
        sample += "125"
    hist_name = '{}_{}'.format(sample, region)
    if syst is not None:
        hist_name += "_" + syst
    h_vals = f[hist_name].values[:, ptbin]
    h_edges = f[hist_name].edges[0]
    h_key = 'msd'
    if read_sumw2:
        h_variances = f[hist_name].variances[:, ptbin]
        return (h_vals, h_edges, h_key, h_variances)
    return (h_vals, h_edges, h_key)


def dummy_rhalphabet(pseudo, throwPoisson, MCTF, scalesmear_syst):
    fitTF = True

    # Default lumi (needs at least one systematics for prefit)
    sys_lumi = rl.NuisanceParameter('CMS_lumi', 'lnN')
    # Systematics
    sys_JES = rl.NuisanceParameter('CMS_scale_j_2017 ', 'lnN')
    sys_JER = rl.NuisanceParameter('CMS_res_j_2017 ', 'lnN')
    sys_Pu = rl.NuisanceParameter('CMS_PU_2017', 'lnN')
    sys_trigger = rl.NuisanceParameter('CMS_gghcc_trigger_2018', 'lnN')

    sys_ddxeff = rl.NuisanceParameter('CMS_eff_cc', 'lnN')
    sys_eleveto = rl.NuisanceParameter('CMS_gghcc_e_veto', 'lnN')
    sys_muveto = rl.NuisanceParameter('CMS_gghcc_m_veto', 'lnN')

    sys_veff = rl.NuisanceParameter('CMS_gghcc_veff', 'lnN')
    sys_wznormEW = rl.NuisanceParameter('CMS_gghcc_wznormEW', 'lnN')
    sys_znormEW = rl.NuisanceParameter('CMS_gghcc_znormEW', 'lnN')
    sys_znormQ = rl.NuisanceParameter('CMS_gghcc_znormQ', 'lnN')

    sys_scale = rl.NuisanceParameter('scale', 'shape')
    sys_smear = rl.NuisanceParameter('smear', 'shape')

    sys_Hpt = rl.NuisanceParameter('CMS_gghbb_ggHpt', 'lnN')
    # sys_Hpt_shape = rl.NuisanceParameter('CMS_gghbb_ggHpt', 'shape')

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
    f = uproot.open('hxx/hist_1DZcc_pt_scalesmear.root')

    # Get QCD efficiency
    if MCTF:
        qcdmodel = rl.Model("qcdmodel")

    qcdpass, qcdfail = 0., 0.
    for ptbin in range(npt):
        failCh = rl.Channel("ptbin%d%s" % (ptbin, 'fail'))
        passCh = rl.Channel("ptbin%d%s" % (ptbin, 'pass'))

        passTempl = get_templ(f, "pass", "qcd", ptbin, read_sumw2=True)
        failTempl = get_templ(f, "fail", "qcd", ptbin, read_sumw2=True)

        failCh.setObservation(failTempl, read_sumw2=True)
        passCh.setObservation(passTempl, read_sumw2=True)
        qcdfail += failCh.getObservation()[0].sum()
        qcdpass += passCh.getObservation()[0].sum()

        if MCTF:
            qcdmodel.addChannel(failCh)
            qcdmodel.addChannel(passCh)

    qcdeff = qcdpass / qcdfail

    # Separate out QCD to QCD fit
    if MCTF:
        tf_MCtempl = rl.BernsteinPoly("tf_MCtempl", (2, 2), ['pt', 'rho'],
                                      limits=(0, 10))
        tf_MCtempl_params = qcdeff * tf_MCtempl(ptscaled, rhoscaled)

        for ptbin in range(npt):
            failCh = qcdmodel['ptbin%dfail' % ptbin]
            passCh = qcdmodel['ptbin%dpass' % ptbin]
            failObs = failCh.getObservation()[0]
            qcdparams = np.array([
                rl.IndependentParameter('qcdparam_ptbin%d_msdbin%d' % (ptbin, i), 0)
                for i in range(msd.nbins)
            ])
            sigmascale = 10.
            scaledparams = failObs * (
                1 + sigmascale / np.maximum(1., np.sqrt(failObs)))**qcdparams
            fail_qcd = rl.ParametericSample('ptbin%dfail_qcd' % ptbin,
                                            rl.Sample.BACKGROUND, msd, scaledparams)
            failCh.addSample(fail_qcd)
            pass_qcd = rl.TransferFactorSample('ptbin%dpass_qcd' % ptbin,
                                               rl.Sample.BACKGROUND,
                                               tf_MCtempl_params[ptbin, :], fail_qcd)
            passCh.addSample(pass_qcd)

            failCh.mask = validbins[ptbin]
            passCh.mask = validbins[ptbin]

        qcdfit_ws = ROOT.RooWorkspace('qcdfit_ws')
        simpdf, obs = qcdmodel.renderRoofit(qcdfit_ws)
        qcdfit = simpdf.fitTo(obs,
                              ROOT.RooFit.Extended(True),
                              ROOT.RooFit.SumW2Error(True),
                              ROOT.RooFit.Strategy(2),
                              ROOT.RooFit.Save(),
                              ROOT.RooFit.Minimizer('Minuit2', 'migrad'),
                              ROOT.RooFit.Offset(True),
                              ROOT.RooFit.PrintLevel(-1),
                              )
        qcdfit_ws.add(qcdfit)
        qcdfit_ws.writeToFile('qcdfit.root')
        if qcdfit.status() != 0:
            qcdfit.Print()
            raise RuntimeError('Could not fit qcd')

        param_names = [p.name for p in tf_MCtempl.parameters.reshape(-1)]
        decoVector = rl.DecorrelatedNuisanceVector.fromRooFitResult(
            tf_MCtempl.name + '_deco', qcdfit, param_names)
        tf_MCtempl.parameters = decoVector.correlated_params.reshape(
            tf_MCtempl.parameters.shape)
        tf_MCtempl_params_final = tf_MCtempl(ptscaled, rhoscaled)

    # build actual fit model now
    model = rl.Model("tempModel")

    for ptbin in range(npt):
        for region in ['pass', 'fail']:
            ch = rl.Channel("ptbin%d%s" % (ptbin, region))
            model.addChannel(ch)
            include_samples = ['zbb', 'zcc', 'zqq', 'wcq', 'wqq', 'hcc', 'tqq', 'hqq']
            # Define mask
            mask = validbins[ptbin].copy()
            if not pseudo and region == 'pass':
                mask[10:14] = False

            if not fitTF:  # Add QCD sample when not running TF fit
                include_samples.append('qcd')
            for sName in include_samples:
                templ = get_templ(f, region, sName, ptbin)
                stype = rl.Sample.SIGNAL if sName in ['zcc'] else rl.Sample.BACKGROUND
                sample = rl.TemplateSample(ch.name + '_' + sName, stype, templ)

                # Systematics
                sample.setParamEffect(sys_lumi, 1.023)

                # Shape systematics
                # Not actuall in ggH
                # sys_names = ['JES', "JER", 'trigger', 'Pu']
                # sys_list = [JES, JER, trigger, Pu]
                # for sys_name, sys in zip(sys_names, sys_list):
                #     _up = get_templ(f, region, sName, ptbin, syst=sys_name+"Up")
                #     _dn = get_templ(f, region, sName, ptbin, syst=sys_name+"Down")
                #     sample.setParamEffect(sys, _up[0], _dn[0])

                sys_names = ['JES', "JER", 'Pu']
                sys_list = [sys_JES, sys_JER, sys_Pu]
                for sys_name, sys in zip(sys_names, sys_list):
                    _sys_ef = shape_to_num(f, region, sName, ptbin, sys_name, mask)
                    sample.setParamEffect(sys, _sys_ef)

                # Sample specific
                if sName not in ["qcd"]:
                    sample.setParamEffect(sys_eleveto, 1.005)
                    sample.setParamEffect(sys_muveto, 1.005)
                    sample.setParamEffect(sys_lumi, 1.025)
                    sample.setParamEffect(sys_trigger, 1.02)
                if sName not in ["qcd", 'tqq']:
                    sample.setParamEffect(sys_veff,
                                          1.0 + SF2017['V_SF_ERR'] / SF2017['V_SF'])
                if sName not in ["qcd", "tqq", "wqq", "zqq"]:
                    sample.setParamEffect(
                        sys_ddxeff, ddx_SF(f, region, sName, ptbin, sys_name, mask))
                if sName.startswith("z"):
                    sample.setParamEffect(sys_znormQ, 1.1)
                    sample.setParamEffect(sys_znormEW, 1.05)
                if sName.startswith("w"):
                    sample.setParamEffect(sys_znormQ, 1.1)
                    sample.setParamEffect(sys_znormEW, 1.05)
                    sample.setParamEffect(sys_wznormEW, 1.02)
                if sName.startswith("h"):
                    sample.setParamEffect(sys_Hpt, 1.2)

                if scalesmear_syst:
                    # Scale and Smear
                    mtempl = AffineMorphTemplate((templ[0], templ[1]))
                    # import pprint.pprint as pprint
                    np.set_printoptions(linewidth=1000, precision=2)
                    if sName == "zcc" and ptbin == 4:
                        print(region)
                        print(templ[0])
                        print(np.sum(templ[0]))
                        print(mtempl.get(shift=-7.)[0])
                        print(mtempl.get(shift=7.)[0])
                    sample.setParamEffect(sys_scale,
                                          mtempl.get(shift=7.)[0],
                                          mtempl.get(shift=-7.)[0])

                ch.addSample(sample)

            if not pseudo:
                data_obs = get_templ(f, region, 'data_obs', ptbin)
                if ptbin == 0 and region == "pass":
                    print("Reading real data")

            else:
                yields = []
                for samp in include_samples + ['qcd']:
                    yields.append(get_templ(f, region, samp, ptbin)[0])
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
        if MCTF:
            tf_params = qcdeff * tf_MCtempl_params_final * tf_dataResidual_params
        else:
            tf_params = qcdeff * tf_dataResidual_params

        for ptbin in range(npt):
            failCh = model['ptbin%dfail' % ptbin]
            passCh = model['ptbin%dpass' % ptbin]

            qcdparams = np.array([
                rl.IndependentParameter('qcdparam_ptbin%d_msdbin%d' % (ptbin, i), 0)
                for i in range(msd.nbins)
            ])
            initial_qcd = failCh.getObservation().astype(float)
            # was integer, and numpy complained about subtracting float from it
            for sample in failCh:
                initial_qcd -= sample.getExpectation(nominal=True)
            if np.any(initial_qcd < 0.):
                raise ValueError("initial_qcd negative for some bins..", initial_qcd)
            sigmascale = 10  # to scale the deviation from initial
            scaledparams = initial_qcd * (
                1 + sigmascale / np.maximum(1., np.sqrt(initial_qcd)))**qcdparams
            fail_qcd = rl.ParametericSample('ptbin%dfail_qcd' % ptbin,
                                            rl.Sample.BACKGROUND, msd, scaledparams)
            failCh.addSample(fail_qcd)
            pass_qcd = rl.TransferFactorSample('ptbin%dpass_qcd' % ptbin,
                                               rl.Sample.BACKGROUND,
                                               tf_params[ptbin, :], fail_qcd)
            passCh.addSample(pass_qcd)

        # tqqpass = passCh['tqq']
        # tqqfail = failCh['tqq']
        # tqqPF = tqqpass.getExpectation(nominal=True).sum() \
        #     / tqqfail.getExpectation(nominal=True).sum()
        # tqqpass.setParamEffect(tqqeffSF, 1*tqqeffSF)
        # tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
        # tqqpass.setParamEffect(tqqnormSF, 1*tqqnormSF)
        # tqqfail.setParamEffect(tqqnormSF, 1*tqqnormSF)

    # Fill in muon CR
    # templates = {}
    # for ptbin in range(npt):
    #     for region in ['pass', 'fail']:
    #         templates[region+str(ptbin)] = {}

    # for region in ['pass', 'fail']:
    #     ch = rl.Channel("muonCR%s" % (region, ))
    #     model.addChannel(ch)

    #     templates = {}
    #     f = uproot.open(
    #         'hxx/hist_1DZcc_pt_scalesmear.root')
    #     for sample in ['tqq', 'qcd']:
    #         hist_name = '{}_{}'.format(sample, region)
    #         h_vals = f[hist_name].values[:, ptbin]
    #         h_edges = f[hist_name].edges[0]
    #         h_key = 'msd'
    #         templates[sample.replace('125', '')] = (h_vals, h_edges, h_key)

    #     for sName, templ in templates.items():
    #         stype = rl.Sample.BACKGROUND
    #         sample = rl.TemplateSample(ch.name + '_' + sName, stype, templ)

    #         # mock systematics
    #         jecup_ratio = np.random.normal(loc=1, scale=0.05, size=msd.nbins)
    #         sample.setParamEffect(jec, jecup_ratio)

    #         ch.addSample(sample)

    #     yields = sum(tpl[0] for tpl in templates.values())
    #     if throwPoisson:
    #         yields = np.random.poisson(yields)
    #     data_obs = (yields, msd.binning, msd.name)
    #     ch.setObservation(data_obs)

    # tqqpass = model['muonCRpass_tqq']
    # tqqfail = model['muonCRfail_tqq']
    # tqqPF = tqqpass.getExpectation(nominal=True).sum() / tqqfail.getExpectation(
    #     nominal=True).sum()
    # tqqpass.setParamEffect(tqqeffSF, 1*tqqeffSF)
    # tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
    # tqqpass.setParamEffect(tqqnormSF, 1*tqqnormSF)
    # tqqfail.setParamEffect(tqqnormSF, 1*tqqnormSF)

    with open("tempModel.pkl", "wb") as fout:
        pickle.dump(model, fout)

    model.renderCombine("tempModel")


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
                        help="ToFix, Fit QCD in MC first")

    parser.add_argument("--scale",
                        type=str2bool,
                        default='False',
                        choices={True, False},
                        help="ToFix, Generate with scale/smear systematics")

    pseudo = parser.add_mutually_exclusive_group(required=True)
    pseudo.add_argument('--data', action='store_false', dest='pseudo')
    pseudo.add_argument('--MC', action='store_true', dest='pseudo')

    args = parser.parse_args()

    dummy_rhalphabet(pseudo=args.pseudo,
                     throwPoisson=args.throwPoisson,
                     MCTF=args.MCTF,
                     scalesmear_syst=args.scale,
                     )
