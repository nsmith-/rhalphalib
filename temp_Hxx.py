from __future__ import print_function, division
import warnings
import rhalphalib as rl
import pickle
import numpy as np
import ROOT
import uproot
from template_morph import AffineMorphTemplate
#import matplotlib.pyplot as plt
#import mplhep as hep
#plt.style.use(hep.style.ROOT)
#plt.switch_backend('agg')

rl.util.install_roofit_helpers()


# warnings.filterwarnings('error')

SF = {
    # https://github.com/kakwok/ZPrimePlusJet/blob/fidxs/fitting/PbbJet/buildRhalphabetHbb.py
    "2017": {
        'shift_SF': 0.978,
        'shift_SF_ERR': 0.012,
        'smear_SF': 0.9045,
        'smear_SF_ERR': 0.048,
        'V_SF': 0.924,
        'V_SF_ERR': 0.018,
        'CC_SF': 0.9,  # 1.0,
        'CC_SF_ERR': .8  # 0.3,  # prelim ddb SF
    },
    "2018": {
        'shift_SF': 0.970,
        'shift_SF_ERR': 0.012,
        'smear_SF': 0.9076,
        'smear_SF_ERR': 0.0146,
        'V_SF': 0.953,
        'V_SF_ERR': 0.016,
        'CC_SF': 0.9,  # 1.0,
        'CC_SF_ERR': .8  # 0.3,  # prelim ddb SF
    },
    "2016": {
        'V_SF': 0.993,
        'V_SF_ERR': 0.043,
        'shift_SF': 1.001,
        'shift_SF_ERR': 0.012,
        'smear_SF': 1.084,
        'smear_SF_ERR': 0.0905,
        'CC_SF': 0.9,  # 1.0,
        'CC_SF_ERR': .8  # 0.3,  # prelim ddb SF
    }
}


def ddx_SF(f, region, sName, ptbin, mask,
           SF=SF["2017"]['CC_SF'], SF_unc=SF["2017"]['CC_SF_ERR']):
    if region == "pass":
        return 1. + SF_unc/SF
    else:
        _pass = get_templX(f, "pass", sName, ptbin)
        _pass_rate = np.sum(_pass[0] * mask)

        _fail = get_templX(f, "fail", sName, ptbin)
        _fail_rate = np.sum(_fail[0] * mask)

        if _fail_rate > 0:
            return 1. - SF_unc * (_pass_rate/_fail_rate)
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


def get_templ(f, region, sample, ptbin, syst=None, read_sumw2=False, muon=False):
    # if sample in ["hcc", "hqq"]:
    #     sample += "125"
    if sample in ["hcc"]:
        sample += "125"
    if sample in ['hbb', 'zhbb', 'vbfhbb', 'whbb', 'tthbb']:
        sample = sample.replace("bb", "qq")+"125"
    hist_name = '{}_{}'.format(sample, region)
    if syst is not None:
        hist_name += "_" + syst
    if syst is None and muon:
        hist_name += "_"
    if muon:
        h_vals = f[hist_name].values
        h_edges = f[hist_name].edges
    else:
        h_vals = f[hist_name].values[:, ptbin]
        h_edges = f[hist_name].edges[0]
    h_key = 'msd'
    if read_sumw2:
        h_variances = f[hist_name].variances[:, ptbin]
        return (h_vals, h_edges, h_key, h_variances)
    return (h_vals, h_edges, h_key)


def get_templM(f, region, sample, ptbin, syst=None, read_sumw2=False, muon=False):
    if sample in ["hcc"]:
        sample += "125"
    if sample in ['hbb', 'zhbb', 'vbfhbb', 'whbb', 'tthbb']:
        sample = sample.replace("bb", "qq")+"125"
    hist_name = '{}_{}'.format(sample, region)
    if syst is not None:
        hist_name += "_" + syst
    if syst is None and muon:
        hist_name += "_"
    if (sample.startswith("w") or sample.startswith("z")
            or sample.startswith("h")) and syst is None:
        hist_name += "_" + 'matched'
    if muon:
        h_vals = f[hist_name].values
        h_edges = f[hist_name].edges
    else:
        h_vals = f[hist_name].values[:, ptbin]
        h_edges = f[hist_name].edges[0]
    h_key = 'msd'
    if read_sumw2:
        h_variances = f[hist_name].variances[:, ptbin]
        return (h_vals, h_edges, h_key, h_variances)
    return (h_vals, h_edges, h_key)


def shape_to_numM(f, region, sName, ptbin, syst, mask):
    # With Matched logic
    _nom = get_templM(f, region, sName, ptbin)
    _nom_rate = np.sum(_nom[0] * mask)
    if _nom_rate < .1:
        return 1.0
    _up = get_templ(f, region, sName, ptbin, syst=syst+"Up")
    _up_unmatched = get_templ(f, region, sName, ptbin, 'unmatched')
    _up_rate = np.sum(_up[0] * mask) - np.sum(_up_unmatched[0] * mask)
    _down = get_templ(f, region, sName, ptbin, syst=syst+"Up")
    _down_unmatched = get_templ(f, region, sName, ptbin, 'unmatched')
    _down_rate = np.sum(_down[0] * mask) - np.sum(_down_unmatched[0] * mask)
    _diff = np.abs(_up_rate-_nom_rate) + np.abs(_down_rate-_nom_rate)
    return 1.0 + _diff / (2. * _nom_rate)


def dummy_rhalphabet(pseudo, throwPoisson, MCTF, justZ=False,
                     scale_syst=True, smear_syst=True, systs=True,
                     blind=True, runhiggs=False, fitTF=True, muonCR=True,
                     runboth=False, year=2017,
                     opts=None
                     ):

    # Default lumi (needs at least one systematics for prefit)
    sys_lumi = rl.NuisanceParameter('CMS_lumi', 'lnN')
    # TT params
    tqqeffSF = rl.IndependentParameter('tqqeffSF', 1., 0, 10)
    tqqnormSF = rl.IndependentParameter('tqqnormSF', 1., 0, 10)
    # Systematics
    sys_JES = rl.NuisanceParameter('CMS_scale_j_{}'.format(year), 'lnN')
    sys_JER = rl.NuisanceParameter('CMS_res_j_{}'.format(year), 'lnN')
    sys_Pu = rl.NuisanceParameter('CMS_PU_{}_'.format(year), 'lnN')
    sys_trigger = rl.NuisanceParameter('CMS_gghcc_trigger_{}'.format(year), 'lnN')

    sys_ddxeff = rl.NuisanceParameter('CMS_eff_cc', 'lnN')
    sys_ddxeffbb = rl.NuisanceParameter('CMS_eff_bb', 'lnN')
    sys_ddxeffw = rl.NuisanceParameter('CMS_eff_w', 'lnN')
    sys_eleveto = rl.NuisanceParameter('CMS_gghcc_e_veto', 'lnN')
    sys_muveto = rl.NuisanceParameter('CMS_gghcc_m_veto', 'lnN')

    sys_veff = rl.NuisanceParameter('CMS_gghcc_veff', 'lnN')
    sys_wznormEW = rl.NuisanceParameter('CMS_gghcc_wznormEW', 'lnN')
    sys_znormEW = rl.NuisanceParameter('CMS_gghcc_znormEW', 'lnN')
    sys_znormQ = rl.NuisanceParameter('CMS_gghcc_znormQ', 'lnN')

    sys_scale = rl.NuisanceParameter('CMS_gghcc_scale', 'shape')
    sys_smear = rl.NuisanceParameter('CMS_gghcc_smear', 'shape')

    sys_Hpt = rl.NuisanceParameter('CMS_gghcc_ggHpt', 'lnN')
    # sys_Hpt_shape = rl.NuisanceParameter('CMS_gghbb_ggHpt', 'shape')

    # Import binnings
    # Hidden away to be available to other functions
    from config_Hxx import ptbins, msdbins  # ptpts, msdpts, rhopts
    from config_Hxx import ptscaled, rhoscaled, validbins
    msd = rl.Observable('msd', msdbins)
    npt = len(ptbins) - 1

    # Year setup
    if year == "2018":
        print("Year: 2018")
        model_name = "temp18Model"
        #f = uproot.open('hxx18/hist_1DZcc_pt_scalesmear.root')
        f = uproot.open('2018v2/hist_1DZcc_pt_scalesmear.root')
        f_mu = uproot.open('2018v2/hist_1DZcc_muonCR.root')
    elif year == "2017":
        print("Year: 2017")
        model_name = "temp17Model"
        #f = uproot.open('hxx/hist_1DZcc_pt_scalesmear.root')
        f = uproot.open('2017v2/hist_1DZcc_pt_scalesmear.root')
        f_mu = uproot.open('2017v2/hist_1DZcc_muonCR.root')
    elif year == "2016":
        print("Year: 2016")
        model_name = "temp16Model"
        f = uproot.open('2016v2/hist_1DZcc_pt_scalesmear.root')
        f_mu = uproot.open('2016v2/hist_1DZcc_muonCR.root')
    else:
        raise ValueError("Invalid Year")

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
        degsMC = tuple([int(s) for s in opts.degsMC.split(',')])
        tf_MCtempl = rl.BernsteinPoly("tf_MCtempl", degsMC, ['pt', 'rho'],
                                      limits=(-50, 50))
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

        qcdmodel.readRooFitResult(qcdfit)

        # # Plot it
        from _plot_TF import TF_smooth_plot, TF_params
        from _plot_TF import plotTF as plotMCTF
        from utils import make_dirs
        _values = [par.value for par in tf_MCtempl.parameters.flatten()]
        _names = [par.name for par in tf_MCtempl.parameters.flatten()]
        make_dirs('{}/plots'.format(model_name))
        np.save('{}/MCTF'.format(model_name), _values)
        print('ptdeg', degsMC[0], 'rhodeg', degsMC[1])
        plotMCTF(*TF_smooth_plot(*TF_params(_values, _names)), MC=True, raw=True,
                 ptdeg=degsMC[0], rhodeg=degsMC[1],
                 out='{}/plots/TF_MC_only'.format(model_name))

        param_names = [p.name for p in tf_MCtempl.parameters.reshape(-1)]
        decoVector = rl.DecorrelatedNuisanceVector.fromRooFitResult(
            tf_MCtempl.name + '_deco', qcdfit, param_names)
        np.save('{}/decoVector'.format(model_name), decoVector._transform)
        tf_MCtempl.parameters = decoVector.correlated_params.reshape(
            tf_MCtempl.parameters.shape)
        tf_MCtempl_params_final = tf_MCtempl(ptscaled, rhoscaled)

    # build actual fit model now
    model = rl.Model(model_name)

    for ptbin in range(npt):
        for region in ['pass', 'fail']:
            ch = rl.Channel("ptbin%d%s" % (ptbin, region))
            model.addChannel(ch)
            if justZ:
                include_samples = ['zcc']
            elif opts.justHZ is True:
                include_samples = ['zcc', "hcc"]
            else:
                include_samples = ['zbb', 'zcc', 'zqq', 'wcq', 'wqq', 'tqq', 'stqq',
                                   'hcc',
                                   'hbb', 'zhbb', 'vbfhbb', 'whbb', 'tthbb',  # hbb signals
                                  ]
            # Define mask
            mask = validbins[ptbin].copy()
            if not pseudo and region == 'pass':
                if blind and 'hbb' in include_samples:
                    mask[10:14] = False
            # Remove empty samples
            for sName in include_samples:
                templ = get_templX(f, region, sName, ptbin)
                if np.sum(templ[0][mask]) < 0.00001:
                    print('Sample {} in region = {}, ptbin = {}, would be empty, so it will be removed'.format(sName, region, ptbin))
                    include_samples.remove(sName)

            if not fitTF:  # Add QCD sample when not running TF fit
                include_samples.append('qcd')
            for sName in include_samples:
                templ = get_templX(f, region, sName, ptbin)
                if runhiggs:
                    _signals = ["hcc"]
                elif runboth:
                    _signals = ["hcc", "zcc"]
                else:
                    _signals = ["zcc"]
                stype = rl.Sample.SIGNAL if sName in _signals else rl.Sample.BACKGROUND

                sample = rl.TemplateSample(ch.name + '_' + sName, stype, templ)
                #print(sName, region, ptbin,  np.sum(templ[0]))
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

                #####################################################
                if systs:
                    sys_names = ['JES', "JER", 'Pu']
                    sys_list = [sys_JES, sys_JER, sys_Pu]
                    for sys_name, sys in zip(sys_names, sys_list):
                        _sys_ef = shape_to_numX(f, region, sName, ptbin, sys_name, mask)
                        sample.setParamEffect(sys, _sys_ef)

                    # Sample specific
                    if sName not in ["qcd"]:
                        sample.setParamEffect(sys_eleveto, 1.005)
                        sample.setParamEffect(sys_muveto, 1.005)
                        sample.setParamEffect(sys_lumi, 1.025)
                        sample.setParamEffect(sys_trigger, 1.02)
                    if sName not in ["qcd", 'tqq', 'stqq']:
                        sample.scale(SF[year]['V_SF'])
                        sample.setParamEffect(sys_veff,
                                            1.0 + SF[year]['V_SF_ERR'] / SF[year]['V_SF'])
                        #1.3)
                    if sName in ["zcc", "hcc"]:
                        sample.scale(SF[year]['CC_SF'])
                        sample.setParamEffect(
                            sys_ddxeff,
                            ddx_SF(f, region, sName, ptbin, mask, SF_unc=0.3))
                    if sName in ["zbb", "hbb", 'zhbb', 'vbfhbb', 'whbb', 'tthbb']:
                        # 1 +- 0.3
                        sample.setParamEffect(
                            sys_ddxeffbb,
                            ddx_SF(f, region, sName, ptbin, mask, SF=1, SF_unc=0.3))
                    # if sName in ["wcq", "wqq"]:
                    #     # 1 +- 0.3
                    #     sample.setParamEffect(
                    #         sys_ddxeffw,
                    #         ddx_SF(f, region, sName, ptbin, mask, use_matched,
                    #             SF=1, SF_unc=0.3))
                    if sName.startswith("z"):
                        sample.setParamEffect(sys_znormQ, 1.1)
                        if ptbin >= 2:
                            sample.setParamEffect(sys_znormEW, 1.07)
                        else:
                            sample.setParamEffect(sys_znormEW, 1.05)
                    if sName.startswith("w"):
                        sample.setParamEffect(sys_znormQ, 1.1)
                        if ptbin >= 2:
                            sample.setParamEffect(sys_znormEW, 1.07)
                        else:
                            sample.setParamEffect(sys_znormEW, 1.05)
                        if ptbin >= 3:
                            sample.setParamEffect(sys_wznormEW, 1.06)
                        else:
                            sample.setParamEffect(sys_wznormEW, 1.02)
                    if sName.startswith("h"):
                        sample.setParamEffect(sys_Hpt, 1.2)

                # Scale and Smear
                mtempl = AffineMorphTemplate(templ)

                if scale_syst and sName not in ["qcd", 'tqq', 'stqq']:
                    # import pprint.pprint as pprint
                    # np.set_printoptions(linewidth=1000, precision=2)
                    if sName.startswith("h"):
                        _mass = 125.
                    elif sName.startswith("w"):
                        _mass = 80.4
                    elif sName.startswith("z"):
                        _mass = 91.
                    else:
                        pass
                    realshift = _mass * SF[year]['shift_SF'] * SF[year]['shift_SF_ERR']
                    # realshift = 90 * 0.01
                    sample.setParamEffect(sys_scale,
                                          mtempl.get(shift=7.),
                                          mtempl.get(shift=-7.),
                                          scale=realshift/7.)

                if smear_syst and sName not in ["qcd", 'tqq', 'stqq']:
                    # To Do
                    # Match to boson mass instead of mean
                    # smear_in, smear_unc = 1, 0.11
                    smear_in, smear_unc = SF[year]['smear_SF'], SF[year]['smear_SF_ERR']
                    #smear_in, smear_unc = 1, 0.3
                    _smear_up = mtempl.get(scale=smear_in + 1 * smear_unc,
                                           shift=-mtempl.mean *
                                           (smear_in + 1 * smear_unc - 1))
                    #print(_smear_up)
                    _smear_down = mtempl.get(scale=smear_in + -1 * smear_unc,
                                             shift=-mtempl.mean *
                                             (smear_in + -1 * smear_unc - 1))
                    #print(_smear_down)
                    sample.setParamEffect(sys_smear, _smear_up, _smear_down)

                ch.addSample(sample)

            if not pseudo:
                data_obs = get_templ(f, region, 'data_obs', ptbin)
                if ptbin == 0 and region == "pass": print("Reading real data")

            else:
                yields = []
                if 'qcd' not in include_samples:
                    include_samples = include_samples + ['qcd']
                for samp in include_samples:
                    if samp == "qcd" and opts.mockQCD and region == "pass":
                        _temp_yields = get_templX(f, "fail", samp, ptbin)[0] * qcdeff * np.linspace(0.8, 1.2, len(get_templX(f, "fail", samp, ptbin)[0]))
                    else:
                        _temp_yields = get_templX(f, region, samp, ptbin)[0]
                    if samp not in ['qcd', 'tqq', 'stqq'] and systs:
                        _temp_yields *= SF[year]['V_SF']
                    yields.append(_temp_yields)
                yields = np.sum(np.array(yields), axis=0)
                if throwPoisson:
                    yields = np.random.poisson(yields)

                data_obs = (yields, msd.binning, msd.name)
            ch.setObservation(data_obs)

            # drop bins outside rho validity
            ch.mask = mask

    if fitTF:
        degs = tuple([int(s) for s in opts.degs.split(',')])
        tf_dataResidual = rl.BernsteinPoly("tf_dataResidual", degs, ['pt', 'rho'],
                                           limits=(-50, 50))
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
                # raise ValueError("initial_qcd negative for some bins..", initial_qcd)
                warnings.warn("initial_qcd negative for some bins..", UserWarning)
                print(initial_qcd)
                warnings.warn("Negative bins will be forced positive", UserWarning)
                initial_qcd[0 > initial_qcd] = abs(initial_qcd[0 > initial_qcd])
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

    if muonCR:
        for ptbin in range(npt):
            failCh = model['ptbin%dfail' % ptbin]
            passCh = model['ptbin%dpass' % ptbin]
            tqqpass = passCh['tqq']
            tqqfail = failCh['tqq']
            tqqPF = tqqpass.getExpectation(nominal=True).sum() \
                / tqqfail.getExpectation(nominal=True).sum()
            tqqpass.setParamEffect(tqqeffSF, 1*tqqeffSF)
            tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
            tqqpass.setParamEffect(tqqnormSF, 1*tqqnormSF)
            tqqfail.setParamEffect(tqqnormSF, 1*tqqnormSF)

    # Fill in muon CR
    if muonCR:
        for region in ['pass', 'fail']:
            ch = rl.Channel("muonCR%s" % (region, ))
            model.addChannel(ch)
            include_samples = ["qcd", "tqq", 'stqq']

            for sName in include_samples:

                templ = get_templX(f_mu, region, sName, ptbin, muon=True)
                stype = rl.Sample.BACKGROUND
                sample = rl.TemplateSample(ch.name + '_' + sName, stype, templ)

                # mock systematics
                #jecup_ratio = np.random.normal(loc=1, scale=0.05, size=msd.nbins)
                #sample.setParamEffect(jec, jecup_ratio)
                sample.setParamEffect(sys_lumi, 1.023)

                ch.addSample(sample)

            if not pseudo:
                data_obs = get_templ(f_mu, region, 'data_obs', ptbin, muon=True)
                if ptbin == 0 and region == "pass":
                    print("Reading real data")

            else:
                yields = []
                for samp in include_samples:
                    _temp_yields = get_templX(f_mu, region, samp, ptbin, muon=True)[0]
                    # if samp not in ['qcd', 'tqq'] and systs:
                    #     _temp_yields *= SF[year]['V_SF']
                    yields.append(_temp_yields)
                yields = np.sum(np.array(yields), axis=0)
                if throwPoisson:
                    yields = np.random.poisson(yields)
                data_obs = (yields, msd.binning, msd.name)

            _nbinsmu = len(data_obs[0])

            ch.setObservation(data_obs)

        tqqpass = model['muonCRpass_tqq']
        tqqfail = model['muonCRfail_tqq']
        tqqPF = tqqpass.getExpectation(nominal=True).sum() / tqqfail.getExpectation(
            nominal=True).sum()
        tqqpass.setParamEffect(tqqeffSF, 1*tqqeffSF)
        tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
        tqqpass.setParamEffect(tqqnormSF, 1*tqqnormSF)
        tqqfail.setParamEffect(tqqnormSF, 1*tqqnormSF)

    with open("{}.pkl".format(model_name), "wb") as fout:
        pickle.dump(model, fout)

    model.renderCombine(model_name)

    conf_dict = vars(opts)
    # add info for F-test
    conf_dict['NBINS'] = np.sum(validbins)
    conf_dict['NBINSMU'] = _nbinsmu if muonCR else 0

    import json
    # Serialize data into file:
    json.dump(conf_dict,
              open("{}/config.json".format(model_name), 'w'),
              sort_keys=True,
              indent=4,
              separators=(',', ': '))


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
                        default='False',
                        choices={True, False},
                        help="If plotting data, redraw from poisson distribution")

    parser.add_argument("--fitTF",
                        type=str2bool,
                        default='True',
                        choices={True, False},
                        help="Fit TF for QCD")

    parser.add_argument("--MCTF",
                        type=str2bool,
                        default='True',
                        choices={True, False},
                        help="Fit QCD in MC first")

    parser.add_argument("--muCR",
                        type=str2bool,
                        default='True',
                        choices={True, False},
                        help="Include muonCR to constrain ttbar")

    parser.add_argument("--scale",
                        type=str2bool,
                        default='True',
                        choices={True, False},
                        help="Include scale systematic")

    parser.add_argument("--smear",
                        type=str2bool,
                        default='True',
                        choices={True, False},
                        help="Include smear systematics")

    parser.add_argument("--systs",
                        type=str2bool,
                        default='True',
                        choices={True, False},
                        help="Include all systematics (separate from scale/smear)")

    parser.add_argument("--justZ",
                        type=str2bool,
                        default='False',
                        choices={True, False},
                        help="Only run Z sample with QCD")

    parser.add_argument("--justHZ",
                        type=str2bool,
                        default='False',
                        choices={True, False},
                        help="Only run H and Z sample with QCD")

    parser.add_argument("--year",
                        type=int,
                        default=2017,
                        help="Year")

    parser.add_argument("--matched",
                        type=str2bool,
                        default='True',
                        choices={True, False},
                        help=("Use matched/unmatched templates"
                              "(w/o there is some W/Z/H contamination from QCD)"))

    pseudo = parser.add_mutually_exclusive_group(required=True)
    pseudo.add_argument('--data', action='store_false', dest='pseudo')
    pseudo.add_argument('--MC', action='store_true', dest='pseudo')

    parser.add_argument('--unblind', action='store_true', dest='unblind')
    parser.add_argument('--higgs', action='store_true', dest='runhiggs', help="Set Higgs as signal instead of z")
    parser.add_argument('--both', action='store_true', dest='runboth', help="Both Z and H signals")

    parser.add_argument('--mockQCD', action='store_true', dest='mockQCD', help="Replace true pass QCD with scaled true fail QCD in pseudo data")

    parser.add_argument("--degs",
                        type=str,
                        default='1,1',
                        help="Polynomial degrees in the shape 'pt,rho' e.g. '2,2'")

    parser.add_argument("--degsMC",
                        type=str,
                        default='1,2',
                        help="Polynomial degrees in the shape 'pt,rho' e.g. '2,2'")


    args = parser.parse_args()
    print("Running with options:")
    print("    ", args)

    def get_templX(f, region, sample, ptbin, syst=None, read_sumw2=False, muon=False):
        if args.matched:
            return get_templM(f,
                              region,
                              sample,
                              ptbin,
                              syst=syst,
                              read_sumw2=read_sumw2,
                              muon=muon)
        else:
            return get_templ(f,
                             region,
                             sample,
                             ptbin,
                             syst=syst,
                             read_sumw2=read_sumw2,
                             muon=muon)

    def shape_to_numX(f, region, sName, ptbin, syst, mask):
        if args.matched:
            return shape_to_numM(f, region, sName, ptbin, syst, mask)
        else:
            return shape_to_num(f, region, sName, ptbin, syst, mask)

    dummy_rhalphabet(pseudo=args.pseudo,
                     throwPoisson=args.throwPoisson,
                     MCTF=args.MCTF,
                     scale_syst=args.scale,
                     smear_syst=args.smear,
                     systs=args.systs,
                     justZ=args.justZ,
                     blind=(not args.unblind),
                     runhiggs=args.runhiggs,
                     runboth=args.runboth,
                     fitTF=args.fitTF,
                     year=str(args.year),
                     muonCR=args.muCR,
                     opts=args
                     )
