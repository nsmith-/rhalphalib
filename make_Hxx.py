from __future__ import print_function, division
import rhalphalib as rl
import numpy as np
import scipy.stats
import pickle
import ROOT
import uproot
rl.util.install_roofit_helpers()


def expo_sample(norm, scale, obs):
    cdf = scipy.stats.expon.cdf(scale=scale, x=obs.binning) * norm
    return (np.diff(cdf), obs.binning, obs.name)


def gaus_sample(norm, loc, scale, obs):
    cdf = scipy.stats.norm.cdf(loc=loc, scale=scale, x=obs.binning) * norm
    return (np.diff(cdf), obs.binning, obs.name)


def dummy_rhalphabet():
    throwPoisson = False

    jec = rl.NuisanceParameter('CMS_jec', 'lnN')
    massScale = rl.NuisanceParameter('CMS_msdScale', 'shape')
    lumi = rl.NuisanceParameter('CMS_lumi', 'lnN')
    tqqeffSF = rl.IndependentParameter('tqqeffSF', 1., 0, 10)
    tqqnormSF = rl.IndependentParameter('tqqnormSF', 1., 0, 10)

    JES = rl.NuisanceParameter('JES', 'lnN')
    #JER = rl.NuisanceParameter('JER', 'lnN')
    Pu = rl.NuisanceParameter('Pu', 'lnN')
    #trigger = rl.NuisanceParameter('trigger', 'lnN')

    ddxeff = rl.NuisanceParameter('DDXeff', 'lnN')
    eleveto = rl.NuisanceParameter('eleveto', 'lnN')
    muveto = rl.NuisanceParameter('muveto', 'lnN')
    #h_pt = rl.NuisanceParameter('h_pt', 'lnN')
    #h_pt_shape = rl.NuisanceParameter('h_pt_shape', 'shape')

    veff = rl.NuisanceParameter('veff', 'lnN')
    wznormEW = rl.NuisanceParameter('wznormEW', 'lnN')
    znormEW = rl.NuisanceParameter('znormEW', 'lnN')
    znormQ = rl.NuisanceParameter('znormQ', 'lnN')

    scale = rl.NuisanceParameter('scale', 'shape')
    smear = rl.NuisanceParameter('smear', 'shape')

    ptbins = np.array([450, 500, 550, 600, 675, 800, 1200])
    npt = len(ptbins) - 1
    msdbins = np.linspace(40, 201, 24)
    msd = rl.Observable('msd', msdbins)

    # here we derive these all at once with 2D array
    ptpts, msdpts = np.meshgrid(ptbins[:-1] + 0.3 * np.diff(ptbins), msdbins[:-1] + 0.5 * np.diff(msdbins), indexing='ij')
    rhopts = 2*np.log(msdpts/ptpts)
    ptscaled = (ptpts - 450.) / (1200. - 450.)
    rhoscaled = (rhopts - (-6)) / ((-2.1) - (-6))
    validbins = (rhoscaled >= 0) & (rhoscaled <= 1)
    rhoscaled[~validbins] = 1  # we will mask these out later

    # Build qcd MC pass+fail model and fit to polynomial
    qcdmodel = rl.Model("qcdmodel")
    qcdpass, qcdfail = 0., 0.

    def get_templ(region, sample, ptbin):
        import uproot
        f = uproot.open('hxx/hist_1DZcc_pt_scalesmear.root')
        hist_name = '{}_{}'.format(sample, region)
        h_vals = f[hist_name].values[:, ptbin]
        h_edges = f[hist_name].edges[0]
        h_key = 'msd'
        return (h_vals, h_edges, h_key)

    for ptbin in range(npt):
        failCh = rl.Channel("ptbin%d%s" % (ptbin, 'fail'))
        passCh = rl.Channel("ptbin%d%s" % (ptbin, 'pass'))
        qcdmodel.addChannel(failCh)
        qcdmodel.addChannel(passCh)

        passTempl = get_templ("pass", "qcd", ptbin)
        failTempl = get_templ("fail", "qcd", ptbin)

        # mock template
        #ptnorm = 1
        #failTempl = expo_sample(norm=ptnorm*1e5, scale=40, obs=msd)
        #print(failTempl)
        #passTempl = expo_sample(norm=ptnorm*1e3, scale=40, obs=msd)
        #import sys
        #sys.exit()
        failCh.setObservation(failTempl)
        passCh.setObservation(passTempl)
        qcdfail += failCh.getObservation().sum()
        qcdpass += passCh.getObservation().sum()

    qcdeff = qcdpass / qcdfail
    tf_MCtempl = rl.BernsteinPoly("tf_MCtempl", (2, 2), ['pt', 'rho'], limits=(0, 10))
    tf_MCtempl_params = qcdeff * tf_MCtempl(ptscaled, rhoscaled)
    for ptbin in range(npt):
        failCh = qcdmodel['ptbin%dfail' % ptbin]
        passCh = qcdmodel['ptbin%dpass' % ptbin]
        failObs = failCh.getObservation()
        qcdparams = np.array([rl.IndependentParameter('qcdparam_ptbin%d_msdbin%d' % (ptbin, i), 0) for i in range(msd.nbins)])
        sigmascale = 10.
        scaledparams = failObs * (1 + sigmascale/np.maximum(1., np.sqrt(failObs)))**qcdparams
        fail_qcd = rl.ParametericSample('ptbin%dfail_qcd' % ptbin, rl.Sample.BACKGROUND, msd, scaledparams)
        failCh.addSample(fail_qcd)
        pass_qcd = rl.TransferFactorSample('ptbin%dpass_qcd' % ptbin, rl.Sample.BACKGROUND, tf_MCtempl_params[ptbin, :], fail_qcd)
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
                          ROOT.RooFit.PrintLevel(-1),
                          )
    qcdfit_ws.add(qcdfit)
    qcdfit_ws.writeToFile('qcdfit.root')
    if qcdfit.status() != 0:
        raise RuntimeError('Could not fit qcd')

    param_names = [p.name for p in tf_MCtempl.parameters.reshape(-1)]
    decoVector = rl.DecorrelatedNuisanceVector.fromRooFitResult(tf_MCtempl.name + '_deco', qcdfit, param_names)
    tf_MCtempl.parameters = decoVector.correlated_params.reshape(tf_MCtempl.parameters.shape)
    tf_MCtempl_params_final = tf_MCtempl(ptscaled, rhoscaled)
    tf_dataResidual = rl.BernsteinPoly("tf_dataResidual", (2, 2), ['pt', 'rho'], limits=(0, 10))
    tf_dataResidual_params = tf_dataResidual(ptscaled, rhoscaled)
    tf_params = qcdeff * tf_MCtempl_params_final * tf_dataResidual_params

    # build actual fit model now
    model = rl.Model("hxxModel")

    # Load templates
    import uproot
    #f = uproot.open('~/nobackup/coffeandbacon/analysis/hist_1DZcc_pt_scalesmear.root')
    f = uproot.open('hxx/hist_1DZcc_pt_scalesmear.root')
    templates = {}
    #sys_list = ['JES', 'JER', 'trigger', 'Pu']  # Sys list
    sys_list = ['JES']  # Sys list
    sysud_list = sum([[sys+"Up", sys+"Down"] for sys in sys_list], [])  # Sys name list
    for ptbin in range(npt):
        for region in ['pass', 'fail']:
            templates[region+str(ptbin)] = {}
            for sample in ['zbb', 'zcc', 'zqq', 'wcq', 'wqq', 'hcc125', 'tqq', 'hqq125', 'qcd']:
                for sys_name in [""] + sysud_list:
                    hist_name = '{}_{}'.format(sample, region)
                    _outname = sample.replace('125', '')
                    if len(sys_name) > 0:
                        hist_name += "_"+sys_name
                        _outname += "_"+sys_name
                    h_vals = f[hist_name].values[:, ptbin]
                    h_edges = f[hist_name].edges[0]
                    h_key = 'msd'
                    templates[region+str(ptbin)][_outname] = (h_vals, h_edges, h_key)

    for ptbin in range(npt):
        for region in ['pass', 'fail']:
            ch = rl.Channel("ptbin%d%s" % (ptbin, region))
            model.addChannel(ch)
            ptnorm = 1.

            SF2017 = {
             # cristina July8
             # 'shift_SF'  : 0.978,           'shift_SF_ERR' : 0.012,
             # 'smear_SF'  : 0.9045,          'smear_SF_ERR' : 0.048,
             # 'V_SF'      : 0.924,           'V_SF_ERR'  : 0.018,
             # cristina Jun25
             'shift_SF': 0.979,       'shift_SF_ERR': 0.012,
             'smear_SF': 1.037,       'smear_SF_ERR': 0.049,
             'V_SF': 0.92,            'V_SF_ERR': 0.018,
             'BB_SF': 1.0,            'BB_SF_ERR': 0.3
            }

            def GetSF(sample, region, templates, sf_dict=SF2017):
                SF = 1
                _isScalar = sample.startswith('z') or sample.startswith('h')
                _isBoson = sample.startswith('z') or sample.startswith('h') or sample.startswith('w')
                if _isScalar:
                    SF *= 1.
                    # if region == 'pass':
                    #     SF *= sf_dict['BB_SF']
                    # else:
                    #     templates[region][sName]
                    #     passInt = f.Get(sample + '_pass').Integral()
                    #     failInt = f.Get(sample + '_fail').Integral()
                    #     if failInt > 0:
                    #         SF *= (1. + (1. - sf_dict['BB_SF']) * passInt / failInt)

                #if _isBoson:
                #    SF *= sf_dict['V_SF']
                #print(SF)
                return SF

            for sName in ['zbb', 'zcc', 'zqq', 'wcq', 'wqq', 'hcc', 'tqq', 'hqq']:
                templ = templates[region+str(ptbin)][sName]
                stype = rl.Sample.SIGNAL if sName in ['zcc', 'hcc'] else rl.Sample.BACKGROUND
                # stype = rl.Sample.SIGNAL if sName in ['hcc'] else rl.Sample.BACKGROUND
                sample = rl.TemplateSample(ch.name + '_' + sName, stype, templ)

                # mock systematics
                # jecup_ratio = np.random.normal(loc=1, scale=0.05, size=msd.nbins)
                # msdUp = np.linspace(0.9, 1.1, msd.nbins)
                # msdDn = np.linspace(1.2, 0.8, msd.nbins)

                # for jec we set lnN prior, shape will automatically be converted to norm systematic
                # sample.setParamEffect(jec, jecup_ratio)
                # sample.setParamEffect(massScale, msdUp, msdDn)
                # sample.setParamEffect(lumi, 1.027)

                # Shape systematics general
                #sys_list = ['JES', 'JER', 'trigger', 'Pu']
                sys_list = ['JES']
                #for sys_name, sys in zip(sys_list, [JES, JER, trigger, Pu]):
                for sys_name, sys in zip(sys_list, [JES]):
                    _up = templates[region+str(ptbin)][sName+"_"+sys_name+"Up"]
                    _dn = templates[region+str(ptbin)][sName+"_"+sys_name+"Down"]
                    _sf = GetSF(sName, region+str(ptbin), templates)
                    sample.setParamEffect(sys, _up[0]*_sf, _dn[0]*_sf)

                # Systematics by group
                # if sName.startswith("h"):
                    # sample.setParamEffect(h_pt, 1.3)
                    # sample.setParamEffect(h_pt_shape, 1.0)
                if sName not in ["qcd"]:
                    sample.setParamEffect(eleveto, 1.005)
                    sample.setParamEffect(muveto, 1.005)
                    #sample.setParamEffect(trigger, 1.02)
                if sName not in ["qcd", 'tqq']:
                    sample.setParamEffect(lumi, 1.025)
                    sample.setParamEffect(veff, 1.043)
                if sName.startswith("z"):
                    sample.setParamEffect(znormQ, 1.1)
                    sample.setParamEffect(znormEW, 1.35)
                if sName.startswith("w"):
                    sample.setParamEffect(wznormEW, 1.15)
                    sample.setParamEffect(znormEW, 1.35)

                # Missing
                # scale, smear

                ch.addSample(sample)

            yields = sum(tpl[0] for tpl in templates[region+str(ptbin)].values())
            if throwPoisson:
                yields = np.random.poisson(yields)
            data_obs = (yields, msd.binning, msd.name)
            ch.setObservation(data_obs)

            # drop bins outside rho validity
            mask = validbins[ptbin]
            # blind bins 11, 12, 13
            # mask[11:14] = False
            ch.mask = mask

    for ptbin in range(npt):
        failCh = model['ptbin%dfail' % ptbin]
        passCh = model['ptbin%dpass' % ptbin]

        qcdparams = np.array([rl.IndependentParameter('qcdparam_ptbin%d_msdbin%d' % (ptbin, i), 0) for i in range(msd.nbins)])
        initial_qcd = failCh.getObservation().astype(float)
        # was integer, and numpy complained about subtracting float from it
        for sample in failCh:
            initial_qcd -= sample.getExpectation(nominal=True)
        if np.any(initial_qcd < 0.):
            raise ValueError("initial_qcd negative for some bins..", initial_qcd)
        sigmascale = 10  # to scale the deviation from initial
        scaledparams = initial_qcd * (1 + sigmascale/np.maximum(1., np.sqrt(initial_qcd)))**qcdparams
        fail_qcd = rl.ParametericSample('ptbin%dfail_qcd' % ptbin, rl.Sample.BACKGROUND, msd, scaledparams)
        failCh.addSample(fail_qcd)
        pass_qcd = rl.TransferFactorSample('ptbin%dpass_qcd' % ptbin, rl.Sample.BACKGROUND, tf_params[ptbin, :], fail_qcd)
        passCh.addSample(pass_qcd)

        tqqpass = passCh['tqq']
        tqqfail = failCh['tqq']
        tqqPF = tqqpass.getExpectation(nominal=True).sum() \
            / tqqfail.getExpectation(nominal=True).sum()
        tqqpass.setParamEffect(tqqeffSF, 1*tqqeffSF)
        tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
        tqqpass.setParamEffect(tqqnormSF, 1*tqqnormSF)
        tqqfail.setParamEffect(tqqnormSF, 1*tqqnormSF)

    # Fill in muon CR
    templates = {}
    for ptbin in range(npt):
        for region in ['pass', 'fail']:
            templates[region+str(ptbin)] = {}

    for region in ['pass', 'fail']:
        ch = rl.Channel("muonCR%s" % (region, ))
        model.addChannel(ch)

        templates = {}
        f = uproot.open(
            '~/nobackup/coffeandbacon/analysis/hist_1DZcc_pt_scalesmear.root')
        for sample in ['tqq', 'qcd']:
            hist_name = '{}_{}'.format(sample, region)
            h_vals = f[hist_name].values[:, ptbin]
            h_edges = f[hist_name].edges[0]
            h_key = 'msd'
            templates[sample.replace('125', '')] = (h_vals, h_edges, h_key)

        for sName, templ in templates.items():
            stype = rl.Sample.BACKGROUND
            sample = rl.TemplateSample(ch.name + '_' + sName, stype, templ)

            # mock systematics
            jecup_ratio = np.random.normal(loc=1, scale=0.05, size=msd.nbins)
            sample.setParamEffect(jec, jecup_ratio)

            ch.addSample(sample)

        yields = sum(tpl[0] for tpl in templates.values())
        if throwPoisson:
            yields = np.random.poisson(yields)
        data_obs = (yields, msd.binning, msd.name)
        ch.setObservation(data_obs)

    tqqpass = model['muonCRpass_tqq']
    tqqfail = model['muonCRfail_tqq']
    tqqPF = tqqpass.getExpectation(nominal=True).sum() / tqqfail.getExpectation(nominal=True).sum()
    tqqpass.setParamEffect(tqqeffSF, 1*tqqeffSF)
    tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
    tqqpass.setParamEffect(tqqnormSF, 1*tqqnormSF)
    tqqfail.setParamEffect(tqqnormSF, 1*tqqnormSF)

    with open("hxxModel.pkl", "wb") as fout:
        pickle.dump(model, fout)

    model.renderCombine("hxxModel")

if __name__ == '__main__':
    dummy_rhalphabet()
