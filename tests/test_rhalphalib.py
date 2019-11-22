from __future__ import print_function, division
import sys
import os
import rhalphalib as rl
import numpy as np
import scipy.stats
import pickle
import ROOT
rl.util.install_roofit_helpers()
rl.ParametericSample.PreferRooParametricHist = False


def expo_sample(norm, scale, obs):
    cdf = scipy.stats.expon.cdf(scale=scale, x=obs.binning) * norm
    return (np.diff(cdf), obs.binning, obs.name)


def gaus_sample(norm, loc, scale, obs):
    cdf = scipy.stats.norm.cdf(loc=loc, scale=scale, x=obs.binning) * norm
    return (np.diff(cdf), obs.binning, obs.name)


def test_rhalphabet(tmpdir):
    throwPoisson = False

    jec = rl.NuisanceParameter('CMS_jec', 'lnN')
    massScale = rl.NuisanceParameter('CMS_msdScale', 'shape')
    lumi = rl.NuisanceParameter('CMS_lumi', 'lnN')
    tqqeffSF = rl.IndependentParameter('tqqeffSF', 1., 0, 10)
    tqqnormSF = rl.IndependentParameter('tqqnormSF', 1., 0, 10)

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
    for ptbin in range(npt):
        failCh = rl.Channel("ptbin%d%s" % (ptbin, 'fail'))
        passCh = rl.Channel("ptbin%d%s" % (ptbin, 'pass'))
        qcdmodel.addChannel(failCh)
        qcdmodel.addChannel(passCh)
        # mock template
        ptnorm = 1
        failTempl = expo_sample(norm=ptnorm*1e5, scale=40, obs=msd)
        passTempl = expo_sample(norm=ptnorm*1e3, scale=40, obs=msd)
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
    if "pytest" not in sys.modules:
         qcdfit_ws.writeToFile(os.path.join(str(tmpdir), 'testModel_qcdfit.root'))
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
    model = rl.Model("testModel")

    for ptbin in range(npt):
        for region in ['pass', 'fail']:
            ch = rl.Channel("ptbin%d%s" % (ptbin, region))
            model.addChannel(ch)

            isPass = region == 'pass'
            ptnorm = 1.
            templates = {
                'wqq': gaus_sample(norm=ptnorm*(100 if isPass else 300), loc=80, scale=8, obs=msd),
                'zqq': gaus_sample(norm=ptnorm*(200 if isPass else 100), loc=91, scale=8, obs=msd),
                'tqq': gaus_sample(norm=ptnorm*(40 if isPass else 80), loc=150, scale=20, obs=msd),
                'hqq': gaus_sample(norm=ptnorm*(20 if isPass else 5), loc=125, scale=8, obs=msd),
            }
            for sName in ['zqq', 'wqq', 'tqq', 'hqq']:
                # some mock expectations
                templ = templates[sName]
                stype = rl.Sample.SIGNAL if sName == 'hqq' else rl.Sample.BACKGROUND
                sample = rl.TemplateSample(ch.name + '_' + sName, stype, templ)

                # mock systematics
                jecup_ratio = np.random.normal(loc=1, scale=0.05, size=msd.nbins)
                msdUp = np.linspace(0.9, 1.1, msd.nbins)
                msdDn = np.linspace(1.2, 0.8, msd.nbins)

                # for jec we set lnN prior, shape will automatically be converted to norm systematic
                sample.setParamEffect(jec, jecup_ratio)
                sample.setParamEffect(massScale, msdUp, msdDn)
                sample.setParamEffect(lumi, 1.027)

                ch.addSample(sample)

            # make up a data_obs, with possibly different yield values
            templates = {
                'wqq': gaus_sample(norm=ptnorm*(100 if isPass else 300), loc=80, scale=8, obs=msd),
                'zqq': gaus_sample(norm=ptnorm*(200 if isPass else 100), loc=91, scale=8, obs=msd),
                'tqq': gaus_sample(norm=ptnorm*(40 if isPass else 80), loc=150, scale=20, obs=msd),
                'hqq': gaus_sample(norm=ptnorm*(20 if isPass else 5), loc=125, scale=8, obs=msd),
                'qcd': expo_sample(norm=ptnorm*(1e3 if isPass else 1e5), scale=40, obs=msd),
            }
            yields = sum(tpl[0] for tpl in templates.values())
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
        initial_qcd = failCh.getObservation().astype(float)  # was integer, and numpy complained about subtracting float from it
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
        tqqPF = tqqpass.getExpectation(nominal=True).sum() / tqqfail.getExpectation(nominal=True).sum()
        tqqpass.setParamEffect(tqqeffSF, 1*tqqeffSF)
        tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
        tqqpass.setParamEffect(tqqnormSF, 1*tqqnormSF)
        tqqfail.setParamEffect(tqqnormSF, 1*tqqnormSF)

    # Fill in muon CR
    for region in ['pass', 'fail']:
        ch = rl.Channel("muonCR%s" % (region, ))
        model.addChannel(ch)

        isPass = region == 'pass'
        templates = {
            'tqq': gaus_sample(norm=10*(30 if isPass else 60), loc=150, scale=20, obs=msd),
            'qcd': expo_sample(norm=10*(5e2 if isPass else 1e3), scale=40, obs=msd),
        }
        for sName, templ in templates.items():
            stype = rl.Sample.BACKGROUND
            sample = rl.TemplateSample(ch.name + '_' + sName, stype, templ)

            # mock systematics
            jecup_ratio = np.random.normal(loc=1, scale=0.05, size=msd.nbins)
            sample.setParamEffect(jec, jecup_ratio)

            ch.addSample(sample)

        # make up a data_obs
        templates = {
            'tqq': gaus_sample(norm=10*(30 if isPass else 60), loc=150, scale=20, obs=msd),
            'qcd': expo_sample(norm=10*(5e2 if isPass else 1e3), scale=40, obs=msd),
        }
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

    with open(os.path.join(str(tmpdir), 'testModel.pkl'), "wb") as fout:
        pickle.dump(model, fout)

    model.renderCombine(os.path.join(str(tmpdir), 'testModel'))


def test_monojet(tmpdir):
    model = rl.Model("testMonojet")

    # lumi = rl.NuisanceParameter('CMS_lumi', 'lnN')
    jec = rl.NuisanceParameter('CMS_jec', 'shape')
    ele_id_eff = rl.NuisanceParameter('CMS_ele_id_eff', 'shape')
    pho_id_eff = rl.NuisanceParameter('CMS_pho_id_eff', 'shape')
    gamma_to_z_ewk = rl.NuisanceParameter('Theory_gamma_z_ewk', 'shape')

    recoilbins = np.linspace(300, 1200, 13)
    recoil = rl.Observable('recoil', recoilbins)

    signalCh = rl.Channel("signalCh")
    model.addChannel(signalCh)

    zvvTemplate = expo_sample(1000, 400, recoil)
    zvvJetsMC = rl.TemplateSample('zvvJetsMC', rl.Sample.BACKGROUND, zvvTemplate)
    zvvJetsMC.setParamEffect(jec, np.random.normal(loc=1, scale=0.01, size=recoil.nbins))

    # these parameters are large, should probably log-transform them
    zvvBinYields = np.array([rl.IndependentParameter('tmp', b, 0, zvvTemplate[0].max()*2) for b in zvvTemplate[0]])  # name will be changed by ParametericSample
    zvvJets = rl.ParametericSample('signalCh_zvvJets', rl.Sample.BACKGROUND, recoil, zvvBinYields)
    signalCh.addSample(zvvJets)

    dmTemplate = expo_sample(100, 800, recoil)
    dmSample = rl.TemplateSample('signalCh_someDarkMatter', rl.Sample.SIGNAL, dmTemplate)
    signalCh.addSample(dmSample)

    signalCh.setObservation(expo_sample(1000, 400, recoil))

    zllCh = rl.Channel("zllCh")
    model.addChannel(zllCh)

    zllTemplate = expo_sample(1000*6.6/20, 400, recoil)
    zllJetsMC = rl.TemplateSample('zllJetsMC', rl.Sample.BACKGROUND, zllTemplate)
    zllJetsMC.setParamEffect(jec, np.random.normal(loc=1, scale=0.05, size=recoil.nbins))
    zllJetsMC.setParamEffect(ele_id_eff, np.random.normal(loc=1, scale=0.02, size=recoil.nbins), np.random.normal(loc=1, scale=0.02, size=recoil.nbins))

    zllTransferFactor = zllJetsMC.getExpectation() / zvvJetsMC.getExpectation()
    zllJets = rl.TransferFactorSample('zllCh_zllJets', rl.Sample.BACKGROUND, zllTransferFactor, zvvJets)
    zllCh.addSample(zllJets)

    otherbkgTemplate = expo_sample(200, 250, recoil)
    otherbkg = rl.TemplateSample('zllCh_otherbkg', rl.Sample.BACKGROUND, otherbkgTemplate)
    otherbkg.setParamEffect(jec, np.random.normal(loc=1, scale=0.01, size=recoil.nbins))
    zllCh.addSample(otherbkg)

    zllCh.setObservation(expo_sample(1200, 380, recoil))

    gammaCh = rl.Channel("gammaCh")
    model.addChannel(gammaCh)

    gammaTemplate = expo_sample(2000, 450, recoil)
    gammaJetsMC = rl.TemplateSample('gammaJetsMC', rl.Sample.BACKGROUND, gammaTemplate)
    gammaJetsMC.setParamEffect(jec, np.random.normal(loc=1, scale=0.05, size=recoil.nbins))
    gammaJetsMC.setParamEffect(pho_id_eff, np.random.normal(loc=1, scale=0.02, size=recoil.nbins))

    gammaTransferFactor = gammaJetsMC.getExpectation() / zvvJetsMC.getExpectation()
    gammaJets = rl.TransferFactorSample('gammaCh_gammaJets', rl.Sample.BACKGROUND, gammaTransferFactor, zvvJets)
    gammaJets.setParamEffect(gamma_to_z_ewk, np.linspace(1.01, 1.05, recoil.nbins))
    gammaCh.addSample(gammaJets)

    gammaCh.setObservation(expo_sample(2000, 450, recoil))

    with open(os.path.join(str(tmpdir), 'monojetModel.pkl'), "wb") as fout:
        pickle.dump(model, fout)

    model.renderCombine(os.path.join(str(tmpdir), 'monojetModel'))


if __name__ == '__main__':
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    test_rhalphabet('tmp')
    test_monojet('tmp')
