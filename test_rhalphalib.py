from __future__ import print_function, division
import rhalphalib as rl
import numpy as np
import scipy.stats
import pickle


def expo_sample(norm, scale, obs):
    cdf = scipy.stats.expon.cdf(scale=scale, x=obs.binning) * norm
    return (np.diff(cdf), obs.binning, obs.name)


def gaus_sample(norm, loc, scale, obs):
    cdf = scipy.stats.norm.cdf(loc=loc, scale=scale, x=obs.binning) * norm
    return (np.diff(cdf), obs.binning, obs.name)


def dummy_rhalphabet():
    model = rl.Model("testModel")

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
    ptpts, msdpts = np.meshgrid(ptbins[:-1] + 0.3 * np.diff(ptbins), msdbins[:-1] + 0.3 * np.diff(msdbins), indexing='ij')
    rhopts = 2*np.log(msdpts/ptpts)
    ptscaled = (ptpts - 450.) / (1200. - 450.)
    rhoscaled = (rhopts - (-6)) / ((-2.1) - (-6))
    validbins = (rhoscaled >= 0) & (rhoscaled <= 1)
    rhoscaled[~validbins] = 1  # we will mask these out later

    order = (2, 2)
    initial = np.full((order[0] + 1, order[1] + 1), 1.)
    tf = rl.BernsteinPoly("qcd_pass_rhalphTF", order, ['pt', 'rho'], initial, limits=(0, 10))
    tf_params = tf(ptscaled, rhoscaled)

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
                'qcd': expo_sample(norm=ptnorm*(1e3 if isPass else 1e4), scale=40, obs=msd),
            }
            notqcdsum = np.zeros(msd.nbins)
            for sName in ['zqq', 'wqq', 'tqq', 'hqq']:
                # some mock expectations
                templ = templates[sName]
                notqcdsum += templ[0]
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

            # make up a data_obs
            data_obs = (templates['qcd'][0] + notqcdsum, msd.binning, msd.name)
            ch.setObservation(data_obs)

            # drop bins outside rho validity
            mask = validbins[ptbin]
            # blind bins 11, 12, 13
            # mask[11:14] = False
            ch.mask = mask

    # compute overall qcd efficiency (to scale transfer factor close to 1)
    qcdpass, qcdfail = 0., 0.
    for ptbin in range(npt):
        failCh = model['ptbin%dfail' % ptbin]
        passCh = model['ptbin%dpass' % ptbin]
        qcdfail += failCh.getObservation().sum()
        for sample in failCh:
            qcdfail -= sample.getExpectation(nominal=True).sum()
        qcdpass += passCh.getObservation().sum()
        for sample in passCh:
            qcdpass -= sample.getExpectation(nominal=True).sum()

    qcdeff = qcdpass / qcdfail
    tf_params = tf_params * qcdeff

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
        templSum = np.zeros(msd.nbins)
        for sName, templ in templates.items():
            templSum += templ[0]
            stype = rl.Sample.BACKGROUND
            sample = rl.TemplateSample(ch.name + '_' + sName, stype, templ)

            # mock systematics
            jecup_ratio = np.random.normal(loc=1, scale=0.05, size=msd.nbins)
            sample.setParamEffect(jec, jecup_ratio)

            ch.addSample(sample)

        # make up a data_obs
        data_obs = (templSum, msd.binning, msd.name)
        ch.setObservation(data_obs)

    tqqpass = model['muonCRpass_tqq']
    tqqfail = model['muonCRfail_tqq']
    tqqPF = tqqpass.getExpectation(nominal=True).sum() / tqqfail.getExpectation(nominal=True).sum()
    tqqpass.setParamEffect(tqqeffSF, 1*tqqeffSF)
    tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
    tqqpass.setParamEffect(tqqnormSF, 1*tqqnormSF)
    tqqfail.setParamEffect(tqqnormSF, 1*tqqnormSF)

    with open("testModel.pkl", "wb") as fout:
        pickle.dump(model, fout)

    model.renderCombine("testModel")


def dummy_monojet():
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

    with open("monojetModel.pkl", "wb") as fout:
        pickle.dump(model, fout)

    model.renderCombine("monojetModel")


if __name__ == '__main__':
    dummy_rhalphabet()
    dummy_monojet()
