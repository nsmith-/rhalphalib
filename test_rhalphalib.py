import rhalphalib as rl
import numpy as np
import pickle


def dummy_rhalphabet():
    model = rl.Model("testModel")

    jec = rl.NuisanceParameter('CMS_jec', 'lnN')
    massScale = rl.NuisanceParameter('CMS_msdScale', 'shape')
    lumi = rl.NuisanceParameter('CMS_lumi', 'lnN')
    tqqeffSF = rl.IndependentParameter('tqqeffSF', 1.)
    tqqnormSF = rl.IndependentParameter('tqqnormSF', 1.)

    ptbins = np.array([450, 500, 550, 600, 675, 800, 1200])
    npt = len(ptbins) - 1
    msdbins = np.linspace(40, 201, 24)
    nmsd = len(msdbins) - 1

    tf = rl.BernsteinPoly("qcd_pass_rhalphTF", (2, 3), ['pt', 'rho'])
    # here we derive these all at once with 2D array
    ptpts, msdpts = np.meshgrid(ptbins[:-1] + 0.3 * np.diff(ptbins), msdbins[:-1] + 0.3 * np.diff(msdbins), indexing='ij')
    rhopts = 2*np.log(msdpts/ptpts)
    ptscaled = (ptpts - 450.) / (1200. - 450.)
    rhoscaled = (rhopts - (-6)) / ((-2.1) - (-6))
    validbins = (rhoscaled >= 0) & (rhoscaled <= 1)
    rhoscaled[~validbins] = 1  # we will mask these out later
    tf_params = tf(ptscaled, rhoscaled)

    for ptbin in range(npt):
        for region in ['pass', 'fail']:
            ch = rl.Channel("ptbin%d%s" % (ptbin, region))
            model.addChannel(ch)

            notqcdsum = np.zeros(nmsd)
            for sName in ['zqq', 'wqq', 'tqq', 'hqq']:
                # some mock expectations
                templ = (np.random.exponential(5, size=nmsd), msdbins, 'msd')
                notqcdsum += templ[0]
                stype = rl.Sample.SIGNAL if sName == 'hqq' else rl.Sample.BACKGROUND
                sample = rl.TemplateSample(ch.name + '_' + sName, stype, templ)

                # mock systematics
                jecup_ratio = np.random.normal(loc=1, scale=0.05, size=nmsd)
                msdUp = np.linspace(0.9, 1.1, nmsd)
                msdDn = np.linspace(1.2, 0.8, nmsd)

                # for jec we set lnN prior, shape will automatically be converted to norm systematic
                sample.setParamEffect(jec, jecup_ratio)
                sample.setParamEffect(massScale, msdUp, msdDn)
                sample.setParamEffect(lumi, 1.027)

                ch.addSample(sample)

            # make up a data_obs
            data_obs = (np.random.poisson(notqcdsum + 50), msdbins, 'msd')
            ch.setObservation(data_obs)

            # drop bins outside rho validity
            mask = validbins[ptbin]
            # blind bins 4,5
            mask[4:6] = False
            ch.mask = mask

        # steal observable definition from fail channel
        failCh = model['ptbin%dfail' % ptbin]
        obs = failCh.observable
        qcdparams = np.array([rl.IndependentParameter('qcdparam_ptbin%d_msdbin%d' % (ptbin, i), 0) for i in range(nmsd)])
        initial_qcd = failCh.getObservation().astype(float)  # was integer, and numpy complained about subtracting float from it
        for sample in failCh:
            initial_qcd -= sample.getExpectation(nominal=True)
        if np.any(initial_qcd < 0.):
            raise ValueError("uh-oh")
        sigmascale = 10  # to scale the deviation from initial
        scaledparams = initial_qcd + sigmascale*np.sqrt(initial_qcd)*qcdparams
        fail_qcd = rl.ParametericSample('ptbin%dfail_qcd' % ptbin, rl.Sample.BACKGROUND, obs, scaledparams)
        failCh.addSample(fail_qcd)
        pass_qcd = rl.TransferFactorSample('ptbin%dpass_qcd' % ptbin, rl.Sample.BACKGROUND, tf_params[ptbin, :], fail_qcd)
        model['ptbin%dpass' % ptbin].addSample(pass_qcd)

        tqqpass = model['ptbin%dpass_tqq' % ptbin]
        tqqfail = model['ptbin%dfail_tqq' % ptbin]
        tqqPF = tqqpass.getExpectation(nominal=True).sum() / tqqfail.getExpectation(nominal=True).sum()
        tqqpass.setParamEffect(tqqeffSF, 1*tqqeffSF)
        tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
        tqqpass.setParamEffect(tqqnormSF, 1*tqqnormSF)
        tqqfail.setParamEffect(tqqnormSF, 1*tqqnormSF)

        # Fill in muon CR

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

    zvvBinYields = np.array([rl.IndependentParameter('zvvYield_recoilbin%d' % i, 0) for i in range(recoil.nbins)])
    zvvJets = rl.ParametericSample('signalCh_zvvJets', rl.Sample.BACKGROUND, recoil, zvvBinYields)
    signalCh.addSample(zvvJets)

    signalCh.setObservation((np.random.poisson(1000*(20/6.6)*np.exp(-0.5*np.arange(recoil.nbins))), recoil.binning, recoil.name))

    zllCh = rl.Channel("zllCh")
    model.addChannel(zllCh)

    zllTemplate = (np.random.poisson(1000*np.exp(-0.5*np.arange(recoil.nbins))), recoil.binning, recoil.name)
    zllJetsMC = rl.TemplateSample('zllJetsMC', rl.Sample.BACKGROUND, zllTemplate)
    zllJetsMC.setParamEffect(jec, np.random.normal(loc=1, scale=0.05, size=recoil.nbins))
    zllJetsMC.setParamEffect(ele_id_eff, np.random.normal(loc=1, scale=0.02, size=recoil.nbins), np.random.normal(loc=1, scale=0.02, size=recoil.nbins))

    zvvTemplate = (np.random.poisson(1000*(20/6.6)*np.exp(-0.5*np.arange(recoil.nbins))), recoil.binning, recoil.name)
    zvvJetsMC = rl.TemplateSample('zvvJetsMC', rl.Sample.BACKGROUND, zvvTemplate)
    zvvJetsMC.setParamEffect(jec, np.random.normal(loc=1, scale=0.01, size=recoil.nbins))

    zllTransferFactor = zllJetsMC.getExpectation() / zvvJetsMC.getExpectation()
    zllJets = rl.TransferFactorSample('zllCh_zllJets', rl.Sample.BACKGROUND, zllTransferFactor, zvvJets)
    zllCh.addSample(zllJets)

    zllCh.setObservation((np.random.poisson(1000*np.exp(-0.5*np.arange(recoil.nbins))), recoil.binning, recoil.name))

    gammaCh = rl.Channel("gammaCh")
    model.addChannel(gammaCh)

    gammaTemplate = (np.random.poisson(4000*np.exp(-0.5*np.arange(recoil.nbins))), recoil.binning, recoil.name)
    gammaJetsMC = rl.TemplateSample('gammaJetsMC', rl.Sample.BACKGROUND, gammaTemplate)
    gammaJetsMC.setParamEffect(jec, np.random.normal(loc=1, scale=0.05, size=recoil.nbins))
    gammaJetsMC.setParamEffect(pho_id_eff, np.random.normal(loc=1, scale=0.02, size=recoil.nbins))

    gammaTransferFactor = gammaJetsMC.getExpectation() / zvvJetsMC.getExpectation()
    gammaJets = rl.TransferFactorSample('gammaCh_gammaJets', rl.Sample.BACKGROUND, gammaTransferFactor, zvvJets)
    gammaJets.setParamEffect(gamma_to_z_ewk, np.linspace(1.01, 1.05, recoil.nbins))
    gammaCh.addSample(gammaJets)

    gammaCh.setObservation((np.random.poisson(4000*np.exp(-0.5*np.arange(recoil.nbins))), recoil.binning, recoil.name))

    with open("monojetModel.pkl", "wb") as fout:
        pickle.dump(model, fout)

    model.renderCombine("monojetModel")


if __name__ == '__main__':
    dummy_rhalphabet()
    dummy_monojet()
