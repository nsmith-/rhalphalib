import rhalphalib as rl
import numpy as np


def dummy_rhalphabet():
    model = rl.Model("testModel")

    jec = rl.NuisanceParameter('CMS_jec', 'lnN')
    massScale = rl.NuisanceParameter('CMS_msdScale', 'shape')
    lumi = rl.NuisanceParameter('CMS_lumi', 'lnN')

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
            for sName in ['zqq', 'wqq', 'hqq']:
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

    import pickle
    with open("model.pkl", "wb") as fout:
        pickle.dump(model, fout)

    import sys
    print("ROOT used? ", 'ROOT' in sys.modules)
    model.renderCombine("testModel")
    print("ROOT used? ", 'ROOT' in sys.modules)


if __name__ == '__main__':
    dummy_rhalphabet()
