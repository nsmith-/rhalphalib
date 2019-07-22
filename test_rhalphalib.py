import rhalphalib as rl
import numpy as np


def test_simple():
    model = rl.Model("testModel")

    jec = rl.NuisanceParameter('CMS_jec', 'shape')
    massScale = rl.NuisanceParameter('CMS_msdScale', 'shape')
    lumi = rl.NuisanceParameter('CMS_lumi', 'lnN')

    for chIdx in ['0', '1']:
        ch = rl.Channel("channel" + chIdx)
        model.addChannel(ch)
        bins = np.linspace(0, 100, 6)

        bkgsum = np.zeros(5)
        for sName in ['zqq', 'wqq']:
            templ = (np.random.exponential(5, size=5), bins, 'x')
            bkgsum += templ[0]
            sample = rl.TemplateSample(ch.name + '_' + sName, rl.Sample.BACKGROUND, templ)

            jecup_ratio = np.array([0.02, 0.05, 0.1, 0.11, 0.2])
            sample.setParamEffect(jec, jecup_ratio)

            msdUp = np.linspace(0.9, 1.1, 5)
            msdDn = np.linspace(1.2, 0.8, 5)
            sample.setParamEffect(massScale, msdUp, msdDn)

            sample.setParamEffect(lumi, 1.027)

            ch.addSample(sample)

        # make up a data_obs
        data_obs = (np.random.poisson(bkgsum + 50), bins, 'x')
        ch.setObservation(data_obs)

    # steal observable definition from previous template
    obs = model['channel0_wqq'].observable

    qcdparams = [rl.IndependentParameter('qcdparam_bin%d' % i, 0) for i in range(5)]
    initial_qcd = model['channel0'].getObservation().astype(float)  # was integer, and numpy complained about subtracting float from it
    for p in model['channel0']:
        initial_qcd -= p.getExpectation(nominal=True)
    if np.any(initial_qcd < 0.):
        raise ValueError("uh-oh")
    sigmascale = 10  # to scale the deviation from initial
    scaledparams = initial_qcd + sigmascale*np.sqrt(initial_qcd)*qcdparams
    fail_sample = rl.ParametericSample('channel0_qcd', rl.Sample.BACKGROUND, obs, scaledparams)
    model['channel0'].addSample(fail_sample)

    tf = rl.BernsteinPoly("qcd_pass_rhalphTF", (2, 3), ['pt', 'rho'])
    ptval = 0.1
    rhovals = np.array([0.1, 0.2, 0.4, 0.5, 0.8])
    tf_params = np.array([tf(ptval, r) for r in rhovals])
    pass_sample = rl.TransferFactorSample('channel1_qcd', rl.Sample.BACKGROUND, tf_params, fail_sample)
    model['channel1'].addSample(pass_sample)

    # import ROOT
    # ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
    model.renderCombine("simplemodel")


if __name__ == '__main__':
    test_simple()
