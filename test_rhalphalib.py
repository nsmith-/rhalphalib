import rhalphalib as rl
import numpy as np


def test_simple():
    model = rl.Model("testModel")

    jec = rl.NuisanceParameter('CMS_jec', 'shape')
    massScale = rl.NuisanceParameter('CMS_msdScale', 'shape')
    lumi = rl.NuisanceParameter('CMS_lumi', 'lnN')
    bins = np.linspace(40, 201, 24)[:6]
    nbins = len(bins) - 1

    for chName in ['pt450to500Fail', 'pt450to500Pass']:
        ch = rl.Channel(chName)
        model.addChannel(ch)

        notqcdsum = np.zeros(nbins)
        for sName in ['zqq', 'wqq', 'hqq']:
            templ = (np.random.exponential(5, size=nbins), bins, 'x')
            notqcdsum += templ[0]
            stype = rl.Sample.SIGNAL if sName == 'hqq' else rl.Sample.BACKGROUND
            sample = rl.TemplateSample(ch.name + '_' + sName, stype, templ)

            jecup_ratio = np.random.normal(loc=1, scale=0.05, size=nbins)
            sample.setParamEffect(jec, jecup_ratio)

            msdUp = np.linspace(0.9, 1.1, nbins)
            msdDn = np.linspace(1.2, 0.8, nbins)
            sample.setParamEffect(massScale, msdUp, msdDn)

            sample.setParamEffect(lumi, 1.027)

            ch.addSample(sample)

        # make up a data_obs
        data_obs = (np.random.poisson(notqcdsum + 50), bins, 'x')
        ch.setObservation(data_obs)

    # steal observable definition from previous template
    obs = model['pt450to500Fail_wqq'].observable

    qcdparams = [rl.IndependentParameter('qcdparam_bin%d' % i, 0) for i in range(nbins)]
    initial_qcd = model['pt450to500Fail'].getObservation().astype(float)  # was integer, and numpy complained about subtracting float from it
    for p in model['pt450to500Fail']:
        initial_qcd -= p.getExpectation(nominal=True)
    if np.any(initial_qcd < 0.):
        raise ValueError("uh-oh")
    sigmascale = 10  # to scale the deviation from initial
    scaledparams = initial_qcd + sigmascale*np.sqrt(initial_qcd)*qcdparams
    fail_sample = rl.ParametericSample('pt450to500Fail_qcd', rl.Sample.BACKGROUND, obs, scaledparams)
    model['pt450to500Fail'].addSample(fail_sample)

    tf = rl.BernsteinPoly("qcd_pass_rhalphTF", (2, 3), ['pt', 'rho'])
    # suppose the scaled sampling point is 0.02 and the original is 465 (first pt bin)
    ptval = 0.02
    # suppose 'bins' is the msd binning, here we compute rho = 2*ln(msd/pt) using the msd value 0.3 of the way into the bin
    msdpts = bins[:-1] + 0.3 * np.diff(bins)
    rhovals = 2*np.log(msdpts/465.)
    # here we would derive these all at once with 2D array, and thus the bounds would envelope the whole space
    rhovals = (rhovals - rhovals.min()) / np.ptp(rhovals)
    tf_params = np.array([tf(ptval, r) for r in rhovals])
    pass_sample = rl.TransferFactorSample('pt450to500Pass_qcd', rl.Sample.BACKGROUND, tf_params, fail_sample)
    model['pt450to500Pass'].addSample(pass_sample)

    # import ROOT
    # ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
    model.renderCombine("simplemodel")


if __name__ == '__main__':
    test_simple()
