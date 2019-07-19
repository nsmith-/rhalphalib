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

        for sName in ['zqq', 'wqq']:
            templ = (np.random.exponential(5, size=5), bins, 'x')
            sample = rl.TemplateSample(ch.name + '_' + sName, rl.Sample.BACKGROUND, templ)

            jecup_ratio = np.array([0.02, 0.05, 0.1, 0.11, 0.2])
            sample.setParamEffect(jec, jecup_ratio)

            msdUp = np.linspace(0.9, 1.1, 5)
            msdDn = np.linspace(1.2, 0.8, 5)
            sample.setParamEffect(massScale, msdUp, msdDn)

            sample.setParamEffect(lumi, 1.027)

            ch.addSample(sample)

        # make up a data_obs
        templ = (np.random.poisson(5, size=5), bins, 'x')
        ch.addSample(rl.TemplateSample(ch.name + '_' + 'data_obs', rl.Sample.OBSERVATION, templ))

    model.renderWorkspace("simplemodel")

if __name__ == '__main__':
    test_simple()
