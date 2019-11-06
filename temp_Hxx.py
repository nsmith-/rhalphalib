from __future__ import print_function, division
import rhalphalib as rl
import numpy as np
import scipy.stats
import pickle
import ROOT
import uproot
rl.util.install_roofit_helpers()

def dummy_rhalphabet():
    throwPoisson = True
    fitTF = True
    pseudo = True

    # Default lumi (needs at least one systematics for prefit)
    lumi = rl.NuisanceParameter('CMS_lumi', 'lnN')

    # Define Bins
    ptbins = np.array([450, 500, 550, 600, 675, 800, 1200])
    npt = len(ptbins) - 1
    msdbins = np.linspace(40, 201, 24)
    msd = rl.Observable('msd', msdbins)

    # Define pt/msd/rho grids
    ptpts, msdpts = np.meshgrid(ptbins[:-1] + 0.3 * np.diff(ptbins), msdbins[:-1] + 0.5 * np.diff(msdbins), indexing='ij')
    rhopts = 2*np.log(msdpts/ptpts)
    ptscaled = (ptpts - 450.) / (1200. - 450.)
    rhoscaled = (rhopts - (-6)) / ((-2.1) - (-6))
    validbins = (rhoscaled >= 0) & (rhoscaled <= 1)
    rhoscaled[~validbins] = 1  # we will mask these out later

    # Template reading
    def get_templ(region, sample, ptbin):
        import uproot
        f = uproot.open('hxx/hist_1DZcc_pt_scalesmear.root')
        hist_name = '{}_{}'.format(sample, region)
        h_vals = f[hist_name].values[:, ptbin]
        h_edges = f[hist_name].edges[0]
        h_key = 'msd'
        return (h_vals, h_edges, h_key)


    # Get QCD efficiency
    qcdpass, qcdfail = 0., 0.
    for ptbin in range(npt):
        failCh = rl.Channel("ptbin%d%s" % (ptbin, 'fail'))
        passCh = rl.Channel("ptbin%d%s" % (ptbin, 'pass'))

        passTempl = get_templ("pass", "qcd", ptbin)
        failTempl = get_templ("fail", "qcd", ptbin)

        failCh.setObservation(failTempl)
        passCh.setObservation(passTempl)
        qcdfail += failCh.getObservation().sum()
        qcdpass += passCh.getObservation().sum()

    qcdeff = qcdpass / qcdfail

    # build actual fit model now
    model = rl.Model("tempModel")

    # Load templates
    f = uproot.open('hxx/hist_1DZcc_pt_scalesmear.root')
    templates = {}
    sys_list = ['JES']  # Sys list
    sysud_list = sum([[sys+"Up", sys+"Down"] for sys in sys_list], [])  # Sys name list
    for ptbin in range(npt):
        for region in ['pass', 'fail']:
            templates[region+str(ptbin)] = {}
            for sample in ['zbb', 'zcc', 'zqq', 'wcq', 'wqq', 'hcc125', 'tqq', 'hqq125', 'qcd', 'data_obs']:
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

            #for sName in ['zbb', 'zcc', 'zqq', 'wcq', 'wqq', 'hcc', 'tqq', 'hqq']:
            #include_samples = ['zcc']
            include_samples = ['zbb', 'zcc', 'zqq', 'wcq', 'wqq', 'hcc', 'tqq', 'hqq']
            if not fitTF:  # Add QCD sample when not running TF fit
                include_samples.append('qcd')
            for sName in include_samples:
                templ = templates[region+str(ptbin)][sName]
                stype = rl.Sample.SIGNAL if sName in ['zcc'] else rl.Sample.BACKGROUND
                sample = rl.TemplateSample(ch.name + '_' + sName, stype, templ)

                sample.setParamEffect(lumi, 1.027)

                ch.addSample(sample)

            if not pseudo:
                data_obs = templates[region+str(ptbin)]['data_obs']
                print("Reading real data")

            else:
                yields = sum(tpl[0] for samp, tpl in templates[region+str(ptbin)].items() if  samp in [*include_samples, 'qcd'])
                if throwPoisson:
                    yields = np.random.poisson(yields)

                data_obs = (yields, msd.binning, msd.name)
            ch.setObservation(data_obs)

            # drop bins outside rho validity
            mask = validbins[ptbin]
            if not pseudo:
                mask[10:14] = False
            ch.mask = mask

    if fitTF:
        tf_dataResidual = rl.BernsteinPoly("tf_dataResidual", (2, 2), ['pt', 'rho'], limits=(0, 10))
        tf_dataResidual_params = tf_dataResidual(ptscaled, rhoscaled)
        tf_params = qcdeff * tf_dataResidual_params

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
    # tqqPF = tqqpass.getExpectation(nominal=True).sum() / tqqfail.getExpectation(nominal=True).sum()
    # tqqpass.setParamEffect(tqqeffSF, 1*tqqeffSF)
    # tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
    # tqqpass.setParamEffect(tqqnormSF, 1*tqqnormSF)
    # tqqfail.setParamEffect(tqqnormSF, 1*tqqnormSF)

    with open("tempModel.pkl", "wb") as fout:
        pickle.dump(model, fout)

    model.renderCombine("tempModel")

if __name__ == '__main__':
    dummy_rhalphabet()
