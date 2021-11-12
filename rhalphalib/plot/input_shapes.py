import numpy as np
import pickle


def input_dict_maker(input_file):
    with open(input_file, 'rb') as fout:
        model = pickle.load(fout)

    class ndict(dict):
        def __init__(self, *arg, **kw):
            super(ndict, self).__init__(*arg, **kw)
            self._name = ''

        @property
        def name(self):
            return self._name

        @name.setter
        def name(self, new_name):
            self._name = new_name

    from uproot_methods.classes.TH1 import Methods

    def name(self, name):
        self._fName = name

    def variance(self, name):
        self._fName = name

    Methods.name = Methods.name.setter(name)
    import uproot_methods.classes.TH1 as TH1
    mockd = ndict()
    for i, cat in enumerate(model.channels):
        cat_name = cat.name + '_inputs'
        mockd[cat_name] = ndict()
        mockd[cat_name].name = cat_name
        TotalSig = []
        TotalBkg = []
        TotalProcs = []
        for sample in model[cat.name].samples:
            sname = sample.name.split('_')[(-1)]
            try:
                h = sample.getExpectation(nominal=True)
                bins = sample.observable.binning
                if sample.sampletype == 0:
                    TotalSig.append(h)
                else:
                    TotalBkg.append(h)
                TotalProcs.append(h)
                mockd[cat_name][sname] = TH1.from_numpy((h, bins))
                mockd[cat_name][sname].name = sname
            except:  # noqa
                pass
        if TotalSig == []:
            TotalSig = [np.zeros(len(bins) - 1)]
        TotalSig = np.vstack(TotalSig).sum(axis=0)
        TotalBkg = np.vstack(TotalBkg).sum(axis=0)
        TotalProcs = np.vstack(TotalProcs).sum(axis=0)
        mockd[cat_name]['TotalSig'] = TH1.from_numpy((TotalSig, bins))
        mockd[cat_name]['TotalSig'].name = 'TotalSig'
        mockd[cat_name]['TotalBkg'] = TH1.from_numpy((TotalBkg, bins))
        mockd[cat_name]['TotalBkg'].name = 'TotalBkg'
        mockd[cat_name]['TotalProcs'] = TH1.from_numpy((TotalProcs, bins))
        mockd[cat_name]['TotalProcs'].name = 'TotalProcs'
        data = model[cat.name].getObservation()
        mockd[cat_name]['data'] = TH1.from_numpy((data, bins))
        mockd[cat_name]['data'].name = 'data'
        if 'muon' in cat_name:
            continue
        qcd = data - TotalProcs
        mockd[cat_name]['qcd'] = TH1.from_numpy((qcd, bins))
        mockd[cat_name]['qcd'].name = 'qcd'

    qcdp, qcdf = (0, 0)
    for i, cat in enumerate(model.channels):
        cat_name = cat.name + '_inputs'
        if 'qcd' not in mockd[cat_name].keys():
            continue
        if 'fail' in cat_name:
            qcdf += mockd[cat_name]['qcd'].values.sum()
        else:
            qcdp += mockd[cat_name]['qcd'].values.sum()

    qcdeff = qcdp / qcdf
    for i, cat in enumerate(model.channels):
        cat_name = cat.name + '_inputs'
        if 'fail' in cat_name or 'muon' in cat_name:
            continue
        _fail_qcd = mockd[cat_name.replace('pass', 'fail')]['qcd']
        _fail_qcd_scaled = _fail_qcd.values * qcdeff
        mockd[cat_name]['qcd'] = TH1.from_numpy((_fail_qcd_scaled, _fail_qcd.edges))
        mockd[cat_name]['qcd'].name = 'qcd'

    # binwnorm to match with fitDiag output
    for key, cat in mockd.items():
        for hname, h in cat.items():
            vals, bins = h.numpy()
            cat[hname] = TH1.from_numpy((vals / 7., bins))

    return mockd
