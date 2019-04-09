import ROOT
from fnal_column_analysis_tools import hist
import numpy as np


def normalize_histogram(hinput):
    if isinstance(hinput, ROOT.TH1):
        return hinput
    elif isinstance(hinput, hist.Hist):
        binning = hinput.axes()[0].edges()
        sumw, sumw2 = hinput.values(sumw2=True, overflow='all')[()]
        h = ROOT.TH1D("temp", "", binning.size - 1, binning)
        h.SetDirectory(0)
        for i, (w, w2) in enumerate(zip(sumw, sumw2)):
            h.SetBinContent(i, w)
            h.SetBinError(i, w2)
        return h
    else:
        raise ValueError


class Process(object):
    """
    Process base class
    """
    SIGNAL, BACKGROUND, DATA = range(3)

    def __init__(self, processtype):
        self.processtype = processtype
        self._link = (None, "undefined")

    def __repr__(self):
        return "<%s (linked to %r as '%s') at 0x%x>" % (
            self.__class__.__name__,
            self._link[0],
            self._link[1],
            id(self),
        )

    def link(self, parent, key=''):
        self._link = (parent, key)

    @property
    def name(self):
        return self._link[1]

    @property
    def nuisanceParameters(self):
        '''
        Set of nuisance parameters that influence this process
        '''
        return set()

    @property
    def parameters(self):
        '''
        Set of parameters other than nuisance params that affect this process
        '''
        return set()

    def normalization(self, systematicName=''):
        '''
        Process yield (or integral for shape), either the nominal value or
        when a systematic shift is applied (include 'Up' or 'Down' as appropriate)
        systematicName: string
        '''
        return 0.

    def renderRoofitModel(self, workspace, processName):
        '''
        Import the necessary Roofit objects into the workspace for this process
        '''
        pass

    def nuisanceParamString(self, param):
        '''
        A formatted string for placement into the combine datacard
        '''
        raise KeyError(param)


class SingleBinProcess(Process):
    def __init__(self, processtype, normalization):
        super(SingleBinProcess, self).__init__(processtype)
        self._norm = normalization
        self._nuisanceShifts = {}
        self._nuisanceParams = set()

    @property
    def nuisanceParameters(self):
        return self._nuisanceParams

    def normalization(self, systematicName=''):
        if systematicName == '':
            return self._norm
        else:
            return self._nuisanceShifts[systematicName]

    def nuisanceParamString(self, param):
        if param not in self.nuisanceParameters:
            return '-'
        if param + 'Up' in self._nuisanceShifts:
            up = self._nuisanceShifts[param + 'Up']
            if param + 'Down' in self._nuisanceShifts:
                down = self._nuisanceShifts[param + 'Down']
                return '%.3f/%.3f' % (up, down)
            return '%.3f' % up

    def addSystematic(self, syst, value):
        self._nuisanceShifts[syst] = value
        if syst[-2:] == 'Up':
            self._nuisanceParams.add(syst[:-2])
        elif syst[-4:] == 'Down':
            self._nuisanceParams.add(syst[:-4])


class TemplateProcess(Process):
    def __init__(self, processtype, template):
        super(TemplateProcess, self).__init__(processtype)
        self._nominal = normalize_histogram(template)
        self._nuisanceTemplates = {}
        self._nuisanceParams = set()

    @property
    def nuisanceParameters(self):
        return self._nuisanceParams

    def normalization(self, systematicName=''):
        if systematicName == '':
            return self._nominal.Integral()
        return self._nuisanceTemplates[systematicName].Integral()

    def nuisanceParamString(self, param):
        if param not in self.nuisanceParameters:
            return '-'
        if param+'Up' in self._nuisanceTemplates:
            return '1'

    def addshape(self, syst, template):
        self._nuisanceTemplates[syst] = normalize_histogram(template)
        if syst[-2:] == 'Up':
            self._nuisanceParams.add(syst[:-2])
        elif syst[-4:] == 'Down':
            self._nuisanceParams.add(syst[:-4])


class PerBinParameterProcess(Process):
    def __init__(self, processtype, binningDefinition):
        super(PerBinParameterProcess, self).__init__(processtype)
        if isinstance(binningDefinition, np.ndarray):
            self._binning = binningDefinition
        elif isinstance(binningDefinition, hist.Bin):
            self._binning = binningDefinition.edges()
        else:
            raise ValueError
        self._initialParams = np.zeros(self._binning.size - 1)
        self._initialized = False

    @property
    def parameters(self):
        return set()

    def _initialize(self):
        if not self._initialized:
            raise NotImplementedError

    def normalization(self, systematicName=''):
        self._initialize()
        return 0.

    def renderRoofitModel(self, workspace, processName):
        pass


class TemplateTransferFactorProcess(Process):
    pass


class ParameterizedTransferFactorProcess(Process):
    pass
