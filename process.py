import ROOT
from fnal_column_analysis_tools import hist
import numpy as np


def normalize_histogram(hinput):
    if isinstance(hinput, ROOT.TH1):
        return hinput
    elif isinstance(hinput, hist.Hist):
        binning = hinput.axes()[0].edges()
        sumw, sumw2 = hinput.values(sumw2=True, overflow='all')[()]
        h = ROOT.TH1D("template", "template;%s;Counts" % hinput.axes()[0].name, binning.size - 1, binning)
        h.SetDirectory(0)
        for i, (w, w2) in enumerate(zip(sumw, sumw2)):
            h.SetBinContent(i, w)
            h.SetBinError(i, w2)
        return h
    else:
        raise ValueError


def rooObservableFromAxis(name, haxis):
    return ROOT.RooRealVar(name,
                           haxis.GetTitle(),
                           haxis.GetBinLowEdge(1),
                           haxis.GetBinUpEdge(haxis.GetNbins())
                           )


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
        raise NotImplementedError

    @property
    def parameters(self):
        '''
        Set of parameters other than nuisance params that affect this process
        '''
        raise NotImplementedError

    def normalization(self, systematicName=''):
        '''
        Process yield (or integral for shape), either the nominal value or
        when a systematic shift is applied (include 'Up' or 'Down' as appropriate)
        systematicName: string
        '''
        raise NotImplementedError

    def renderRoofitModel(self, workspace, channelName):
        '''
        Import the necessary Roofit objects into the workspace for this process
        '''
        raise NotImplementedError

    def nuisanceParamString(self, param):
        '''
        A formatted string for placement into the combine datacard
        '''
        raise NotImplementedError


class SingleBinProcess(Process):
    def __init__(self, processtype, normalization):
        super(SingleBinProcess, self).__init__(processtype)
        self._norm = normalization
        self._nuisanceShifts = {}
        self._nuisanceParams = set()
        self._parameters = set()

    @property
    def nuisanceParameters(self):
        return self._nuisanceParams

    @property
    def parameters(self):
        return self._parameters

    def normalization(self, systematicName=''):
        if systematicName == '':
            return self._norm
        else:
            return self._nuisanceShifts[systematicName]

    def renderRoofitModel(self, workspace, channelName):
        # Nothing needed in combine workspace for single bin process
        pass

    def nuisanceParamString(self, param):
        if param not in self.nuisanceParameters:
            return '-'
        if param + 'Up' in self._nuisanceShifts:
            up = self._nuisanceShifts[param + 'Up']
            if param + 'Down' in self._nuisanceShifts:
                down = self._nuisanceShifts[param + 'Down']
                return '%.3f/%.3f' % (up, down)
            return '%.3f' % up

    def addSystematic(self, syst, value, relative=False):
        '''
        syst: name including direction of shift ('Up' or 'Down')
            if symmetric shift, only add the 'Up' version
        value: scalar
        relative: True if delta relative to the nominal
        '''
        # locate existing
        if syst[-2:] == 'Up':
            self._nuisanceParams.add(syst[:-2])
        elif syst[-4:] == 'Down':
            self._nuisanceParams.add(syst[:-4])
        if relative:
            value = value * self.normalization
        self._nuisanceShifts[syst] = value


class TemplateProcess(Process):
    def __init__(self, processtype, template, observable="x"):
        '''
        processtype: Process.SIGNAL or BACKGROUND or DATA
        template: a histogram (either ROOT TH1 or fnal Hist object)
        observable: the name for the variable on x axis, really doesn't matter but RooFit wants it
        '''
        super(TemplateProcess, self).__init__(processtype)
        self._nominal = normalize_histogram(template)
        self._observable = observable
        self._nuisanceShifts = {}
        self._nuisanceTemplates = {}
        self._nuisanceParams = set()
        self._parameters = set()

    @property
    def nuisanceParameters(self):
        return self._nuisanceParams

    @property
    def parameters(self):
        return self._parameters

    def normalization(self, systematicName=''):
        if systematicName == '':
            return self._nominal.Integral()
        return self._nuisanceTemplates[systematicName].Integral()

    def renderRoofitModel(self, workspace, channelName):
        processName = channelName + '_' + self.name
        observable = workspace.var(self._observable)
        if not observable:
            observable = rooObservableFromAxis(self._observable, self._nominal.GetXaxis())
        rootemplate = ROOT.RooDataHist(processName, processName, ROOT.RooArgList(observable), self._nominal)
        getattr(workspace, 'import')(rootemplate)
        for syst, template in self._nuisanceTemplates.items():
            name = processName + '_' + syst
            rootemplate = ROOT.RooDataHist(name, name, ROOT.RooArgList(observable), template)
            getattr(workspace, 'import')(rootemplate)

    def nuisanceParamString(self, param):
        if param not in self.nuisanceParameters:
            return '-'
        if param+'Up' in self._nuisanceTemplates:
            return '1'
        elif param + 'Up' in self._nuisanceShifts:
            up = self._nuisanceShifts[param + 'Up']
            if param + 'Down' in self._nuisanceShifts:
                down = self._nuisanceShifts[param + 'Down']
                return '%.3f/%.3f' % (up, down)
            return '%.3f' % up

    def addSystematic(self, syst, value, relative=False):
        '''
        Add a normalization systematic
        syst: name including direction of shift ('Up' or 'Down')
            if symmetric shift, only add the 'Up' version
        value: scalar
        relative: True if delta relative to the nominal
        '''
        # locate existing
        if syst[-2:] == 'Up':
            self._nuisanceParams.add(syst[:-2])
        elif syst[-4:] == 'Down':
            self._nuisanceParams.add(syst[:-4])
        if relative:
            value = value * self.normalization
        self._nuisanceShifts[syst] = value

    def addTemplateSystematic(self, syst, template):
        '''
        Add a template systematic
        syst: name including direction of shift ('Up' or 'Down')
            if symmetric shift, only add the 'Up' version
        template: a histogram
        '''
        # locate existing
        if syst[-2:] == 'Up':
            self._nuisanceParams.add(syst[:-2])
        elif syst[-4:] == 'Down':
            self._nuisanceParams.add(syst[:-4])
        self._nuisanceTemplates[syst] = normalize_histogram(template)


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

    def renderRoofitModel(self, workspace, channelName):
        pass


class TemplateTransferFactorProcess(Process):
    pass


class ParameterizedTransferFactorProcess(Process):
    pass
