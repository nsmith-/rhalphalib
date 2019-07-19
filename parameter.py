import ROOT


class Parameter(object):
    def __init__(self, name):
        self._name = name
        self._hasPrior = False

    @property
    def name(self):
        return self._name

    def hasPrior(self):
        '''
        True if the prior is not flat
        '''
        return self._hasPrior


class NuisanceParameter(Parameter):
    def __init__(self, name, combinePrior):
        '''
        A nuisance parameter.
        name: name of parameter
        combinePrior: one of 'shape', 'shapeN', 'lnN', etc.
        '''
        super(NuisanceParameter, self).__init__(name)
        self._hasPrior = True
        self._prior = combinePrior

    def __str__(self):
        return "%s %s" % self.name, self.prior

    # TODO: setter?
    @property
    def combinePrior(self):
        return self._prior


class IndependentParameter(Parameter):
    DefaultRange = (-10, 10)

    def __init__(self, name, val, lo=None, hi=None):
        super(IndependentParameter, self).__init__(name)
        self._val = val
        self._lo = lo if lo is not None else self.DefaultRange[0]
        self._hi = hi if hi is not None else self.DefaultRange[1]

    def renderRoofit(self):
        return ROOT.RooRealVar(self._name, self._name, self._val, self._lo, self._hi)


class DependentParameter(Parameter):
    def __init__(self, name, formula, *dependents):
        super(DependentParameter, self).__init__(name)
        self._formula = formula
        self._dependentParams = dependents

    def renderRoofit(self):
        return ROOT.RooFormulaVar(self._name, self._name, self._formula, ROOT.RooArgList(*self._dependentParams))
