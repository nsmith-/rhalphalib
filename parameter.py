import ROOT


class Parameter(object):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name


class NuisanceParameter(Parameter):
    def __init__(self, name, prior):
        '''
        name: name of parameter
        prior: one of 'shape', 'shapeN', 'lnN', etc.
        '''
        super(NuisanceParameter, self).__init__(name)
        self._prior = prior

    def __str__(self):
        return "%s %s" % self.name, self.prior

    # TODO: setter?
    @property
    def prior(self):
        return self._prior


class IndependentParameter(Parameter):
    defaultRange = (-10, 10)

    def __init__(self, name, formula, *dependents):
        super(IndependentParameter, self).__init__(name)

    def renderRoofit(self):
        return ROOT.RooRealVar(self._name, self._name, defaultRange[0], defaultRange[1])


class DependentParameter(Parameter):
    def __init__(self, name, formula, *dependents):
        super(DependentParameter, self).__init__(name)
        self._formula = formula
        self._dependentParams = set(dependents)

    def renderRoofit(self):
        pass
