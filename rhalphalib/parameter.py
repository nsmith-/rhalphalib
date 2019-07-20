import ROOT
import numbers


class Parameter(object):
    def __init__(self, name):
        self._name = name
        self._hasPrior = False

    def __repr__(self):
        return "<%s (%s) instance at 0x%x>" % (
            self.__class__.__name__,
            self._name,
            id(self),
        )

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def hasPrior(self):
        '''
        True if the prior is not flat
        '''
        return self._hasPrior

    @property
    def combinePrior(self):
        '''
        By default assume param has no prior and we are just informing combine about it
        '''
        return 'param'

    def getDependents(self, recursive=False):
        return {self}

    def formula(self, recursive=False):
        return self._name

    def renderRoofit(self, workspace):
        raise NotImplementedError

    def _binary_op(self, opinfo, other):
        opname, op, right = opinfo
        if isinstance(other, Parameter):
            if right:
                name = other.name + opname + self.name
                return DependentParameter(name, "{0}%s{1}" % op, other, self)
            else:
                name = self.name + opname + other.name
                return DependentParameter(name, "{0}%s{1}" % op, self, other)
        elif isinstance(other, numbers.Number):
            if right:
                name = type(other).__name__ + opname + self.name
                return DependentParameter(name, "%r%s{0}" % (other, op), self)
            else:
                name = self.name + opname + type(other).__name__
                return DependentParameter(name, "{0}%s%r" % (op, other), self)
        raise TypeError("unsupported operand type(s) for %s: '%s' and '%s'" % (op, str(type(self)), str(type(other))))

    def __radd__(self, other):
        return self._binary_op(('_add_', '+', True), other)

    def __rsub__(self, other):
        return self._binary_op(('_sub_', '-', True), other)

    def __rmul__(self, other):
        return self._binary_op(('_mul_', '*', True), other)

    def __rtruediv__(self, other):
        return self._binary_op(('_div_', '/', True), other)

    def __add__(self, other):
        return self._binary_op(('_add_', '+', False), other)

    def __sub__(self, other):
        return self._binary_op(('_sub_', '-', False), other)

    def __mul__(self, other):
        return self._binary_op(('_mul_', '*', False), other)

    def __truediv__(self, other):
        return self._binary_op(('_div_', '/', False), other)


class IndependentParameter(Parameter):
    DefaultRange = (-10, 10)

    def __init__(self, name, val, lo=None, hi=None):
        super(IndependentParameter, self).__init__(name)
        self._val = val
        self._lo = lo if lo is not None else self.DefaultRange[0]
        self._hi = hi if hi is not None else self.DefaultRange[1]

    def renderRoofit(self, workspace):
        if workspace.var(self._name) == None:
            var = ROOT.RooRealVar(self._name, self._name, self._val, self._lo, self._hi)
            workspace.add(var)
        return workspace.var(self._name)


class NuisanceParameter(IndependentParameter):
    def __init__(self, name, combinePrior, val=0, lo=None, hi=None):
        '''
        A nuisance parameter.
        name: name of parameter
        combinePrior: one of 'shape', 'shapeN', 'lnN', etc.

        Render the prior somewhere else?  Probably in Model because the prior needs
        to be added at the RooSimultaneus level (I think)
        Filtering the set of model parameters for these classes can collect needed priors.
        '''
        super(NuisanceParameter, self).__init__(name, val, lo, hi)
        self._hasPrior = True
        self._prior = combinePrior

    # TODO: unused?
    def __str__(self):
        return "%s %s" % self.name, self.prior

    @property
    def combinePrior(self):
        # TODO: setter?
        return self._prior


class DependentParameter(Parameter):
    def __init__(self, name, formula, *dependents):
        '''
        Create a dependent parameter
            name: name of parameter
            formula: a python format-string using only indices, e.g.
                '{0} + sin({1})*{2}'
        '''
        super(DependentParameter, self).__init__(name)
        if not all(isinstance(d, Parameter) for d in dependents):
            raise ValueError
        self._formula = formula
        self._dependents = dependents

    def getDependents(self, recursive=False):
        if not recursive:
            return set(self._dependents)
        dependents = set()
        for p in self._dependents:
            dependents.update(p.getDependents(True))
        return dependents

    def formula(self, recursive=False):
        return "(" + self._formula.format(*(p.formula(recursive) for p in self._dependents)) + ")"

    def renderRoofit(self, workspace):
        if workspace.function(self._name) == None:
            rooVars = [v.renderRoofit(workspace) for v in self.getDependents(recursive=True)]
            var = ROOT.RooFormulaVar(self._name, self._name, self.formula(recursive=True), ROOT.RooArgList.fromiter(rooVars))
            workspace.add(var)
        return workspace.function(self._name)
