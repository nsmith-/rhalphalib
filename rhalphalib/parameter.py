import numbers
import warnings
import numpy as np
from .util import install_roofit_helpers


class Parameter(object):
    def __init__(self, name, value):
        self._name = name
        self._value = value
        self._hasPrior = False
        self._intermediate = False

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

    @property
    def value(self):
        return self._value

    @property
    def intermediate(self):
        '''
        An intermediate parameter is one that should not be explicitly rendered.
        The formula will be expanded recursively until it depends only on non-intermediate value.
        Only DependentParameters can be intermediate, hence one can modify this flag for them.
        '''
        return self._intermediate

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
        return 'flatParam'

    def getDependents(self, rendering=False, deep=False):
        return {self}

    def formula(self):
        return '{' + self._name + '}'

    def renderRoofit(self, workspace):
        raise NotImplementedError

    def _binary_op(self, opinfo, other):
        opname, op, right = opinfo
        if isinstance(other, Parameter):
            if right:
                name = other.name + opname + self.name
                out = DependentParameter(name, "{0}%s{1}" % op, other, self)
            else:
                name = self.name + opname + other.name
                out = DependentParameter(name, "{0}%s{1}" % op, self, other)
            out.intermediate = True
            return out
        elif isinstance(other, numbers.Number):
            if right:
                name = type(other).__name__ + opname + self.name
                out = DependentParameter(name, "%r%s{0}" % (other, op), self)
            else:
                name = self.name + opname + type(other).__name__
                out = DependentParameter(name, "{0}%s%r" % (op, other), self)
            out.intermediate = True
            return out
        return NotImplemented

    def __radd__(self, other):
        return self._binary_op(('_add_', '+', True), other)

    def __rsub__(self, other):
        return self._binary_op(('_sub_', '-', True), other)

    def __rmul__(self, other):
        return self._binary_op(('_mul_', '*', True), other)

    def __rtruediv__(self, other):
        return self._binary_op(('_div_', '/', True), other)

    def __rpow__(self, other):
        return self._binary_op(('_pow_', '**', True), other)

    def __add__(self, other):
        return self._binary_op(('_add_', '+', False), other)

    def __sub__(self, other):
        return self._binary_op(('_sub_', '-', False), other)

    def __mul__(self, other):
        return self._binary_op(('_mul_', '*', False), other)

    def __truediv__(self, other):
        return self._binary_op(('_div_', '/', False), other)

    def __pow__(self, other):
        return self._binary_op(('_pow_', '**', False), other)

    def max(self, val):
        """Return maximum out of param value and given ``val``"""
        return DependentParameter("max(%s,%s)" % (self.name, val), "TMath::Max({0}, %s)" % val, self)


class IndependentParameter(Parameter):
    DefaultRange = (-10, 10)

    def __init__(self, name, value, lo=None, hi=None, constant=False):
        super(IndependentParameter, self).__init__(name, value)
        self._lo = lo if lo is not None else self.DefaultRange[0]
        self._hi = hi if hi is not None else self.DefaultRange[1]
        self._constant = constant

    @Parameter.value.setter
    def value(self, val):
        self._value = val

    @property
    def lo(self):
        return self._lo

    @lo.setter
    def lo(self, lo):
        self._lo = lo

    @property
    def hi(self):
        return self._hi

    @hi.setter
    def hi(self, hi):
        self._hi = hi

    @property
    def constant(self):
        return self._constant

    @constant.setter
    def constant(self, const):
        self._constant = const

    def renderRoofit(self, workspace):
        import ROOT
        install_roofit_helpers()
        if workspace.var(self._name) == None:  # noqa: E711
            var = ROOT.RooRealVar(self._name, self._name, self._value, self._lo, self._hi)
            var.setAttribute("Constant", self._constant)
            workspace.add(var)
        return workspace.var(self._name)


class NuisanceParameter(IndependentParameter):
    def __init__(self, name, combinePrior, value=0, lo=None, hi=None):
        '''
        A nuisance parameter.
        name: name of parameter
        combinePrior: one of 'shape', 'shapeN', 'lnN', etc.

        Render the prior somewhere else?  Probably in Model because the prior needs
        to be added at the RooSimultaneus level (I think)
        Filtering the set of model parameters for these classes can collect needed priors.
        '''
        super(NuisanceParameter, self).__init__(name, value, lo, hi)
        self._hasPrior = True
        if combinePrior not in {'shape', 'shapeN', 'shapeU', 'lnN', 'lnU', 'gmM', 'trG', 'param'}:
            raise ValueError("Unrecognized combine prior %s" % combinePrior)
        self._prior = combinePrior

    @property
    def combinePrior(self):
        return self._prior


class DependentParameter(Parameter):
    def __init__(self, name, formula, *dependents):
        '''
        Create a dependent parameter
            name: name of parameter
            formula: a python format-string using only indices, e.g.
                '{0} + sin({1})*{2}'
        '''
        super(DependentParameter, self).__init__(name, np.nan)
        if not all(isinstance(d, Parameter) for d in dependents):
            raise ValueError
        # TODO: validate formula for allowed functions
        self._formula = formula
        self._dependents = dependents

    @property
    def value(self):
        # TODO: value from rendering formula and eval() or numexpr or TFormula or ...
        raise NotImplementedError

    @Parameter.intermediate.setter
    def intermediate(self, val):
        self._intermediate = val

    def getDependents(self, rendering=False, deep=False):
        '''
        Return a set of parameters that this parameter depends on, which will be rendered.
        By default, this means all non-intermediate dependent parameters, recursively descending and stopping at
        the first renderable parameter (i.e. either non-intermediate or an IndependentParameter)
        If this parameter itself is renderable, we return a set of just this parameter.
        If rendering=True, we pass through this parameter if it is renderable.
        If deep=True, descend all the way to the IndependentParameters
        '''
        dependents = set()
        if deep:
            for p in self._dependents:
                if isinstance(p, DependentParameter):
                    dependents.update(p.getDependents(deep=True))
                else:
                    dependents.add(p)
            return dependents
        if not (self.intermediate or rendering):
            return {self}
        for p in self._dependents:
            if p.intermediate:
                dependents.update(p.getDependents())
            else:
                dependents.add(p)
        return dependents

    def formula(self, rendering=False):
        if not (self.intermediate or rendering):
            return "{" + self.name + "}"
        return "(" + self._formula.format(*(p.formula() for p in self._dependents)) + ")"

    def renderRoofit(self, workspace):
        import ROOT
        install_roofit_helpers()
        if workspace.function(self._name) == None:  # noqa: E711
            if self.intermediate:
                # This is a warning because we should make sure the name does not conflict as
                # intermediate parameter names are often autogenerated and might not be unique/appropriate
                warnings.warn("Rendering intermediate parameter: %r" % self, RuntimeWarning)
                self.intermediate = False
            rooVars = [v.renderRoofit(workspace) for v in self.getDependents(rendering=True)]
            # Originally just passed the named variables to RooFormulaVar but it seems the TFormula class
            # is more sensitive to variable names than is reasonable, so we reindex here
            formula = self.formula(rendering=True).format(**{var.GetName(): '@%d' % i for i, var in enumerate(rooVars)})
            var = ROOT.RooFormulaVar(self._name, self._name, formula, ROOT.RooArgList.fromiter(rooVars))
            workspace.add(var)
        return workspace.function(self._name)


class SmoothStep(DependentParameter):
    def __init__(self, param):
        if not isinstance(param, Parameter):
            raise ValueError("Expected a Parameter instance, got %r" % param)
        if param.intermediate:
            raise ValueError("SmoothStep can only depend on a non-intermediate parameter")
        super(SmoothStep, self).__init__(param.name + '_smoothstep', '{0}', param)
        self.intermediate = False

    @property
    def value(self):
        raise NotImplementedError

    def formula(self, rendering=False):
        return "{" + self.name + "}"

    def renderRoofit(self, workspace):
        import ROOT
        install_roofit_helpers()
        if workspace.function(self._name) == None:  # noqa: E711
            # Formula satisfies f(x<=-1) = 0, f(x>=1) = 1, f'(-1) = f'(1) = f''(-1) = f''(1) = 0
            formula = "(((0.1875*x*x - 0.625)*x*x + 0.9375)*x + 0.5)*(x > -1)*(x < 1) + 1*(x >= 1)".replace("x", "@0")
            rooVars = [v.renderRoofit(workspace) for v in self.getDependents(rendering=True)]
            if len(rooVars) != 1:
                raise RuntimeError("Unexpected number of parameters encountered while rendering SmoothStep")
            var = ROOT.RooFormulaVar(self._name, self._name, formula, ROOT.RooArgList.fromiter(rooVars))
            workspace.add(var)
        return workspace.function(self._name)


class Observable(Parameter):
    '''
    A simple struct that holds the name of an observable (e.g. x axis of discriminator histogram) and its binning
    The first sample attached to a channel will dictate how the rendering of the observable is done.
    Subequent samples attached will be checked against the first, and if they match, their observable will be set
    to the first samples' instance of this class.
    '''
    def __init__(self, name, binning):
        super(Observable, self).__init__(name, np.nan)
        self._binning = np.array(binning)

    def __eq__(self, other):
        if isinstance(other, Observable) and self._name == other._name and np.allclose(self._binning, other._binning):
            return True
        return False

    @property
    def name(self):
        return self._name

    @property
    def binning(self):
        return self._binning

    @property
    def nbins(self):
        return len(self._binning) - 1

    def binningTArrayD(self):
        import ROOT
        return ROOT.TArrayD(len(self._binning), self._binning)

    def renderRoofit(self, workspace):
        '''
        Return a RooObservable following the definition
        '''
        import ROOT
        install_roofit_helpers()
        if workspace.var(self._name) == None:  # noqa: E711
            var = ROOT.RooRealVar(self.name, self.name, self.binning[0], self.binning[-1])
            var.setBinning(ROOT.RooBinning(self.nbins, self.binning))
            workspace.add(var)
        return workspace.var(self._name)

    def formula(self):
        raise RuntimeError("Observables cannot be used in formulas, as this would necessitate support for numeric integration, which is outside the scope of rhalphalib.")
