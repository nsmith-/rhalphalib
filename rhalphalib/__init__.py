from .model import (
    Model,
    Channel,
)
from .sample import (
    Observable,
    Sample,
    TemplateSample,
    ParametericSample,
    TransferFactorSample,
)
from .parameter import (
    NuisanceParameter,
    IndependentParameter,
    DependentParameter,
)

__all__ = [
    'Model',
    'Channel',
    'Observable',
    'TemplateSample',
    'ParametericSample',
    'TransferFactorSample',
    'NuisanceParameter',
    'IndependentParameter',
    'DependentParameter',
]

# Install roofit helpers
PYROOFIT_INSTALLED = False
if not PYROOFIT_INSTALLED:
    PYROOFIT_INSTALLED = True
    import ROOT as _ROOT

    def _RooWorkspace_add(self, obj, recycle=True):
        '''
        Shorthand for RooWorkspace::import() since that is reserved in python
        Parameters:
            obj: the RooFit object to import
            recycle: when True, if 'obj' or any dependent has a name and type
                     matching a pre-existing object in the workspace, use existing object.
        '''
        wsimport = getattr(self, 'import')
        if recycle:
            return wsimport(obj, _ROOT.RooFit.RecycleConflictNodes())
        else:
            return wsimport(obj)

    _ROOT.RooWorkspace.add = _RooWorkspace_add

    def _RooAbsCollection__iter__(self):
        it = self.iterator()
        obj = it.Next()
        while obj != None:  # noqa: E711
            yield obj
            obj = it.Next()

    _ROOT.RooAbsCollection.__iter__ = _RooAbsCollection__iter__

    # This is mainly for collections of parameters
    def _RooAbsCollection_assign(self, other):
        if self == other:
            return
        for el in self:
            if not hasattr(el, 'setVal'):
                continue
            theirs = other.find(el)
            if not theirs:
                continue
            el.setVal(theirs.getVal())
            el.setError(theirs.getError())
            el.setAsymError(theirs.getErrorLo(), theirs.getErrorHi())
            el.setAttribute("Constant", theirs.isConstant())

    _ROOT.RooAbsCollection.assign = _RooAbsCollection_assign
