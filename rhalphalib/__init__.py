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
from .function import (
    BernsteinPoly,
)

__all__ = [
    'Model',
    'Channel',
    'Observable',
    'Sample',
    'TemplateSample',
    'ParametericSample',
    'TransferFactorSample',
    'NuisanceParameter',
    'IndependentParameter',
    'DependentParameter',
    'BernsteinPoly',
]

# Install roofit helpers
import ROOT as _ROOT
import numbers as _numbers


def _embed_ref(obj, dependents):
    # python reference counting gc will drop rvalue dependents
    # and we don't want to hand ownership to ROOT/RooFit because it's gc is garbage
    # So utilize python to save the day
    if not hasattr(obj, '_rhalphalib_embed_ref'):
        obj._rhalphalib_embed_ref = list()
    if isinstance(dependents, list):
        obj._rhalphalib_embed_ref.extend(dependents)
    else:
        obj._rhalphalib_embed_ref.append(dependents)


def _RooWorkspace_add(self, obj, recycle=False):
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


def _RooArgList_fromiter(iterable, silent=False):
    items = _ROOT.RooArgList()
    for item in iterable:
        items.add(item, silent)
    return items


_ROOT.RooArgList.fromiter = _RooArgList_fromiter


def _RooAbsReal__add__(self, other):
    '''
    Add two RooAbsReal instances or an RooAbsReal and a number.
    '''
    if hasattr(other, 'InheritsFrom') and other.InheritsFrom("RooAbsReal"):
        name = self.GetName() + '_add_' + other.GetName()
        out = _ROOT.RooAddition(name, name, _ROOT.RooArgList(self, other))
        _embed_ref(out, [self, other])
        return out
    elif isinstance(other, _numbers.Number):
        name = self.GetName() + '_add_const'
        out = _ROOT.RooFormulaVar(name, name, "%r + @0" % other, _ROOT.RooArgList(self))
        _embed_ref(out, [self])
        return out
    raise TypeError("unsupported operand type(s) for +: '%s' and '%s'" % (str(type(self)), str(type(other))))


_ROOT.RooAbsReal.__add__ = _RooAbsReal__add__


def _RooAbsReal__mul__(self, other):
    '''
    Multiplies two RooAbsReal instances or an RooAbsReal and a number.
    '''
    if hasattr(other, 'InheritsFrom') and other.InheritsFrom("RooAbsReal"):
        name = self.GetName() + '_mul_' + other.GetName()
        out = _ROOT.RooProduct(name, name, _ROOT.RooArgList(self, other))
        _embed_ref(out, [self, other])
        return out
    elif isinstance(other, _numbers.Number):
        name = self.GetName() + '_mul_const'
        out = _ROOT.RooFormulaVar(name, name, "%r * @0" % other, _ROOT.RooArgList(self))
        _embed_ref(out, [self])
        return out
    raise TypeError("unsupported operand type(s) for *: '%s' and '%s'" % (str(type(self)), str(type(other))))


_ROOT.RooAbsReal.__mul__ = _RooAbsReal__mul__
