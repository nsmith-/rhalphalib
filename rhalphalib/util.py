import numpy as np


def _to_numpy(hinput):
    if isinstance(hinput, tuple) and len(hinput) == 3:
        if not isinstance(hinput[0], np.ndarray):
            raise ValueError("Expected numpy array for element 0 of tuple %r" % hinput)
        if not isinstance(hinput[1], np.ndarray):
            raise ValueError("Expected numpy array for element 1 of tuple %r" % hinput)
        if not isinstance(hinput[2], str):
            raise ValueError("Expected string for element 2 of tuple %r" % hinput)
        if hinput[0].size != hinput[1].size - 1:
            raise ValueError("Counts array and binning array are incompatible in tuple %r" % hinput)
        return hinput
    elif str(type(hinput)) == "<class 'ROOT.TH1D'>":
        sumw = np.zeros(hinput.GetNbinsX())
        binning = np.zeros(sumw.size + 1)
        name = hinput.GetName()
        for i in range(1, sumw.size + 1):
            sumw[i-1] = hinput.GetBinContent(i)
            binning[i-1] = hinput.GetXaxis().GetBinLowEdge(i)
        binning[i] = hinput.GetXaxis().GetBinUpEdge(i)
        return (sumw, binning, name)
    elif str(type(hinput)) == "<class 'coffea.hist.hist_tools.Hist'>":
        sumw = hinput.values()[()]
        binning = hinput.axes()[0].edges()
        name = hinput.axes()[0].name
        return (sumw, binning, name)
    else:
        raise ValueError("Cannot understand template type of %r" % hinput)


def _to_TH1(sumw, binning, name):
    import ROOT
    h = ROOT.TH1D(name, "template;%s;Counts" % name, binning.size - 1, binning)
    h.SetDirectory(0)
    for i, w in enumerate(sumw):
        h.SetBinContent(i + 1, w)
    return h


def _pairwise_sum(array):
    if len(array) == 1:
        return array[0]
    elif len(array) % 2 != 0:
        # would be better to pick pseudorandom elements to merge
        array = np.append(array[:-2], array[-2] + array[-1])
    return _pairwise_sum(array[0::2] + array[1::2])


ROOFIT_HELPERS_INSTALLED = False


def install_roofit_helpers():
    global ROOFIT_HELPERS_INSTALLED
    if ROOFIT_HELPERS_INSTALLED:
        return
    ROOFIT_HELPERS_INSTALLED = True

    import ROOT as _ROOT
    import numbers as _numbers

    _ROOT.gEnv.SetValue("RooFit.Banner=0")
    # TODO: configurable verbosity
    _ROOT.RooMsgService.instance().setGlobalKillBelow(_ROOT.RooFit.WARNING)

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

    def _RooArgList_fromiter(cls, iterable, silent=False):
        items = cls()
        for item in iterable:
            items.add(item, silent)
        return items

    _ROOT.RooArgList.fromiter = classmethod(_RooArgList_fromiter)

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
