import numpy as np


def _to_numpy(hinput, read_sumw2=False):
    if isinstance(hinput, tuple) and len(hinput) >= 3:
        if not isinstance(hinput[0], np.ndarray):
            raise ValueError("Expected numpy array for element 0 of tuple {}".format(hinput))
        if not isinstance(hinput[1], np.ndarray):
            raise ValueError("Expected numpy array for element 1 of tuple {}".format(hinput))
        if not isinstance(hinput[2], str):
            raise ValueError("Expected string for element 2 of tuple {}".format(hinput))
        if read_sumw2 and len(hinput) < 4:
            raise ValueError("Expected 4 elements of tuple {}, as read_sumw2=True".format(hinput))
        if read_sumw2 and not isinstance(hinput[3], np.ndarray):
            raise ValueError("Expected numpy array for element 3 of tuple {}, as read_sumw2=True".format(hinput))
        if hinput[0].size != hinput[1].size - 1:
            raise ValueError("Counts array and binning array are incompatible in tuple {}".format(hinput))
        if read_sumw2 and hinput[3].size != hinput[1].size - 1:
            raise ValueError("Sumw2 array and binning array are incompatible in tuple {}".format(hinput))
        return hinput
    elif "<class 'ROOT.TH1" in str(type(hinput)):
        sumw = np.zeros(hinput.GetNbinsX())
        sumw2 = np.zeros(hinput.GetNbinsX())
        binning = np.zeros(sumw.size + 1)
        name = hinput.GetXaxis().GetTitle()
        for i in range(1, sumw.size + 1):
            sumw[i-1] = hinput.GetBinContent(i)
            sumw2[i-1] = hinput.GetBinError(i)**2
            binning[i-1] = hinput.GetXaxis().GetBinLowEdge(i)
        binning[i] = hinput.GetXaxis().GetBinUpEdge(i)
        if read_sumw2:
            return (sumw, binning, name, sumw2)
        return (sumw, binning, name)
    elif str(type(hinput)) == "<class 'coffea.hist.hist_tools.Hist'>":
        sumw, sumw2 = hinput.values(sumw2=True)[()]
        binning = hinput.axes()[0].edges()
        name = hinput.axes()[0].name
        if read_sumw2:
            return (sumw, binning, name, sumw2)
        return (sumw, binning, name)
    elif str(type(hinput)) == "<class 'hist.hist.Hist'>":
        if read_sumw2 and hinput.variances() is None:
            raise ValueError("Expected Weight storage in Hist {}, as read_sumw2=True".format(hinput))

        if len(hinput.axes) > 1:
            if str(type(hinput.axes[0])) == "<class 'hist.axis.StrCategory'>":
                if len(hinput.axes[0]) == 1:
                    hinput = hinput[0, :]
                else:
                    raise ValueError("Expected single sample in Hist {}".format(hinput))
            else:
                raise ValueError("Expected 1D histogram in Hist {}".format(hinput))

        sumw = hinput.values()
        binning = hinput.axes[0].edges
        name = hinput.axes[0].name
        if read_sumw2:
            sumw2 = hinput.variances()
            return (sumw, binning, name, sumw2)
        return (sumw, binning, name)
    else:
        raise ValueError("Cannot understand template type of %r" % hinput)


def _to_TH1(sumw, binning, name):
    import ROOT
    h = ROOT.TH1D(name, "template;%s;Counts" % name, binning.size - 1, binning)
    if isinstance(sumw, tuple):
        for i, (w, w2) in enumerate(zip(sumw[0], sumw[1])):
            h.SetBinContent(i + 1, w)
            h.SetBinError(i + 1, np.sqrt(w2))
    else:
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
    root_version = _ROOT.gROOT.GetVersionInt()

    _ROOT.TH1.AddDirectory(False)

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

    if root_version < 62200:
        # https://sft.its.cern.ch/jira/browse/ROOT-10457
        def _RooAbsCollection__iter__(self):
            it = self.iterator()
            obj = it.Next()
            while obj != None:  # noqa: E711
                yield obj
                obj = it.Next()

        if hasattr(_ROOT.RooAbsCollection, "__iter__"):
            del _ROOT.RooAbsCollection.__iter__
            del _ROOT.RooArgList.__iter__
            del _ROOT.RooArgSet.__iter__

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

    def _RooFitResult_nameArray(self):
        '''
        Returns a numpy array of floating parameter names
        '''
        return np.array([p.GetName() for p in self.floatParsFinal()])

    _ROOT.RooFitResult.nameArray = _RooFitResult_nameArray

    def _RooFitResult_valueArray(self):
        '''
        Returns a numpy array of floating parameter values
        '''
        return np.array([p.getVal() for p in self.floatParsFinal()])

    _ROOT.RooFitResult.valueArray = _RooFitResult_valueArray

    def _RooFitResult_covarianceArray(self):
        '''
        Returns a numpy array of floating parameter covariances
        '''
        param_cov = self.covarianceMatrix()
        param_cov = np.frombuffer(param_cov.GetMatrixArray(), dtype='d', count=param_cov.GetNoElements())
        param_cov = param_cov.reshape(int(np.sqrt(param_cov.size)), -1)
        return param_cov

    _ROOT.RooFitResult.covarianceArray = _RooFitResult_covarianceArray
