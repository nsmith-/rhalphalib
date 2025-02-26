from __future__ import division
import numpy as np
import numbers
import warnings
import logging
from .parameter import (
    Parameter,
    IndependentParameter,
    NuisanceParameter,
    DependentParameter,
    SmoothStep,
    Observable,
)
from .util import _to_numpy, _to_TH1, _pairwise_sum, install_roofit_helpers


class Sample(object):
    """
    Sample base class
    """

    SIGNAL, BACKGROUND = range(2)

    def __init__(self, name, sampletype):
        self._name = name
        self._sampletype = sampletype
        self._observable = None
        self._mask = None
        self._mask_val = 0.0

    def __repr__(self):
        return "<%s (%s) instance at 0x%x>" % (
            self.__class__.__name__,
            self._name,
            id(self),
        )

    @property
    def name(self):
        return self._name

    @property
    def sampletype(self):
        return self._sampletype

    @property
    def observable(self):
        if self._observable is None:
            raise RuntimeError("A Sample was not constructed correctly")
        return self._observable

    @observable.setter
    def observable(self, obs):
        # TODO check compatible?
        self._observable = obs

    @property
    def parameters(self):
        raise NotImplementedError

    @property
    def mask(self):
        """
        An array matching the observable binning that specifies which bins to populate
        i.e. when mask[i] is False, the bin content will be set to 0.
        """
        return self._mask

    @mask.setter
    def mask(self, mask):
        if isinstance(mask, np.ndarray):
            mask = mask.astype(bool)
            if self.observable.nbins != len(mask):
                raise ValueError("Mask shape does not match number of bins in observable")
            # protect from mutation
            mask.setflags(write=False)
        elif mask is not None:
            raise ValueError("Mask should be None or a numpy array")
        self._mask = mask

    def setParamEffect(self, param, effect_up, effect_down=None):
        raise NotImplementedError

    def getParamEffect(self, param, up=True):
        raise NotImplementedError

    def getExpectation(self, nominal=False):
        raise NotImplementedError

    def renderRoofit(self, workspace):
        raise NotImplementedError

    def combineNormalization(self):
        raise NotImplementedError

    def combineParamEffect(self, param):
        raise NotImplementedError


class TemplateSample(Sample):
    def __init__(self, name, sampletype, template, force_positive=False):
        """
        name: self-explanatory
        sampletype: Sample.SIGNAL or BACKGROUND or DATA
        template: Either a ROOT TH1, a 1D Coffea Hist object, a 1D hist Hist object, or a numpy histogram
            in the latter case, please extend the numpy histogram tuple to define an observable name
            i.e. (sumw, binning, name)
            (for the others, the observable name is taken from the x axis name)
        """
        super(TemplateSample, self).__init__(name, sampletype)
        sumw2 = None
        try:
            sumw, binning, obs_name, sumw2 = _to_numpy(template, read_sumw2=True)
        except ValueError:
            sumw, binning, obs_name = _to_numpy(template)
        if force_positive:
            if np.any(sumw < 0):
                logging.info(f"Negative values found in sample '{name}'. They are being set to 0.")
            sumw[sumw < 0] = 0.0
            if sumw2 is not None:
                sumw2[sumw2 < 0] = 0.0
        if np.any(sumw < 0):
            logging.warning(
                f"Sample '{name}' template contains negative yields. This may cause normalization mismatch issues when building the workspace. "
                "Set `log_level` to 'logging.DEBUG' to show the template:"
            )
            logging.debug(f"Sample '{name}' template = {sumw}.")
        observable = Observable(obs_name, binning)
        self._observable = observable
        self._nominal = sumw
        self._sumw2 = sumw2
        self._paramEffectsUp = {}
        self._paramEffectsDown = {}
        self._paramEffectScales = {}
        self._extra_dependencies = set()

    def show(self):
        print(self._nominal)
        if self._sumw2 is not None:
            print(self._sumw2)

    def scale(self, _scale):
        self._nominal *= _scale
        if self._sumw2 is not None:
            self._sumw2 *= _scale * _scale

    @property
    def parameters(self):
        """
        Set of independent parameters that affect this sample
        """
        pset = set(self._paramEffectsUp.keys())
        pset.update(self._extra_dependencies)
        return pset

    def setParamEffect(self, param, effect_up, effect_down=None, scale=None):
        """
        Set the effect of a parameter on a sample (e.g. the size of unc. or multiplier for shape unc.)
        param: a Parameter object
        effect_up: a numpy array representing the relative (multiplicative) effect of the parameter on the bin yields,
                   or a single number representing the relative effect on the sample normalization,
                   or a histogram representing the *bin yield* under the effect of the parameter (i.e. not relative)
                   or a DependentParameter representing the value to scale the *normalization* of this process
        effect_down: if asymmetric effects, fill this in, otherwise the effect_up value will be symmetrized
        scale : number, optional
            ad-hoc rescaling of the effect, most useful for shape effects where the nuisance parameter effect needs to be
            magnified to ensure good vertical interpolation

        N.B. the parameter must have a compatible combinePrior, i.e. if param.combinePrior is 'shape', then one must pass a numpy array
        """
        if not isinstance(param, NuisanceParameter):
            if isinstance(param, IndependentParameter) and isinstance(effect_up, DependentParameter):
                extras = effect_up.getDependents() - {param}
                if not all(isinstance(p, IndependentParameter) for p in extras):
                    raise ValueError("Normalization effects can only depend on one or more IndependentParameters")
                self._extra_dependencies.update(extras)
                for extra in extras:
                    self._paramEffectsUp[extra] = None
                if effect_down is not None:
                    raise ValueError("Asymmetric normalization modifiers not supported. You can encode the effect in the dependent parameter")
                effect_up.name = param.name + "_effect_" + self.name
                self._paramEffectsUp[param] = effect_up
                return
            else:
                raise ValueError("Template morphing can only be done via a NuisanceParameter or IndependentParameter")
        if param.name in [p.name for p in self.parameters]:
            raise ValueError(f"Parameter '{param.name}' already exists in sample '{self.name}': {sorted([p.name for p in self.parameters])}")
        if isinstance(effect_up, np.ndarray):
            if len(effect_up) != self.observable.nbins:
                raise ValueError("effect_up has the wrong number of bins (%d, expected %d)" % (len(effect_up), self.observable.nbins))
        elif isinstance(effect_up, numbers.Number):
            if "shape" in param.combinePrior:
                effect_up = np.full(self.observable.nbins, effect_up)
        else:
            effect_up, binning, _ = _to_numpy(effect_up)
            if not np.array_equal(binning, self.observable.binning):
                raise ValueError("effect_up has incompatible binning with sample %r" % self)
            zerobins = (self._nominal <= 0.0) | (effect_up <= 0.0)
            effect_up[zerobins] = 1.0
            effect_up[~zerobins] /= self._nominal[~zerobins]
        if np.sum(effect_up * self._nominal) <= 0:
            # TODO: warning? this can happen regularly
            # we might even want some sort of threshold
            return
        elif effect_down is None and np.all(effect_up == 1.0):
            # some sort of threshold might be useful here as well
            return
        _weighted_effect_magnitude = np.sum(abs(effect_up - 1) * self._nominal) / np.sum(self._nominal)
        if "shape" in param.combinePrior and _weighted_effect_magnitude > 0.5:
            print(
                "effect_up ({}, {}) has magnitude greater than 50% ({:.2f}%), you might be passing absolute values instead of relative".format(
                    param.name, self._name, _weighted_effect_magnitude * 100
                )
            )
        self._paramEffectsUp[param] = effect_up

        if effect_down is not None:
            if isinstance(effect_down, np.ndarray):
                if len(effect_down) != self.observable.nbins:
                    raise ValueError("effect_down has the wrong number of bins (%d, expected %d)" % (len(effect_down), self.observable.nbins))
            elif isinstance(effect_down, numbers.Number):
                if "shape" in param.combinePrior:
                    effect_down = np.full(self.observable.nbins, effect_down)
            else:
                effect_down, binning, _ = _to_numpy(effect_down)
                if not np.array_equal(binning, self.observable.binning):
                    raise ValueError("effect_down has incompatible binning with sample %r" % self)
                zerobins = (self._nominal <= 0.0) | (effect_down <= 0.0)
                effect_down[zerobins] = 1.0
                effect_down[~zerobins] /= self._nominal[~zerobins]
                if np.sum(effect_down * self._nominal) <= 0:
                    # TODO: warning? this can happen regularly
                    # we might even want some sort of threshold
                    return
                elif np.all(effect_up == 1.0) and np.all(effect_down == 1.0):
                    # some sort of threshold might be useful here as well
                    return
            _weighted_effect_magnitude = np.sum(abs(effect_down - 1) * self._nominal) / np.sum(self._nominal)
            if "shape" in param.combinePrior and _weighted_effect_magnitude > 0.5:
                print(
                    "effect_down ({}, {}) has magnitude greater than 50% ({:.2f}%), you might be passing absolute values instead of relative".format(
                        param.name, self._name, _weighted_effect_magnitude * 100
                    )
                )
            self._paramEffectsDown[param] = effect_down
        else:
            self._paramEffectsDown[param] = None

        if isinstance(scale, numbers.Number):
            if isinstance(effect_up, DependentParameter):
                raise ValueError("Scale not supported for DependentParameter effects. You can encode the effect in the dependent parameter")
            self._paramEffectScales[param] = scale
        elif scale is not None:
            raise ValueError("Cannot understand scale value %r. It should be a number" % scale)

    def getParamEffect(self, param, up=True):
        """
        Get the parameter effect
        """
        if up:
            return self._paramEffectsUp[param]
        else:
            if param not in self._paramEffectsDown or self._paramEffectsDown[param] is None:
                # TODO the symmetrized value depends on if param prior is 'shapeN' or 'shape'
                if param.combinePrior == "lnN":
                    return 1.0 / self._paramEffectsUp[param]
                elif param.combinePrior == "shape":
                    return self._nominal - abs(self._nominal - self._paramEffectsUp[param])
                else:
                    raise NotImplementedError
            return self._paramEffectsDown[param]

    def autoMCStats(self, lnN=False, epsilon=0, threshold=0, sample_name=None, bini=None):
        """
        Set MC statical uncertainties based on self._sumw2. `sample_name` and `bini` parameters
        don't need to modified for typical use cases.

        lnN: aggregate differences
        epsilon: 0 -> epsilon, is only one bin is filled lower syst of 0, gives empty norm
        threshold: if relative uncertainty is < threshold, won't be added (only for lnN = False)
        sample_name: custom name for e.g. using same parameters in two regions. Uses ``self.name``
            by default (if sample_name=None).
        bini: create parameter for a specific bin. By default creates for all (if bin=None)
            (only if lnN = False).
        """

        if self._sumw2 is None:
            raise ValueError("No self._sumw2 defined in template")
            return

        name = self._name if sample_name is None else sample_name

        if lnN:
            if threshold > 0:
                raise ValueError("No threshold implemented for lnN stat uncertainty")
            if bini is not None:
                raise ValueError("Bin-specific uncertainty not implemented for lnN stat uncertainty")

            _nom_rate = np.sum(self._nominal)
            if _nom_rate < 0.0001:
                effect = 1.0
            else:
                _down_rate = np.sum(np.nan_to_num(self._nominal - np.sqrt(self._sumw2), 0.0))
                _up_rate = np.sum(np.nan_to_num(self._nominal + np.sqrt(self._sumw2), 0.0))
                _diff = np.abs(_up_rate - _nom_rate) + np.abs(_down_rate - _nom_rate)
                effect = 1.0 + _diff / (2.0 * _nom_rate)
            param = NuisanceParameter(name + "_mcstat", "lnN")
            self.setParamEffect(param, effect)
        else:
            if bini is not None:
                assert bini >= 0 and bini < self.observable.nbins, "autoMCStats bini %d out of range for sample %r " % (bini, self)

            for i in range(self.observable.nbins):
                if bini is not None and bini != i:
                    continue
                if self._nominal[i] <= 0.0 or self._sumw2[i] <= 0.0:
                    continue
                effect_up = np.ones_like(self._nominal)
                effect_down = np.ones_like(self._nominal)

                if (np.sqrt(self._sumw2[i]) / (self._nominal[i] + 1e-12)) < threshold:
                    continue

                effect_up[i] = (self._nominal[i] + np.sqrt(self._sumw2[i])) / self._nominal[i]
                effect_down[i] = max((self._nominal[i] - np.sqrt(self._sumw2[i])) / self._nominal[i], epsilon)
                param = NuisanceParameter(name + "_mcstat_bin%i" % i, combinePrior="shape")
                self.setParamEffect(param, effect_up, effect_down)

    def getExpectation(self, nominal=False, eval=False):
        """
        Create an array of per-bin expectations, accounting for all nuisance parameter effects
            nominal: if True, calculate the nominal expectation (i.e. just plain numbers)
            eval: if True, calculate the expectation, based on current parameter values
        """

        nominalval = self._nominal.copy()
        if self.mask is not None:
            nominalval[~self.mask] = self._mask_val
        if nominal:
            return nominalval
        else:
            out = np.array([IndependentParameter(self.name + "_bin%d_nominal" % i, v, constant=True) for i, v in enumerate(nominalval)])
            for param in self.parameters:
                effect_up = self.getParamEffect(param, up=True)
                if effect_up is None:
                    continue
                if param in self._paramEffectScales:
                    param_scaled = param * self._paramEffectScales[param]
                else:
                    param_scaled = param
                if isinstance(effect_up, DependentParameter):
                    out = out * effect_up
                elif self._paramEffectsDown[param] is None:
                    if param.combinePrior == "shape":
                        out = out * (1 + (effect_up - 1) * param_scaled)
                    elif param.combinePrior == "shapeN":
                        out = out * (effect_up**param_scaled)
                    elif param.combinePrior == "lnN":
                        # TODO: ensure scalar effect
                        out = out * (effect_up**param_scaled)
                    else:
                        raise NotImplementedError("per-bin effects for other nuisance parameter types")
                else:
                    effect_down = self.getParamEffect(param, up=False)
                    smoothStep = SmoothStep(param_scaled)
                    if param.combinePrior == "shape":
                        combined_effect = smoothStep * (1 + (effect_up - 1) * param_scaled) + (1 - smoothStep) * (1 - (effect_down - 1) * param_scaled)
                    elif param.combinePrior == "shapeN":
                        combined_effect = smoothStep * (effect_up**param_scaled) + (1 - smoothStep) / (effect_down**param_scaled)
                    elif param.combinePrior == "lnN":
                        # TODO: ensure scalar effect
                        combined_effect = smoothStep * (effect_up**param_scaled) + (1 - smoothStep) / (effect_down**param_scaled)
                    else:
                        raise NotImplementedError("per-bin effects for other nuisance parameter types")
                    out = out * combined_effect

            if eval:
                return np.array([p.value for p in out])
            else:
                return out

    def renderRoofit(self, workspace):
        """
        Import the necessary Roofit objects into the workspace for this sample
        and return an extended pdf representing this sample's prediction for pdf and norm.
        """
        import ROOT

        install_roofit_helpers()
        normName = self.name + "_norm"
        rooShape = workspace.pdf(self.name)
        rooNorm = workspace.function(normName)
        if rooShape == None and rooNorm == None:  # noqa: E711
            rooObservable = self.observable.renderRoofit(workspace)
            nominal = self.getExpectation(nominal=True)
            rooTemplate = ROOT.RooDataHist(self.name, self.name, ROOT.RooArgList(rooObservable), _to_TH1(nominal, self.observable.binning, self.observable.name))
            workspace.add(rooTemplate)
            for param in self.parameters:
                effect_up = self.getParamEffect(param, up=True)
                if "shape" not in param.combinePrior:
                    # Normalization systematics can just go into combine datacards (although if we build PDF here, will need it)
                    if isinstance(effect_up, DependentParameter):
                        # this is a rateParam, we should add the IndependentParameter to the workspace
                        param.renderRoofit(workspace)
                    continue
                name = self.name + "_" + param.name + "Up"
                shape = nominal * effect_up
                rooTemplate = ROOT.RooDataHist(name, name, ROOT.RooArgList(rooObservable), _to_TH1(shape, self.observable.binning, self.observable.name))
                workspace.add(rooTemplate)
                name = self.name + "_" + param.name + "Down"
                shape = nominal * self.getParamEffect(param, up=False)
                rooTemplate = ROOT.RooDataHist(name, name, ROOT.RooArgList(rooObservable), _to_TH1(shape, self.observable.binning, self.observable.name))
                workspace.add(rooTemplate)

            rooShape = ROOT.RooHistPdf(self.name, self.name, ROOT.RooArgSet(rooObservable), workspace.data(self.name))
            workspace.add(rooShape)
            rooNorm = IndependentParameter(normName, nominal.sum(), constant=True).renderRoofit(workspace)
            # TODO build the pdf with systematics
        elif rooShape == None or rooNorm == None:  # noqa: E711
            raise RuntimeError("Sample %r has either a shape or norm already embedded in workspace %r" % (self, workspace))
        rooShape = workspace.pdf(self.name)
        rooNorm = workspace.function(self.name + "_norm")
        return rooShape, rooNorm

    def combineNormalization(self):
        return self.getExpectation(nominal=True).sum()

    def combineParamEffect(self, param):
        """
        A formatted string for placement into the combine datacard that represents
        the effect of a parameter on a sample (e.g. the size of unc. or multiplier for shape unc.)
        """
        if self._paramEffectsUp.get(param, None) is None:
            return "-"
        elif "shape" in param.combinePrior:
            return "%.4f" % self._paramEffectScales.get(param, 1)
        elif isinstance(self.getParamEffect(param, up=True), DependentParameter):
            # about here's where I start to feel painted into a corner
            dep = self.getParamEffect(param, up=True)
            channel, sample = self.name[: self.name.find("_")], self.name[self.name.find("_") + 1 :]
            dependents = dep.getDependents()
            formula = dep.formula(rendering=True).format(**{var.name: "@%d" % i for i, var in enumerate(dependents)})
            return "{0} rateParam {1} {2} {3} {4}".format(
                dep.name,
                channel,
                sample,
                formula,
                ",".join(p.name for p in dependents),
            )
        else:
            # TODO the scaling here depends on the prior of the nuisance parameter
            scale = self._paramEffectScales.get(param, 1.0)
            up = (self.getParamEffect(param, up=True) - 1) * scale + 1
            down = (self.getParamEffect(param, up=False) - 1) * scale + 1
            if isinstance(up, np.ndarray):
                # Convert shape to norm (note symmetrized effect on shape != symmetrized effect on norm)
                nominal = self.getExpectation(nominal=True)
                if nominal.sum() == 0:
                    up = 1.0
                    down = None
                else:
                    up = (up * nominal).sum() / nominal.sum()
                    down = (down * nominal).sum() / nominal.sum()
            elif self._paramEffectsDown[param] is None:
                # Here we can safely defer to combine to calculate symmetrized effect
                down = None
            if down is None:
                return "%.4f" % up
            else:
                return "%.4f/%.4f" % (down, up)


class ParametericSample(Sample):
    PreferRooParametricHist = True

    def __init__(self, name, sampletype, observable, params):
        """
        Create a sample that is a binned function, where each bin yield
        is given by the param in params.  The list params should have the
        same number of bins as observable.
        """
        super(ParametericSample, self).__init__(name, sampletype)
        if not isinstance(observable, Observable):
            raise ValueError
        if len(params) != observable.nbins:
            raise ValueError
        self._observable = observable
        self._nominal = np.array(params)
        if not all(isinstance(p, Parameter) for p in self._nominal):
            raise ValueError("ParametericSample expects parameters to derive from Parameter type.")
        self._paramEffectsUp = {}
        self._paramEffectsDown = {}

    @property
    def parameters(self):
        """
        Set of independent parameters that affect this sample
        """
        pset = set()
        for p in self.getExpectation():
            pset.update(p.getDependents(deep=True))
        return pset

    def setParamEffect(self, param, effect_up, effect_down=None):
        """
        Set the effect of a parameter on a sample (e.g. the size of unc. or multiplier for shape unc.)
        param: a Parameter object
        effect_up: a numpy array representing the relative (multiplicative) effect of the parameter on the bin yields,
                   or a single number representing the relative effect on the sample normalization,
        effect_down: if asymmetric effects, fill this in, otherwise the effect_up value will be symmetrized

        N.B. the parameter must have a compatible combinePrior, i.e. if param.combinePrior is 'shape', then one must pass a numpy array
        """
        if not isinstance(param, NuisanceParameter):
            raise ValueError("Template morphing can only be done via a NuisanceParameter")
        if param.name in [p.name for p in self.parameters]:
            raise ValueError(f"Parameter '{param.name}' already exists in sample '{self.name}': {sorted([p.name for p in self.parameters])}")
        if isinstance(effect_up, np.ndarray):
            if len(effect_up) != self.observable.nbins:
                raise ValueError("effect_up has the wrong number of bins (%d, expected %d)" % (len(effect_up), self.observable.nbins))
        elif isinstance(effect_up, numbers.Number):
            if "shape" in param.combinePrior:
                effect_up = np.full(self.observable.nbins, effect_up)
        else:
            raise ValueError("unrecognized effect_up type")
        self._paramEffectsUp[param] = effect_up

        if effect_down is not None:
            if isinstance(effect_down, np.ndarray):
                if len(effect_down) != self.observable.nbins:
                    raise ValueError("effect_down has the wrong number of bins (%d, expected %d)" % (len(effect_down), self.observable.nbins))
            elif isinstance(effect_down, numbers.Number):
                if "shape" in param.combinePrior:
                    effect_down = np.full(self.observable.nbins, effect_down)
            else:
                raise ValueError("unrecognized effect_down type")
            self._paramEffectsDown[param] = effect_down
        else:
            self._paramEffectsDown[param] = None

    def getParamEffect(self, param, up=True):
        """
        Get the parameter effect
        """
        if up:
            return self._paramEffectsUp[param]
        else:
            if self._paramEffectsDown[param] is None:
                # TODO the symmetrized value depends on if param prior is 'shapeN' or 'shape'
                return 1.0 / self._paramEffectsUp[param]
            return self._paramEffectsDown[param]

    def getExpectation(self, nominal=False):
        """
        Create an array of per-bin expectations, accounting for all nuisance parameter effects
            nominal: if True, calculate the nominal expectation (i.e. just plain numbers)
        """

        out = self._nominal.copy()  # this is a shallow copy
        if self.mask is not None:
            out[~self.mask] = [IndependentParameter("masked", self._mask_val, constant=True) for _ in range((~self.mask).sum())]
        if nominal:
            return np.array([p.value for p in out])
        else:
            for param in self._paramEffectsUp.keys():
                effect_up = self.getParamEffect(param, up=True)
                if effect_up is None:
                    pass
                if self._paramEffectsDown[param] is None:
                    out = out * (effect_up**param)
                else:
                    effect_down = self.getParamEffect(param, up=False)
                    smoothStep = SmoothStep(param)
                    combined_effect = smoothStep * (effect_up**param) + (1 - smoothStep) * (effect_down**param)
                    out = out * combined_effect

            for i, p in enumerate(out):
                p.name = self.name + "_bin%d" % i
                if isinstance(p, DependentParameter):
                    # Let's make sure to render these
                    p.intermediate = False

            return out

    def renderRoofit(self, workspace):
        """
        Produce a RooParametricHist (if available) or RooParametricStepFunction and add to workspace
        Note: for RooParametricStepFunction, bin values cannot be zero due to this ridiculous line:
        https://github.com/root-project/root/blob/master/roofit/roofit/src/RooParametricStepFunction.cxx#L212-L213
        """
        import ROOT

        install_roofit_helpers()
        rooShape = workspace.pdf(self.name)
        rooNorm = workspace.function(self.name + "_norm")
        if rooShape == None and rooNorm == None:  # noqa: E711
            rooObservable = self.observable.renderRoofit(workspace)
            params = self.getExpectation()

            if hasattr(ROOT, "RooParametricHist") and self.PreferRooParametricHist:
                rooParams = [p.renderRoofit(workspace) for p in params]
                # need a dummy hist to generate proper binning
                dummyHist = _to_TH1(np.zeros(self.observable.nbins), self.observable.binning, self.observable.name)
                rooShape = ROOT.RooParametricHist(self.name, self.name, rooObservable, ROOT.RooArgList.fromiter(rooParams), dummyHist)
                rooNorm = ROOT.RooAddition(self.name + "_norm", self.name + "_norm", ROOT.RooArgList.fromiter(rooParams))
                workspace.add(rooShape)
                workspace.add(rooNorm)
            else:
                if self.PreferRooParametricHist:
                    warnings.warn(
                        "Could not load RooParametricHist, falling back to RooParametricStepFunction, which has strange rounding issues.\n"
                        "Set ParametericSample.PreferRooParametricHist = False to disable this warning",
                        RuntimeWarning,
                    )
                # RooParametricStepFunction expects parameters to represent PDF density (i.e. bin width normalized, and integrates to 1)
                norm = _pairwise_sum(params)
                norm.name = self.name + "_norm"
                norm.intermediate = False

                binw = np.diff(self.observable.binning)
                dparams = params / binw / norm

                for p, oldp in zip(dparams, params):
                    p.name = oldp.name + "_density"
                    p.intermediate = False

                # The last bin value is defined by 1 - sum(others), so no need to render it
                rooParams = [p.renderRoofit(workspace) for p in dparams[:-1]]
                rooShape = ROOT.RooParametricStepFunction(self.name, self.name, rooObservable, ROOT.RooArgList.fromiter(rooParams), self.observable.binningTArrayD(), self.observable.nbins)
                workspace.add(rooShape)
                rooNorm = norm.renderRoofit(workspace)  # already rendered but we want to return it
        elif rooShape == None or rooNorm == None:  # noqa: E711
            raise RuntimeError("Channel %r has either a shape or norm already embedded in workspace %r" % (self, workspace))
        rooShape = workspace.pdf(self.name)
        rooNorm = workspace.function(self.name + "_norm")
        return rooShape, rooNorm

    def combineNormalization(self):
        """
        For combine, the normalization in the card is used to scale the parametric process PDF
        Since we provide an explicit normalization function, this should always stay at 1.
        """
        # TODO: optionally we could set the normalization here and leave only normalization modifiers
        return 1.0

    def combineParamEffect(self, param):
        """
        Combine cannot build shape param effects for parameterized templates, so we have to do it in the model.
        """
        return "-"


class TransferFactorSample(ParametericSample):
    def __init__(self, name, sampletype, transferfactor, dependentsample, observable=None, min_val=None):
        """
        Create a sample that depends on another Sample by some transfer factor.
        The transfor factor can be a constant, an array of parameters of same length
        as the dependent sample binning, or a matrix of parameters where the second
        dimension matches the sample binning, i.e. expectation = tf @ dependent_expectation.
        The latter requires an additional observable argument to specify the definition of the first dimension.
        In all cases, please use numpy object arrays of Parameter types.
        Passing in a ``min_val`` means param values will be clipped at the min_val.
        """
        if not isinstance(transferfactor, np.ndarray):
            raise ValueError("Transfer factor is not a numpy array")
        if not isinstance(dependentsample, Sample):
            raise ValueError("Dependent sample does not inherit from Sample")
        if len(transferfactor.shape) == 2:
            if observable is None:
                raise ValueError("Transfer factor is 2D array, please provide an observable")
            params = np.dot(transferfactor, dependentsample.getExpectation())
            if min_val is not None:
                for idx, p in np.ndenumerate(params):
                    params[idx] = p.max(min_val)
        elif len(transferfactor.shape) <= 1:
            observable = dependentsample.observable
            params = transferfactor * dependentsample.getExpectation()
            if min_val is not None:
                for i, p in enumerate(params):
                    params[i] = p.max(min_val)
        else:
            raise ValueError("Transfer factor has invalid dimension")
        super(TransferFactorSample, self).__init__(name, sampletype, observable, params)
        self._transferfactor = transferfactor
        self._dependentsample = dependentsample

    @property
    def transferfactor(self):
        return self._transferfactor

    @property
    def dependentsample(self):
        return self._dependentsample
