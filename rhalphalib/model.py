from collections import OrderedDict
import datetime
from functools import reduce
from itertools import chain
import os
import numpy as np
from .sample import Sample
from .parameter import Observable, IndependentParameter, NuisanceParameter
from .util import _to_numpy, _to_TH1, install_roofit_helpers


class Model(object):
    """
    Model -> Channel -> Sample
    """
    def __init__(self, name):
        self._name = name
        self._channels = OrderedDict()
        self.t2w_config = None

    def __getitem__(self, key):
        if key in self._channels:
            return self._channels[key]
        if '_' in key:
            channel = key[:key.find('_')]
            return self[channel][key]
        raise KeyError

    def __iter__(self):
        for item in self._channels.values():
            yield item

    def __len__(self):
        return len(self._channels)

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
    def channels(self):
        return self._channels.values()

    @property
    def parameters(self):
        return reduce(set.union, (c.parameters for c in self), set())

    def addChannel(self, channel):
        if not isinstance(channel, Channel):
            raise ValueError("Only Channel types can be attached to Model. Got: %r" % channel)
        if channel.name in self._channels:
            raise ValueError("Model %r already has a channel named %s" % (self, channel.name))
        self._channels[channel.name] = channel
        return self

    def readRooFitResult(self, res):
        '''
        Update all independent parameters with the values given in the fit result
        res: a RooFitResult object
        '''
        install_roofit_helpers()
        params = {p.name: p for p in self.parameters if isinstance(p, IndependentParameter)}
        for p_in in chain(res.floatParsFinal(), res.constPars()):
            if p_in.GetName() in params:
                p = params[p_in.GetName()]
                p.value = p_in.getVal()
                p.lo = p_in.getMin()
                p.hi = p_in.getMax()
                p.constant = p_in.isConstant()

    def renderRoofit(self, workspace):
        import ROOT
        install_roofit_helpers()
        pdfName = self.name + '_simPdf'
        dataName = self.name + '_observation'
        rooSimul = workspace.pdf(pdfName)
        rooData = workspace.data(dataName)
        if rooSimul == None and rooData == None:  # noqa: E711
            channelCat = ROOT.RooCategory(self.name + '_channel', self.name + '_channel')
            # TODO s+b, b-only separate?
            rooSimul = ROOT.RooSimultaneous(self.name + '_simPdf', self.name + '_simPdf', channelCat)
            obsmap = ROOT.std.map('string, RooDataHist*')()
            for channel in self:
                channelCat.defineType(channel.name)
                pdf, obs = channel.renderRoofit(workspace)
                rooSimul.addPdf(pdf, channel.name)
                # const string magic: https://root.cern.ch/phpBB3/viewtopic.php?f=15&t=16882&start=15#p86985
                obsmap.insert(ROOT.std.pair('const string, RooDataHist*')(channel.name, obs))

            workspace.add(rooSimul)
            rooObservable = ROOT.RooArgList(channel.observable.renderRoofit(workspace))
            # that's right I don't need no CombDataSetFactory
            rooData = ROOT.RooDataHist(self.name + '_observation', 'Combined observation', rooObservable, channelCat, obsmap)
            workspace.add(rooData)
        elif rooSimul == None or rooData == None:  # noqa: E711
            raise RuntimeError('Model %r has a pdf or dataset already embedded in workspace %r' % (self, workspace))
        rooSimul = workspace.pdf(pdfName)
        rooData = workspace.data(dataName)
        return rooSimul, rooData

    def renderCombine(self, outputPath):
        import ROOT
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        workspace = ROOT.RooWorkspace(self.name)
        self.renderRoofit(workspace)
        workspace.writeToFile(os.path.join(outputPath, "%s.root" % self.name))
        for channel in self:
            channel.renderCard(os.path.join(outputPath, "%s.txt" % channel.name), self.name)
        with open(os.path.join(outputPath, "build.sh"), "w") as fout:
            cstr = " ".join("{0}={0}.txt".format(channel.name) for channel in self)
            fout.write("combineCards.py %s > %s_combined.txt\n" % (cstr, "model"))
            if self.t2w_config is None:
                fout.write("text2workspace.py model_combined.txt")
            else:
                fout.write("text2workspace.py {} model_combined.txt".format(self.t2w_config))


class Channel(object):
    """
    Channel -> Sample
    """
    def __init__(self, name):
        self._name = name
        if '_' in self._name:
            raise ValueError("Naming convention restricts '_' characters in channel %r" % self)
        self._samples = OrderedDict()
        self._observable = None
        self._observation = None
        self._mask = None
        self._mask_val = 0.

    def __getitem__(self, key):
        if key in self._samples:
            return self._samples[key]
        elif self.name + '_' + key in self._samples:
            return self._samples[self.name + '_' + key]
        raise KeyError

    def __iter__(self):
        for item in self._samples.values():
            yield item

    def __len__(self):
        return len(self._samples)

    def addSample(self, sample):
        if not isinstance(sample, Sample):
            raise ValueError("Only Sample types can be attached to Channel. Got: %r" % sample)
        if sample.name in self._samples:
            raise ValueError("Channel %r already has a sample named %s" % (self, sample.name))
        if sample.name[:sample.name.find('_')] != self.name:
            raise ValueError("Naming convention requires begining of sample %r name to be %s" % (sample, self.name))
        if self._observable is not None:
            if not sample.observable == self._observable:
                raise ValueError("Sample %r has an incompatible observable with channel %r" % (sample, self))
            sample.observable = self._observable
        else:
            self._observable = sample.observable
        sample.mask = self.mask
        self._samples[sample.name] = sample

    def setObservation(self, obs, read_sumw2=False):
        '''
        Set the observation of the channel.
        obs: Either a ROOT TH1, a 1D Coffea Hist object, a 1D hist Hist object, or a numpy histogram
            in the latter case, please extend the numpy histogram tuple to define an observable name
            i.e. (sumw, binning, name)
            (for the others, the observable name is taken from the x axis name)
        read_sumw2: bool
            If true, don't assume observation is poisson, and read sumw2 for observation into model
        '''
        if read_sumw2:
            sumw, binning, obs_name, sumw2 = _to_numpy(obs, read_sumw2=True)
        else:
            sumw, binning, obs_name = _to_numpy(obs)
        observable = Observable(obs_name, binning)
        if self._observable is not None:
            if not observable == self._observable:
                raise ValueError("Observation has an incompatible observable with channel %r" % self)
        else:
            self._observable = observable
        if read_sumw2:
            self._observation = (sumw, sumw2)
        else:
            self._observation = sumw

    def getObservation(self):
        '''
        Return the current observation set for this Channel as plain numpy array
        If it is non-poisson, it will be returned as a tuple (sumw, sumw2)
        '''

        if self._observation is None:
            raise RuntimeError("Channel %r has no observation set" % self)
        if isinstance(self._observation, tuple):
            obs, var = (self._observation[0].copy(), self._observation[1].copy())
            if self.mask is not None:
                obs[~self.mask] = self._mask_val
                var[~self.mask] = self._mask_val
            return (obs, var)
        else:
            obs = self._observation.copy()
            if self.mask is not None:
                obs[~self.mask] = self._mask_val
            return obs

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
    def samples(self):
        return self._samples.values()

    @property
    def parameters(self):
        return reduce(set.union, (s.parameters for s in self), set())

    @property
    def observable(self):
        if self._observable is None:
            raise RuntimeError("No observable set for channel %r yet.  Add a sample or observation to set observable." % self)
        return self._observable

    @property
    def mask(self):
        '''
        An array matching the observable binning that specifies which bins to populate
        i.e. when mask[i] is False, the bin content for all samples and the observation will be set to 0
        Useful for blinding!
        '''
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
        for sample in self:
            sample.mask = self.mask

    def renderRoofit(self, workspace):
        '''
        Render each sample in the channel and add them into an extended RooAddPdf
        Also render the observation
        '''
        import ROOT
        install_roofit_helpers()
        dataName = self.name + '_data_obs'  # combine convention
        rooPdf = workspace.pdf(self.name)
        rooData = workspace.data(dataName)
        if rooPdf == None and rooData == None:  # noqa: E711
            pdfs = []
            norms = []
            for sample in self:
                pdf, norm = sample.renderRoofit(workspace)
                pdfs.append(pdf)
                norms.append(norm)

            rooPdf = ROOT.RooAddPdf(self.name, self.name, ROOT.RooArgList.fromiter(pdfs), ROOT.RooArgList.fromiter(norms))
            workspace.add(rooPdf)

            rooObservable = self.observable.renderRoofit(workspace)
            rooData = ROOT.RooDataHist(dataName, dataName, ROOT.RooArgList(rooObservable), _to_TH1(self.getObservation(), self.observable.binning, self.observable.name))
            workspace.add(rooData)
        elif rooPdf == None or rooData == None:  # noqa: E711
            raise RuntimeError('Channel %r has either a pdf or dataset already embedded in workspace %r' % (self, workspace))
        rooPdf = workspace.pdf(self.name)
        rooData = workspace.data(dataName)
        return rooPdf, rooData

    def renderCard(self, outputFilename, workspaceName):
        observation = self.getObservation()
        if isinstance(observation, tuple):
            observation = observation[0]
        signalSamples = [s for s in self if s.sampletype == Sample.SIGNAL]
        nSig = len(signalSamples)
        bkgSamples = [s for s in self if s.sampletype == Sample.BACKGROUND]
        nBkg = len(bkgSamples)

        params = self.parameters
        nuisanceParams = [p for p in params if p.hasPrior()]
        nuisanceParams.sort(key=lambda p: p.name)
        otherParams = [p for p in params if p not in nuisanceParams]
        otherParams.sort(key=lambda p: p.name)

        with open(outputFilename, "w") as fout:
            fout.write("# Datacard for %r generated on %s\n" % (self, str(datetime.datetime.now())))
            fout.write("imax %d # number of categories ('bins' but here we are using shape templates)\n" % 1)
            fout.write("jmax %d # number of samples minus 1\n" % (nSig + nBkg - 1))
            fout.write("kmax %d # number of nuisance parameters\n" % len(nuisanceParams))
            fout.write("shapes * {1} {0}.root {0}:{1}_$PROCESS {0}:{1}_$PROCESS_$SYSTEMATIC\n".format(workspaceName, self.name))
            fout.write("bin %s\n" % self.name)
            fout.write("observation %.3f\n" % observation.sum())
            table = []
            table.append(['bin'] + [self.name]*(nSig + nBkg))
            # combine calls 'sample' a 'process', here also we remove channel prefix
            table.append(['process'] + [s.name[s.name.find('_')+1:] for s in signalSamples + bkgSamples])
            table.append(['process'] + [str(i) for i in range(1 - nSig, nBkg + 1)])
            table.append(['rate'] + ["%.3f" % s.combineNormalization() for s in signalSamples + bkgSamples])

            # if a param with prior does not have any effect here, the effect must be embedded in a sample PDF
            # in that case, we declare it as 'param' later in the card
            nuisancesNoCardEffect = []
            for param in nuisanceParams:
                effects = [s.combineParamEffect(param) for s in signalSamples + bkgSamples]
                if all(e == '-' for e in effects):
                    nuisancesNoCardEffect.append(param)
                else:
                    table.append([param.name + ' ' + param.combinePrior] + effects)

            colWidths = [max(len(table[row][col]) + 1 for row in range(len(table))) for col in range(nSig + nBkg + 1)]
            rowfmt = ("{:<%d}" % colWidths[0]) + " ".join("{:>%d}" % w for w in colWidths[1:]) + "\n"
            for row in table:
                fout.write(rowfmt.format(*row))

            for param in nuisancesNoCardEffect:
                fout.write("{0} param 0 1\n".format(param.name))

            for param in otherParams:
                fout.write("{0} extArg {1}.root:{1}\n".format(param.name, workspaceName))

            # identify any normalization modifiers
            for param in otherParams:
                for sample in signalSamples + bkgSamples:
                    effect = sample.combineParamEffect(param)
                    if effect != '-':
                        fout.write(effect + "\n")

    def autoMCStats(self, epsilon=0, threshold=0, include_signal=0, channel_name=None):
        '''
        Barlow-Beeston-lite method i.e. single stats parameter for all processes per bin.
        Same general algorithm as described in
        https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/part2/bin-wise-stats/
        but *without the analytic minimisation*.
        `include_signal` only refers to whether signal stats are included in the *decision* to use bb-lite or not.
        '''
        if not len(self._samples):
            raise RuntimeError('Channel %r has no samples for which to run autoMCStats' % (self))

        name = self._name if channel_name is None else channel_name

        first_sample = self._samples[list(self._samples.keys())[0]]

        for i in range(first_sample.observable.nbins):
            ntot_bb, etot2_bb = 0, 0  # for the decision to use bblite or not
            ntot, etot2 = 0, 0  # for the bblite uncertainty

            # check if neff = ntot^2 / etot2 > threshold
            for sample in self._samples.values():
                ntot += sample._nominal[i]
                etot2 += sample._sumw2[i]

                if not include_signal and sample._sampletype == Sample.SIGNAL:
                    continue

                ntot_bb += sample._nominal[i]
                etot2_bb += sample._sumw2[i]

            if etot2 <= 0.:
                continue
            elif etot2_bb <= 0:
                # this means there is signal but no background, so create stats unc. for signal only
                for sample in self._samples.values():
                    if sample._sampletype == Sample.SIGNAL:
                        sample_name = None if channel_name is None else channel_name + "_" + sample._name[sample._name.find('_') + 1:]
                        sample.autoMCStats(epsilon=epsilon, sample_name=sample_name, bini=i)

                continue

            neff_bb = ntot_bb ** 2 / etot2_bb
            if neff_bb <= threshold:
                for sample in self._samples.values():
                    sample_name = None if channel_name is None else channel_name + "_" + sample._name[sample._name.find('_') + 1:]
                    sample.autoMCStats(epsilon=epsilon, sample_name=sample_name, bini=i)
            else:
                effect_up = np.ones_like(first_sample._nominal)
                effect_down = np.ones_like(first_sample._nominal)

                effect_up[i] = (ntot + np.sqrt(etot2)) / ntot
                effect_down[i] = max((ntot - np.sqrt(etot2)) / ntot, epsilon)

                param = NuisanceParameter(name + '_mcstat_bin%i' % i, combinePrior='shape')

                for sample in self._samples.values():
                    sample.setParamEffect(param, effect_up, effect_down)
