from collections import OrderedDict
import datetime
from functools import reduce
import os
from .sample import Sample
from .parameter import Observable
from .util import _to_numpy, _to_TH1, install_roofit_helpers


class Model(object):
    """
    Model -> Channel -> Sample
    """
    def __init__(self, name):
        self._name = name
        self._channels = OrderedDict()

    def __getitem__(self, key):
        if key in self._channels:
            return self._channels[key]
        channel = key[:key.find('_')]
        return self[channel][key]

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

    def renderCombine(self, outputPath):
        import ROOT
        install_roofit_helpers()
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        workspace = ROOT.RooWorkspace(self.name)

        # TODO: build RooSimultaneus from channels here
        pdfs = []
        for channel in self:
            channelpdf = channel.renderRoofit(workspace)
            pdfs.append(channelpdf)

        # simul = ROOSimultaneuous(...)
        # workspace.add(simul)

        workspace.writeToFile(os.path.join(outputPath, "%s.root" % self.name))
        for channel in self:
            channel.renderCard(os.path.join(outputPath, "%s.txt" % channel.name), self.name)


class Channel():
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

    def __getitem__(self, key):
        return self._samples[key]

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
            if sample.observable != self._observable:
                raise ValueError("Sample %r has an incompatible observable with channel %r" % (sample, self))
            sample.observable = self._observable
        else:
            self._observable = sample.observable
        self._samples[sample.name] = sample

    def setObservation(self, obs):
        '''
        Set the observation of the channel.
        obs: Either a ROOT TH1, a 1D Coffea Hist object, or a numpy histogram
            in the latter case, please extend the numpy histogram tuple to define an observable name
            i.e. (sumw, binning, name)
            (for the others, the observable name is taken from the x axis name)
        '''
        sumw, binning, obs_name = _to_numpy(obs)
        observable = Observable(obs_name, binning)
        if self._observable is not None:
            if observable != self._observable:
                raise ValueError("Observation has an incompatible observable with channel %r" % self)
        else:
            self._observable = observable
        self._observation = sumw

    def getObservation(self):
        '''
        Return the current observation set for this Channel as plain numpy array
        '''
        if self._observation is None:
            raise RuntimeError("Channel %r has no observation set" % self)
        return self._observation

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

    def renderRoofit(self, workspace):
        '''
        Render each sample in the channel and add them into an extended RooAddPdf
        Also render the observation
        '''
        import ROOT
        # TODO: build RooAddPdf from sample pdfs (and norms)
        pdfs = []
        norms = []
        for sample in self:
            pdf, norm = sample.renderRoofit(workspace)
            pdfs.append(pdf)
            norms.append(norm)

        # addpdf = ROOT.RooAddPdf(self.name, self.name, ROOT.RooArgList(pdfs)...)
        # return addpdf

        rooObservable = self.observable.renderRoofit(workspace)
        name = self.name + '_data_obs'  # combine convention
        rooTemplate = ROOT.RooDataHist(name, name, ROOT.RooArgList(rooObservable), _to_TH1(self.getObservation(), self.observable.binning, self.observable.name))
        workspace.add(rooTemplate)

        return None

    def renderCard(self, outputFilename, workspaceName):
        observation = self.getObservation()
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
            table.append(['rate'] + ["%.3f" % s.normalization() for s in signalSamples + bkgSamples])
            for param in nuisanceParams:
                table.append([param.name + ' ' + param.combinePrior] + [s.combineParamEffect(param) for s in signalSamples + bkgSamples])

            colWidths = [max(len(table[row][col]) + 1 for row in range(len(table))) for col in range(nSig + nBkg + 1)]
            rowfmt = ("{:<%d}" % colWidths[0]) + " ".join("{:>%d}" % w for w in colWidths[1:]) + "\n"
            for row in table:
                fout.write(rowfmt.format(*row))
            for param in otherParams:
                table.append([param.name, param.combinePrior])
                fout.write(param.name + ' ' + param.combinePrior + "\n")
