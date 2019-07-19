from collections import OrderedDict
import datetime
from functools import reduce
import os

import ROOT

from .sample import Sample


class Model(object):
    """
    Model -> Channel -> Sample
    """
    def __init__(self, name):
        self._name = name
        self._channels = OrderedDict()

    def __getitem__(self, key):
        return self._channels[key]

    def __iter__(self):
        for item in self._channels.values():
            yield item

    def __len__(self):
        return len(self._channels)

    def __repr__(self):
        return "<Model instance at 0x%x>" % (
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

    def renderWorkspace(self, outputPath, combined=False):
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        workspace = ROOT.RooWorkspace(self.name)

        # TODO: build RooSimultaneus from channels here
        pdfs = []
        for channel in self:
            channelpdf = channel.renderRoofitModel(workspace)
            pdfs.append(channelpdf)

        # simul = ROOSimultaneuous(...)
        # workspace.add(simul)

        workspace.writeToFile(os.path.join(outputPath, "%s.root" % self.name))
        if combined:
            self.renderCombinedCard(os.path.join(outputPath, "combinedCard.txt"))
        else:
            for channel in self:
                channel.renderCard(os.path.join(outputPath, "%s.txt" % channel.name), self.name)

    def renderCombinedCard(self, outputFilename):
        raise NotImplementedError("For now, use combineCards.py")


class Channel():
    """
    Channel -> Sample
    """
    def __init__(self, name):
        self._name = name
        self._samples = OrderedDict()

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
        if len(self._samples) > 0:
            if sample.observable != self._samples[0].observable:
                raise ValueError("Sample %r has an incompatible observable with other smaples in channel %r" % (sample, self))
            sample.observable = self._samples[0].observable
        else:
            sample.observable._attached = True  # FIXME setter in observable?
        self._samples[sample.name] = sample

    def __repr__(self):
        return "<Channel instance at 0x%x>" % (
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

    def renderRoofitModel(self, workspace):
        '''
        Render each sample in the channel and add them into an extended RooAddPdf
        '''
        # TODO: build RooAddPdf from sample pdfs (and norms)
        pdfs = []
        for sample in self:
            pdf = sample.renderRoofitModel(workspace)  # pdf, norm (or can pdf be already extended) ?
            pdfs.append(pdf)

        # addpdf = ROOT.RooAddPdf(self.name, self.name, ROOT.RooArgList(pdfs)...)
        # return addpdf
        return None

    def renderCard(self, outputFilename, workspaceName):
        observation = [s for s in self if s.sampletype == Sample.OBSERVATION]
        if len(observation) == 0:
            raise RuntimeError("Channel %r has no observation attached to it")
        if len(observation) > 1:
            raise RuntimeError("Channel %r has more than one observation attached to it")
        observation = observation[0]
        signalSamples = [s for s in self if s.sampletype == Sample.SIGNAL]
        nSig = len(signalSamples)
        bkgSamples = [s for s in self if s.sampletype == Sample.BACKGROUND]
        nBkg = len(bkgSamples)

        params = self.parameters
        nuisanceParams = [p for p in params if p.hasPrior()]
        otherParams = [p for p in params if p not in nuisanceParams]

        with open(outputFilename, "w") as fout:
            fout.write("# Datacard for %r generated on %s\n" % (self, str(datetime.datetime.now())))
            fout.write("imax %d # number of categories ('bins' but here we are using shape templates)\n" % 1)
            fout.write("jmax %d # number of samples minus 1\n" % (nSig + nBkg - 1))
            fout.write("kmax %d # number of nuisance parameters\n" % len(self.nuisanceParameters))
            fout.write("shapes * {1} {0}.root {0}:{0}_{1}_$PROCESS {0}_{1}_$PROCESS_$SYSTEMATIC\n".format(workspaceName, self.name))
            fout.write("bin %s\n" % self.name)
            fout.write("observation %.3f\n" % observation.normalization())
            table = []
            table.append(['bin'] + [self.name]*(nSig + nBkg))
            table.append(['sample'] + [s.name for s in signalSamples + bkgSamples])
            table.append(['sample'] + [str(i) for i in range(1 - nSig, nBkg + 1)])
            table.append(['rate'] + ["%.3f" % s.normalization() for s in signalSamples + bkgSamples])
            for param in nuisanceParams:
                table.append([param.name, param.combineType()] + [s.combineParamEffect(param) for s in signalSamples + bkgSamples])

            colWidths = [max(len(table[row][col]) for row in range(len(table))) for col in range(nSig + nBkg + 1)]
            rowfmt = ("{:<%d}" % colWidths[0]) + " ".join("{:>%d}" % w for w in colWidths[1:]) + "\n"
            for row in table:
                fout.write(rowfmt.format(*row))
            for param in otherParams:
                table.append([param.name, param.combineType()])
                fout.write(param.name + " " + param.combineType() + "\n")
