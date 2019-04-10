import collections
import datetime
from functools import reduce
import os

import ROOT

from .process import Process


class Model(collections.abc.MutableMapping):
    """
    Model -> Channel -> Process
    """
    def __init__(self):
        self._channel = {}

    def __delitem__(self, key):
        self._channel[key].link(None)
        del self._channel[key]

    def __getitem__(self, key):
        if key in self._channel:
            return self._channel[key]
        else:
            raise KeyError(key)

    def __iter__(self):
        for item in self._channel:
            yield item

    def __len__(self):
        return len(self._channel)

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        if isinstance(value, Channel):
            self._channel[key] = value
            value.link(self, key)
        else:
            raise ValueError("Only Channel types can be attached to Model. Got: %r" % value)

    def __repr__(self):
        return "<Model instance at 0x%x>" % (
            id(self),
        )

    def render(self, outputPath, workspaceName='model', combined=False):
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        workspace = ROOT.RooWorkspace(workspaceName)
        for channel in self.values():
            channel.renderRoofitModel(workspace, workspaceName)
        workspace.writeToFile(os.path.join(outputPath, "%s.root" % workspaceName))
        if combined:
            self.renderCombinedCard(os.path.join(outputPath, "combinedCard.txt"))
        else:
            for name, channel in self.items():
                channel.renderCard(os.path.join(outputPath, "%s.txt" % name), workspaceName)

    def renderCombinedCard(self, outputFilename):
        raise NotImplementedError("For now, use combineCards.py")


class Channel(collections.abc.MutableMapping):
    """
    Channel -> Process
    """
    def __init__(self):
        self._process = {}
        self._link = (None, "undefined")

    def __delitem__(self, key):
        self._process[key].link(None)
        del self._process[key]

    def __getitem__(self, key):
        if key in self._process:
            return self._process[key]
        else:
            raise KeyError(key)

    def __iter__(self):
        for item in self._process:
            yield item

    def __len__(self):
        return len(self._process)

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        if isinstance(value, Process):
            self._process[key] = value
            value.link(self, key)
        else:
            raise ValueError("Only Process types can be attached to Channel. Got: %r" % value)

    def __repr__(self):
        return "<Channel instance (linked to %r as '%s') at 0x%x>" % (
            self._link[0],
            self._link[1],
            id(self),
        )

    def link(self, parent, key=''):
        self._link = (parent, key)

    @property
    def name(self):
        return self._link[1]

    @property
    def nuisanceParameters(self):
        return reduce(set.union, (p.nuisanceParameters for p in self.values()), set())

    @property
    def parameters(self):
        return reduce(set.union, (p.parameters for p in self.values()), set())

    def renderRoofitModel(self, workspace, workspaceName):
        for process in self.values():
            process.renderRoofitModel(workspace, workspaceName + '_' + self.name)

    def renderCard(self, outputFilename, workspaceName):
        if 'data_obs' not in self:
            raise RuntimeError("Missing 'data_obs' process in %r" % self)
        signalProcessNames = [p for p in self if self[p].processtype == Process.SIGNAL]
        signalProcessNames.sort()
        nSig = len(signalProcessNames)
        bkgProcessNames = [p for p in self if self[p].processtype == Process.BACKGROUND]
        bkgProcessNames.sort()
        nBkg = len(bkgProcessNames)

        with open(outputFilename, "w") as fout:
            fout.write("# Datacard for %r generated on %s\n" % (self, str(datetime.datetime.now())))
            fout.write("imax %d # number of categories ('bins' but here we are using shape templates)\n" % 1)
            fout.write("jmax %d # number of processes minus 1\n" % (nSig + nBkg - 1))
            fout.write("kmax %d # number of nuisance parameters\n" % len(self.nuisanceParameters))
            fout.write("shapes * {1} {0}.root {0}:{0}_{1}_$PROCESS {0}_{1}_$PROCESS_$SYSTEMATIC\n".format(workspaceName, self.name))
            fout.write("bin %s\n" % self.name)
            fout.write("observation %.3f\n" % self['data_obs'].normalization())
            table = []
            table.append(['bin'] + [self.name]*(nSig + nBkg))
            table.append(['process'] + signalProcessNames + bkgProcessNames)
            table.append(['process'] + [str(i) for i in range(1 - nSig, nBkg + 1)])
            table.append(['rate'] + ["%.3f" % self[p].normalization() for p in signalProcessNames + bkgProcessNames])
            for param in self.nuisanceParameters:
                table.append([str(param)] + [self[p].nuisanceParamString(param) for p in signalProcessNames + bkgProcessNames])
            colWidths = [max(len(table[row][col]) for row in range(len(table))) for col in range(nSig + nBkg + 1)]
            rowfmt = ("{:<%d}" % colWidths[0]) + " ".join("{:>%d}" % w for w in colWidths[1:]) + "\n"
            for row in table:
                fout.write(rowfmt.format(*row))
            for param in self.parameters:
                fout.write(str(param) + "\n")
