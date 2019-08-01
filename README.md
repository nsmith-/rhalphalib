# rhalphalib

![Ralph](https://upload.wikimedia.org/wikipedia/en/thumb/1/14/Ralph_Wiggum.png/220px-Ralph_Wiggum.png)

## Quickstart
```
source /cvmfs/sft.cern.ch/lcg/views/LCG_95apython3/x86_64-slc6-gcc8-opt/setup.sh
git clone git@github.com:nsmith-/rhalphalib.git
cd rhalphalib
python test_rhalphalib.py
```
Take a look at the folders `testModel` and `monojetModel`.

## Requirements
Standalone model creation requires:
  - Python 3
  - `numpy >= 1.14`

RooFit+combine rendering requires:
  - `ROOT < 6.18` (i.e. LCG96 is too recent, combine+CMSSW8 cannot handle it.  LCG95a is fine)

Use in combine requires, well, [combine](https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit).
There is a python 3 compatible standalone fork of combine [available](https://github.com/guitargeek/combine),
however it is also possible to render the model folder, and then move the folder or switch environments to a CMSSW+combine environment and proceed from there.
