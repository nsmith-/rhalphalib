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

### Hcc
Follow the [recipe](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/#cc7-release-cmssw_10_2_x-recommended-version) from combine. and clone to CMSSW environment.
```
cd CMSSW_10_2_13/src
cmsenv
```
Then ideally in a separte window (no `cmsenv`) if you don't have conda setup, install conda (will manage all your packages, needs few GB of space)
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Setup a new environment to keep pacakges separate. Root needs to be 6.16
```
conda create -n myenv python
conda activate myenv
conda config --add channels conda-forge
conda install root==6.16
```

```
git clone git@github.com:andrzejnovak/rhalphalib.git
cd rhalphalib
git fetch
git checkout -b origin/hxx
python make_Hxx.py

# Go to hxxModel/ and sourc cmsenv to get combine 
cmsenv
bash build.sh
combine -M FitDiagnostics hxxModel_combined.root
```

## Requirements
Standalone model creation requires:
  - Python 2.7+ or 3.6+
  - `numpy >= 1.14`

RooFit+combine rendering requires:
  - `ROOT < 6.18` (i.e. LCG96 is too recent, CMSSW 8 combine cannot handle it.  LCG95a is fine)

Use in combine requires, well, [combine](https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit).
The CMSSW 10 (CC7) [recipe](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/#cc7-release-cmssw_10_2_x-recommended-version)
satisfies the requirements, however the CMSSW 8 recipe has a too old version of numpy.

There is a python 3 compatible standalone fork of combine [available](https://github.com/guitargeek/combine).
It is also possible to render the model folder using the quickstart recipe, and then move the folder or switch
environments to a CMSSW+combine environment and proceed from there.
