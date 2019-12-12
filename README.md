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
Following the [recipe](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/#cc7-release-cmssw_10_2_x-recommended-version) from combine. and clone to CMSSW environment.
```
cd CMSSW_10_2_13/src
cmsenv
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
cd $CMSSW_BASE/src/HiggsAnalysis/CombinedLimit
git fetch origin
git checkout v8.0.1
scramv1 b clean; scramv1 b

git clone https://github.com/cms-analysis/CombineHarvester.git CombineHarvester
scram b -j8

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

# Go to hxxModel/ and source cmsenv to get combine
cmsenv
bash build.sh
combine -M FitDiagnostics hxxModel_combined.root
combine -M FitDiagnostics tempModel_combined.root --saveNormalizations --saveShapes --setParameterRanges r=-1,3

python ../HiggsAnalysis/CombinedLimit/test/diffNuisances.py tempModel/fitDiagnostics.root 

# To extract shapes/norms use combine harvester
PostFitShapesFromWorkspace -w tempModel_combined.root -o shapes.root --print --postfit --sampling -f fitDiagnostics.root:fit_s

combine -M Significance tempModel_combined.root
```
And back in conda env:
```
python plot.py -i tempModel/shapes.root --data --fit postfit
python plotTF.py -i tempModel/shapes.root --fit tempModel/fitDiagnostics.root
```


### OTHER IMPORTANT COMMANDS
```
combineTool.py -M FitDiagnostics -m 125 -d tempModel_combined.root --there --cminDefaultMinimizerStrategy 0 -t -1 --expectSignal 1
combineTool.py -M AsymptoticLimits -m 125 -d tempModel_combined.root --there -t -1 --expectSignal 1 --rMin=-50 --rMax=50
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
