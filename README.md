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
export SCRAM_ARCH=slc7_amd64_gcc700
cmsrel CMSSW_10_2_13
cd CMSSW_10_2_13/src
cmsenv
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
cd $CMSSW_BASE/src/HiggsAnalysis/CombinedLimit
git fetch origin
git checkout v8.0.1
scramv1 b clean; scramv1 b

cd $CMSSW_BASE/src/
git clone https://github.com/cms-analysis/CombineHarvester.git CombineHarvester
scram b -j8
```
Then get the workspace maker

```
cmsenv
cd $CMSSW_BASE/src/
git clone git@github.com:andrzejnovak/rhalphalib.git
cd rhalphalib
git fetch
git checkout origin/hxxdev
# Need to update some packages against the ones in CMSSW (might need a few more)
pip install uproot --user --upgrade
pip install matplotlib --user --upgrade
pip install mplhep --user
# Run
python temp_Hxx.py # Must chose --data or --MC, other options get printed

# Go to tempModel/
cmsenv
bash build.sh
# text2workspace.py -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel  --PO verbose --PO 'map=.*/hcc:r[1,0,10]' --PO 'map=.*/zcc:z[1,0,10]' model_combined.txt

combine -M FitDiagnostics --expectSignal 1 -d model_combined.root --rMin=-5 --rMax=10  --cminDefaultMinimizerStrategy 0 --robustFit=1 -t -1 --toysFrequentist
combine -M Significance model_combined.root --expectSignal 1  -t -1 --toysFrequentist
python ../HiggsAnalysis/CombinedLimit/test/diffNuisances.py tempModel/fitDiagnostics.root 


```
## To extract shapes/norms use combine harvester and make plots
```
PostFitShapesFromWorkspace -w model_combined.root -o shapes.root --print --postfit --sampling -f fitDiagnostics.root:fit_s
# If withing the model dir
python ../plot.py --data 
python ../plotTF.py --data

###
python ../plot.py --MC --year 2017 -o plots_MC_t1
```


### OTHER IMPORTANT COMMANDS
```
combineTool.py -M FitDiagnostics -m 125 -d model_combined.root --there --cminDefaultMinimizerStrategy 0 -t -1 --expectSignal 1
combineTool.py -M AsymptoticLimits -m 125 -d model_combined.root --there -t -1 --expectSignal 1 --rMin=-50 --rMax=50

combine -M FitDiagnostics -t -1 --expectSignal 0 -d tempModel_combined.root --rMin=-5 --rMax=10  --cminDefaultMinimizerStrategy 0 --robustFit=1
```

### Running Impacts
```
# Baseline
combineTool.py -M Impacts -d tempModel_combined.root -m 125 --doInitialFit --robustFit 1 --setParameterRanges r=-1,5 --cminDefaultMinimizerStrategy 0 --X-rtd FITTER_DYN_STEP --expectSignal 1 -t -1 --toysFrequentist 
# Condor
combineTool.py -M Impacts -d tempModel_combined.root -m 125 --doFits --robustFit 1 --allPars --setParameterRanges r=-1,5  -t -1 --toysFrequentist --expectSignal 1 --cminDefaultMinimizerStrategy 0 --X-rtd MINIMIZER_analytic --job-mode condor --sub-opts='+JobFlavour = "workday"' --task-name ggHccFit --exclude 'rgx{qcdparams*}'
# Collect
combineTool.py -M Impacts -d tempModel_combined.root -m 125 --allPars -o impacts.json
plotImpacts.py -i impacts.json -o impacts_out --transparent --blind
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


#################################
##############################
### Shit that worked
```
# For Zcc, all systs
combine -M FitDiagnostics --expectSignal 1 -d tempModel_combined.root --rMin=-1 --rMax=3  --cminDefaultMinimizerStrategy 0 --robustFit=1  -t -1 --toysFrequentist


```