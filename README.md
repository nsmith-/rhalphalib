# rhalphalib

[![Codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![Ralph](https://upload.wikimedia.org/wikipedia/en/thumb/1/14/Ralph_Wiggum.png/220px-Ralph_Wiggum.png)

## Quickstart

### CMSSW with EL8/EL9
Tested May 2024 on CMSLPC el8 (should work with lxplus9 as well). We use the [scram-venv](http://cms-sw.github.io/venv.html) utility
to create a python virtual environment in our CMSSW area.
```bash
cmsrel CMSSW_13_3_2
cd CMSSW_13_3_2/src
cmsenv
scram-venv
cmsenv
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
scram b -j4
python3 -m pip install https://github.com/nsmith-/rhalphalib/archive/master.zip
```

### Elsewhere (no warranty)
First, install [Combine v9](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/#installation-instructions) using
your choice of installation instructions (with CMSSW, using LCG, or inside a Conda environment). In a python virtual environment, run:
```bash
python3 -m pip install --user https://github.com/nsmith-/rhalphalib/archive/master.zip
```

## Usage

Take a look at [test_rhalphalib.py](https://github.com/nsmith-/rhalphalib/blob/master/tests/test_rhalphalib.py)
for examples of how to use the package. You can run a test with, e.g.
```bash
curl -Ol https://raw.githubusercontent.com/nsmith-/rhalphalib/master/tests/test_rhalphalib.py
python3 test_rhalphalib.py
cd tmp/testModel
. build.sh
combine -M FitDiagnostics model_combined.root
```
