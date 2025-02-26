# rhalphalib

[![Codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![Ralph](https://upload.wikimedia.org/wikipedia/en/thumb/1/14/Ralph_Wiggum.png/220px-Ralph_Wiggum.png)

## Quickstart

### CMSSW with EL8/EL9

We use the [scram-venv](http://cms-sw.github.io/venv.html) utility to create a
python virtual environment in our CMSSW area:

```bash
cmsrel CMSSW_14_1_0_pre4
cd CMSSW_14_1_0_pre4/src
cmsenv
scram-venv
cmsenv
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
git checkout v10.1.0
scram b -j4
python3 -m pip install git+https://github.com/nsmith-/rhalphalib.git
```

### Elsewhere (no warranty)

First, install
[Combine v10](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/#installation-instructions)
using your choice of installation instructions (with CMSSW, using LCG, or inside
a Conda environment). In a python virtual environment, run:

```bash
python3 -m pip install --user git+https://github.com/nsmith-/rhalphalib.git
```

## Usage

Take a look at
[test_rhalphalib.py](https://github.com/nsmith-/rhalphalib/blob/master/tests/test_rhalphalib.py)
for examples of how to use the package. You can run a test with, e.g.

```bash
curl -Ol https://raw.githubusercontent.com/nsmith-/rhalphalib/master/tests/test_rhalphalib.py
python3 test_rhalphalib.py
cd tmp/testModel
. build.sh
combine -M FitDiagnostics model_combined.root
```

An example output of the final line is:

```
 <<< Combine >>>
 <<< v10.1.0 >>>
>>> Random number generator seed is 123456
>>> Method used is FitDiagnostics

 --- FitDiagnostics ---
Best fit r: 0.999999  -0.208174/+0.212715  (68% CL)
Done in 0.59 min (cpu), 0.59 min (real)
```
