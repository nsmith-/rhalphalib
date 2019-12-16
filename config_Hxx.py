import numpy as np

# Define Bins
ptbins = np.array([450, 500, 550, 600, 675, 800, 1200])
npt = len(ptbins) - 1
msdbins = np.linspace(40, 201, 24)

# Define pt/msd/rho grids
ptpts, msdpts = np.meshgrid(ptbins[:-1] + 0.3 * np.diff(ptbins),
                            msdbins[:-1] + 0.5 * np.diff(msdbins),
                            indexing='ij')
rhopts = 2*np.log(msdpts/ptpts)
ptscaled = (ptpts - 450.) / (1200. - 450.)
rhoscaled = (rhopts - (-6)) / ((-2.1) - (-6))
validbins = (rhoscaled >= 0) & (rhoscaled <= 1)
rhoscaled[~validbins] = 1  # we will mask these out later

# Define fine bins for smooth TF plots
fptbins = np.arange(450, 1202, 2)
fmsdbins = np.arange(40, 201.5, .5)

fptpts, fmsdpts = np.meshgrid(fptbins[:-1] + 0.3 * np.diff(fptbins),
                              fmsdbins[:-1] + 0.5 * np.diff(fmsdbins),
                              indexing='ij')
frhopts = 2*np.log(fmsdpts/fptpts)
fptscaled = (fptpts - 450.) / (1200. - 450.)
frhoscaled = (frhopts - (-6)) / ((-2.1) - (-6))
fvalidbins = (frhoscaled >= 0) & (frhoscaled <= 1)
frhoscaled[~fvalidbins] = 1  # we will mask these out later


def TF_params(xparlist, xparnames=None, nrho=None, npt=None):
    # TF map from param/name lists
    if xparnames is not None:
        from operator import methodcaller

        def _get(s):
            return s[-1][0]

        ptdeg = max(
            list(
                map(
                    int,
                    list(
                        map(_get, list(map(methodcaller("split", 'pt_par'),
                                           xparnames)))))))
        rhodeg = max(
            list(
                map(
                    int,
                    list(
                        map(_get, list(map(methodcaller("split", 'rho_par'),
                                           xparnames)))))))
    else:
        rhodeg, ptdeg = nrho, npt

    TF_cf_map = np.array(xparlist).reshape(rhodeg + 1, ptdeg + 1)

    return TF_cf_map, rhodeg, ptdeg
