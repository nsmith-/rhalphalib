import numpy as np
import math

import matplotlib.pyplot as plt
import mplhep as hep


# Benstein polynomial calculation
def bern_elem(x, v, n):
    # Bernstein element calculation
    normalization = 1. * math.factorial(n) / (math.factorial(v) * math.factorial(n - v))
    Bvn = normalization * (x**v) * (1 - x)**(n - v)
    return float(Bvn)


def TF(pT, rho, par_map=np.ones((3, 3)), n_rho=2, n_pT=2):
    # Calculate TF Polynomial for (n_pT, n_rho) degree Bernstein poly
    val = 0
    for i_pT in range(0, n_pT + 1):
        for i_rho in range(0, n_rho + 1):
            val += (bern_elem(pT, i_pT, n_pT) * bern_elem(rho, i_rho, n_rho) *
                    par_map[i_pT][i_rho])

    return val


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

    TF_cf_map = np.array(xparlist).reshape(ptdeg + 1, rhodeg + 1)

    return TF_cf_map, rhodeg, ptdeg


def TF_smooth_plot(_tfmap, _rhodeg, _ptdeg):
    # Define fine bins for smooth TF plots
    fptbins = np.arange(450, 1202, 2)
    fmsdbins = np.arange(40, 201.5, .5)
    fptpts, fmsdpts = np.meshgrid(fptbins[:-1] + 0.3 * np.diff(fptbins),
                                  fmsdbins[:-1] + 0.5 * np.diff(fmsdbins),
                                  indexing='ij')
    frhopts = 2 * np.log(fmsdpts / fptpts)
    fptscaled = (fptpts - 450.) / (1200. - 450.)
    frhoscaled = (frhopts - (-6)) / ((-2.1) - (-6))
    fvalidbins = (frhoscaled >= 0) & (frhoscaled <= 1)
    frhoscaled[~fvalidbins] = 1  # we will mask these out later

    def wrapTF(pT, rho):
        return TF(pT, rho, n_pT=_ptdeg, n_rho=_rhodeg, par_map=_tfmap)

    TFres = np.array(list(map(wrapTF, fptscaled.flatten(),
                              frhoscaled.flatten()))).reshape(fptpts.shape)

    # return TF, msd bins, pt bins, mask
    return TFres, fmsdpts, fptpts, fvalidbins


# TF Plots
def plotTF(TF,
           msd,
           pt,
           mask=None,
           MC=False,
           raw=False,
           rhodeg=2,
           ptdeg=2,
           out=None,
           year="2017",
           label=None):
    """
    Parameters:
    TF: Transfer Factor array
    msd: Mass bins array (meshgrid-like)
    pt: pT bins array (meshgrid-like)
    """
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use([hep.style.ROOT, {'font.size': 24}])
    plt.switch_backend('agg')
    fig, ax = plt.subplots()
    if mask is not None:
        TF = np.ma.array(TF, mask=~mask)

    zmin, zmax = np.floor(10 * np.min(TF)) / 10, np.ceil(10 * np.max(TF)) / 10
    zmin, zmax = zmin + 0.001, zmax - 0.001
    clim = np.round(np.min([abs(zmin - 1), abs(zmax - 1)]), 1)
    if clim < .3:
        clim = .3
    if clim > .5:
        clim = .5
    levels = np.linspace(1 - clim, 1 + clim, 500)

    if np.min(TF) < 1 - clim and np.max(TF) > 1 + clim:
        _extend = 'both'
    elif np.max(TF) > 1 + clim:
        _extend = 'max'
    elif np.min(TF) < 1 - clim:
        _extend = 'min'
    else:
        _extend = 'neither'

    if mask is not None:
        contf = ax.contourf(msd,
                            pt,
                            TF,
                            levels=levels,
                            corner_mask=False,
                            cmap='RdBu_r',
                            extend=_extend)
    else:
        contf = ax.contourf(msd, pt, TF, levels=levels, cmap='RdBu_r', extend=_extend)
    cax = hep.make_square_add_cbar(ax, pad=0.2, size=0.5)
    if abs(1 - zmin) > .3 and abs(1 - zmax) > .3:
        c_extend = 'both'
    elif abs(1 - zmin) > .3:
        c_extend = 'min'
    elif abs(1 - zmax) > .3:
        c_extend = 'max'
    else:
        c_extend = 'neither'
    cbar = fig.colorbar(contf, cax=cax, extend=c_extend)
    cbar.set_ticks([np.arange(1 - clim, 1 + clim, 0.1)])

    def rho_bound(ms, rho):
        # rho = {-6, -2.1}
        fpt = ms * np.e**(-rho / 2)
        return fpt

    x = np.arange(40, 70)
    ax.plot(x, rho_bound(x, -6), 'black', lw=3)
    ax.fill_between(x,
                    rho_bound(x, -6),
                    1200,
                    facecolor="none",
                    hatch="xx",
                    edgecolor="black",
                    linewidth=0.0)
    x = np.arange(150, 201)
    ax.plot(x, rho_bound(x, -2.1) + 5, 'black', lw=3)
    ax.fill_between(x,
                    rho_bound(x, -2.1),
                    facecolor="none",
                    hatch="xx",
                    edgecolor="black",
                    linewidth=0.0)

    _mbins, _pbins = np.linspace(40, 201,
                                 24), np.array([450, 500, 550, 600, 675, 800, 1200])
    sampling = np.meshgrid(_mbins[:-1] + 0.5 * np.diff(_mbins),
                           _pbins[:-1] + 0.3 * np.diff(_pbins))
    valmask = (sampling[1] > rho_bound(sampling[0], -2.1)) & (sampling[1] < rho_bound(
        sampling[0], -6))
    ax.scatter(
        sampling[0][valmask],
        sampling[1][valmask],
        marker='x',
        color='black',
        s=40,
        alpha=.4,
    )

    ax.set_xlim(40, 201)
    ax.set_ylim(450, 1200)
    ax.invert_yaxis()

    tMC = "Tagger Response" if MC else "Residual"
    if raw and MC:
        tMC = "Tagger Response (prefit)"
    if label is None:
        label = '{} TF({},{})'.format(tMC, ptdeg, rhodeg)
    ax.set_title(label, pad=9, fontsize=22, loc='left')
    ax.set_title("({})".format(str(year)), pad=9, fontsize=22, loc='right')
    ax.set_xlabel(r'Jet $\mathrm{m_{SD}}$', ha='right', x=1)
    ax.set_ylabel(r'Jet $\mathrm{p_{T}}$', ha='right', y=1)
    cbar.set_label(r'TF', ha='right', y=1)

    label = "MC" if MC else "Data"
    if raw:
        label = "MCRaw"
    import mplhep as hep
    hep.cms.label(loc=2, data=not raw, rlabel="", ax=ax)
    if out is not None:
        fig.savefig('{}.png'.format(out))
    else:
        return fig


def plotTF_ratio(in_ratio, mask, region, args=None, zrange=None):
    fig, ax = plt.subplots()

    H = np.ma.masked_where(in_ratio * mask <= 0.01, in_ratio * mask)
    zmin, zmax = np.nanmin(H), np.nanmax(H)
    if zrange is None:
        # Scale clim to fit range up to a max of 0.6
        clim = np.max([.3, np.min([0.6, 1 - zmin, zmax - 1])])
    else:
        clim = zrange
    ptbins = np.array([450, 500, 550, 600, 675, 800, 1200])
    msdbins = np.linspace(40, 201, 24)
    hep.hist2dplot(H.T,
                   msdbins,
                   ptbins,
                   vmin=1 - clim,
                   vmax=1 + clim,
                   cmap='RdBu_r',
                   cbar=False)
    cax = hep.make_square_add_cbar(ax, pad=0.2, size=0.5)
    if abs(1 - zmin) > .3 and abs(1 - zmax) > .3:
        c_extend = 'both'
    elif abs(1 - zmin) > .3:
        c_extend = 'min'
    elif abs(1 - zmax) > .3:
        c_extend = 'max'
    else:
        c_extend = 'neither'
    cbar = fig.colorbar(ax.get_children()[0], cax=cax, extend=c_extend)

    ax.set_xticks(np.arange(40, 220, 20))
    ax.tick_params(axis='y', which='minor', left=False, right=False)
    ax.invert_yaxis()

    ax.set_title('{} QCD Ratio'.format(region), pad=15, fontsize=26)
    ax.set_xlabel(r'Jet $\mathrm{m_{SD}}$', ha='right', x=1)
    ax.set_ylabel(r'Jet $\mathrm{p_{T}}$', ha='right', y=1)
    cbar.set_label(r'(Pass QCD) / (Fail QCD * eff)', ha='right', y=1)

    fig.savefig('{}/{}{}.png'.format(args.output_folder, "TF_ratio_", region),
                bbox_inches="tight")
