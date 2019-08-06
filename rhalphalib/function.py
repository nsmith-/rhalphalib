import numpy as np
from scipy.special import binom
import numbers
from .parameter import IndependentParameter


class BernsteinPoly(object):
    def __init__(self, name, order, dim_names=None, init_params=None, limits=None):
        '''
        Construct a multidimensional Bernstein polynomial
            name: will be used to prefix any RooFit object names
            order: tuple of order in each dimension
            dim_names: optional, names of each dimension
            initial_params: ndarray of initial params
            limits: tuple of independent parameter limits, default: (0, 10)
        '''
        self._name = name
        if not isinstance(order, tuple):
            raise ValueError
        self._order = order
        self._shape = tuple(n + 1 for n in order)
        if len(order) != len(dim_names):
            raise ValueError
        if dim_names:
            self._dim_names = dim_names
        else:
            self._dim_names = ['dim%d' % i for i in range(len(self._order))]
        if isinstance(init_params, np.ndarray):
            if init_params.shape != self._shape:
                raise ValueError
            self._init_params = init_params
        elif init_params is None:
            self._init_params = np.ones(shape=self._shape)
        else:
            raise ValueError

        # Construct Bernstein matrix for each dimension
        self._bmatrices = []
        for idim, (n, initialv) in enumerate(zip(self._order, self._init_params)):
            v = np.arange(n + 1)
            bmat = np.einsum("l,lv,lv->vl", binom.outer(n, v), binom.outer(v, v), np.power(-1., np.subtract.outer(v, v)))
            bmat[np.greater.outer(v, v)] = 0  # v > l
            self._bmatrices.append(bmat)

        # Construct parameter tensor
        self._params = np.full(self._shape, None)
        for ipar, initial in np.ndenumerate(self._init_params):
            param = IndependentParameter('_'.join([self.name] + ['%s_par%d' % (d, i) for d, i in zip(self._dim_names, ipar)]), initial, lo=0.)
            self._params[ipar] = param

    @property
    def name(self):
        return self._name

    def __call__(self, *vals):
        if len(vals) != len(self._order):
            raise ValueError("Not all dimension values specified")
        xvals = []
        shape = None
        for x in vals:
            if isinstance(x, numbers.Number):
                x = np.array(x)
            if not np.all((x >= 0) & (x <= 1)):
                raise ValueError("Bernstein polynomials are only defined on the interval [0, 1]")
            if shape is None:
                shape = x.shape
            elif shape != x.shape:
                raise ValueError("BernsteinPoly: all variables must have same shape")
            xvals.append(x.flatten())

        # evaluate Bernstein polynomial product tensor
        bpolyval = np.ones_like(xvals[0])
        for x, n, B in zip(xvals, self._order, self._bmatrices):
            xpow = np.power.outer(x, np.arange(n + 1))
            bpolyval = np.einsum("vl,xl,x...->x...v", B, xpow, bpolyval)

        coefficients = bpolyval.reshape(len(xvals[0]), -1)
        parameters = self._params.reshape(-1)

        out = np.full(len(xvals[0]), None)
        for i in range(len(xvals[0])):
            # sum small coefficients first
            order = np.argsort(coefficients[i])
            p = np.sum(parameters[order]*coefficients[i][order])
            dimstr = '_'.join('%s%.3f' % (d, v[i]) for d, v in zip(self._dim_names, xvals))
            p.name = self.name + '_eval_' + dimstr.replace('.', 'p')
            p.intermediate = False
            out[i] = p
        return out.reshape(shape)
