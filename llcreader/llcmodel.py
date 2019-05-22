import numpy as np
import dask.array as dsa
import xarray as xr


def _get_var_metadata():
    # The LLC run data comes with zero metadata. So we import metadata from
    # the xmitgcm package.
    from xmitgcm.utils import parse_available_diagnostics
    from xmitgcm.variables import state_variables
    from xmitgcm.default_diagnostics import diagnostics
    from io import StringIO

    diag_file = StringIO(diagnostics)
    available_diags = parse_available_diagnostics(diag_file)
    var_metadata = state_variables
    var_metadata.update(available_diags)

    # even the file names from the LLC data differ from standard MITgcm output
    aliases = {'Eta': 'ETAN', 'PhiBot': 'PHIBOT', 'Salt': 'SALT',
               'Theta': 'THETA'}
    for a, b in aliases.items():
        var_metadata[a] = var_metadata[b]

    return var_metadata

_VAR_METADATA = _get_var_metadata()


def _reshape_2d_llc_data(data, jdim):
    """Transform flat 2D LLC file to 13-face version."""
    # vendored from xmitgcm
    # https://github.com/xgcm/xmitgcm/blob/master/xmitgcm/utils.py

    LLC_NUM_FACES = 13
    nside = data.shape[jdim] // LLC_NUM_FACES
    # how the LLC data is laid out along the j dimension
    strides = ((0,3), (3,6), (6,7), (7,10), (10,13))
    # whether to reshape each face
    reshape = (False, False, False, True, True)
    # this will slice the data into 5 facets
    slices = [jdim * (slice(None),) + (slice(nside*st[0], nside*st[1]),)
              for st in strides]
    facet_arrays = [data[sl] for sl in slices]
    face_arrays = []
    for ar, rs, st in zip(facet_arrays, reshape, strides):
        nfaces_in_facet = st[1] - st[0]
        shape = list(ar.shape)
        if rs:
            # we assume the other horizontal dimension is immediately after jdim
            shape[jdim] = ar.shape[jdim+1]
            shape[jdim+1] = ar.shape[jdim]
        # insert a length-1 dimension along which to concatenate
        shape.insert(jdim, 1)
        ar.shape = shape
        # now ar is propery shaped, but we still need to slice it into 13 faces
        face_slice_dim = jdim + 1 + rs
        for n in range(nfaces_in_facet):
            face_slice = (face_slice_dim * (slice(None),) +
                          (slice(nside*n, nside*(n+1)),))
            data_face = ar[face_slice]
            face_arrays.append(data_face)

    return np.concatenate(face_arrays, axis=jdim)


def _load_level_from_3D_field(fs, path offset, count, dtype):
    #inum_str = '%010d' % inum
    #fname = os.path.join(ddir, inum_str,
    #                     '%s.%s.data.shrunk' % (varname, inum_str))
    #with open(fname, mode='rb') as file:
    #    file.seek(offset * dtype.itemsize)
    #    data = np.fromfile(file, dtype=dtype, count=count)

    file = fs.open(path)
    file.seek(offset * dtype.itemsize)
    buffer = file.read(length=count)
    data = np.frombuffer(buffer, dtype=dtype)

    data_blank = np.full_like(mask, np.nan, dtype=dtype)
    data_blank[mask] = data
    data_blank.shape = mask.shape
    data_llc = _reshape_llc_data(data_blank, jdim=0).compute(get=dask.get)
    data_llc.shape = (1,) + data_llc.shape
    return data_llc


def _lazily_load_level_from_3D_field(file, offset, count, mask, dtype):
    return dsa.from_delayed(dask.delayed(load_level_from_3D_field)
                            (file, offset, count, mask, dtype),
                            (1, nface, ny, nx), dtype)



def _build_facet_chunk(fs, path, file_shape, which_facets=None,
                       which_levels=None):
    """Create numpy data from a file

    Parameters
    ----------
    fs : fsspec.Filesystem
    path : str
    file_shape : tuple of ints
        The shape of the data in the file
    which_facets : int or list of ints, optional
        Which facets to read
    which_levels : int or lits of ints, optional
        Which k levels to read

    Returns
    -------
    out : np.ndarray
        The data
    """

    assert len(file_shape) == 4, 'file_shape should have length of 4'
    nk, nface, ny, nx = file_shape
    




class BaseLLCModel:
    self.nz = 90
    self.nface = 13
    self.dtype = np.dtype('>f4')

    def __init__(self, datastore, mask_ds):
        self.store = datastore
        self.shape = (self.nz, self.nface, self.nx, self.nx)
        self.masks = self._get_masks(mask_ds)


    def _get_masks(self, mask_ds, check=False):
        for point in ['C', 'W', 'S']:
            # store mask data as a raw array, not xarray object, to avoid any
            # alignment overhead
            data = mask_da['mask' + point].data
            assert data.dtype == np.bool
            assert data.shape == self.shape
            self.masks[point] = data


    def _make_coords(self):
        all_iters = np.arange(self.iter_start, self.iter_stop, self.iter_step)
        time = self.delta_t * all_iters
        coords = {'face': ('face', np.arange(self.nface)),
                  'i': ('i', np.arange(self.nx)),
                  'i_g': ('i_g', np.arange(self.nx)),
                  'j': ('j', np.arange(self.nx)),
                  'j_g': ('j_g', np.arange(self.nx)),
                  'k': ('k', np.arange(self.nz)),
                  'k_u': ('k_u', np.arange(self.nz)),
                  'k_l': ('k_l', np.arange(self.nz)),
                  'k_p1': ('k_p1', np.arange(self.nz + 1)),
                  'niter': ('time', all_iters),
                  'time': ('time', time, {'units': mtime['units']})
                 }
        return xr.decode_cf(xr.Dataset(coords=coords))


    def _make_data_variable(self, varname, iters, k=0, point='C'):
        # look up metadata?

        shape = (1, self.nface, self.ny, self.nz)
        dtype = self.dtype
        strides = [0,] + list(ds_index['hFac' + point].data)
        offset = strides[k]
        count = strides[k+1]
        mask = self._mask[point]

        # TODO
        try:
            mask_future = client.scatter(mask)
        except NameError:
            mask_future = mask

        data = dsa.concatenate([_lazily_load_level_from_3D_field
                                (varname, i, offset, count, mask_future, dtype)
                                for i in all_iters], axis=0)

        return data


    def get_dataset(variables, iter_start=None, iter_stop=None,
                    iter_step=None, k_levels=None, k_chunksize=1):
        """
        Create an xarray Dataset object for this model.

        Parameters
        ----------
        *varnames : list of strings, optional
            The variables to include, e.g. ``['Salt', 'Theta']``. Otherwise
            include all known variables.
        iter_start : int, optional
            Starting iteration number. Otherwise use model default.
            Follows standard `range` conventions. (inclusive)
        iter_start : int, optional
            Stopping iteration number. Otherwise use model default.
            Follows standard `range` conventions. (exclusive)
        iter_step : int, optional
            Iteration number stepsize. Otherwise use model default.
        k_levels : list of ints, optional
            Vertical levels to extract. Default is to get them all
        k_chunksize : int, optional
            How many vertical levels per Dask chunk.
        """

        iters = np.arange(iter_start, iter_stop, iter_step)

        ds = self._make_coords()
        for vname in variables:
            ds[vname] = self._make_data_variable(vname, iters, k=0)
        return ds

class LLC4320Model(BaseLLCModel):
    self.nx = 4320
    self.delta_t = 25
    self.iter_start = 10368
    self.iter_stop = 1310544 + 1
    self.iter_step = 144
    self.time_units='seconds since 2011-09-10'
    self.variables = ['']
