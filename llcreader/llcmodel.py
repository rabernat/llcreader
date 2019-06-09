import numpy as np
import dask
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


def _get_scalars_and_vectors(varnames, type):

    for vname in varnames:
        if vname not in _VAR_METADATA:
            raise ValueError("Varname `%s` not found in metadata." % vname)

    if type != 'latlon':
        return varnames, []

    scalars = []
    vector_pairs = []
    for vname in varnames:
        meta = _VAR_METADATA[vname]
        try:
            mate = meta['attrs']['mate']
            if mate not in varnames:
                raise ValueError("Vector pairs are required to create "
                                 "latlon type datasets. Varname `%s` is "
                                 "missing its vector mate `%s`"
                                 % vname, mate)
            vector_pairs.append((vname, mate))
            varnames.remove(mate)
        except KeyError:
            scalars.append(vname)

def _decompress(data, mask, dtype):
    data_blank = np.full_like(mask, np.nan, dtype=dtype)
    data_blank[mask] = data
    data_blank.shape = mask.shape
    return data_blank



_facet_strides = ((0,3), (3,6), (6,7), (7,10), (10,13))
# whether to reshape each face
_facet_reshape = (False, False, False, True, True)
_nfaces = 13
_nfacets = 5

def _uncompressed_facet_index(nfacet, nside):
    face_size = nside**2
    start = _facet_strides[nfacet][0] * face_size
    end = _facet_strides[nfacet][1] * face_size
    return start, end

def _facet_shape(nfacet, nside):
    facet_length = _facet_strides[nfacet][1] - _facet_strides[nfacet][0]
    if _facet_reshape[nfacet]:
        facet_shape = (1, 1, nside, facet_length*nside)
    else:
        facet_shape = (1, 1, facet_length*nside, nside)
    return facet_shape

def _facet_to_faces(data, nfacet):
    nz, nf, ny, nx = data.shape
    assert nf == 1
    facet_length = _facet_strides[nfacet][1] - _facet_strides[nfacet][0]
    if _facet_reshape[nfacet]:
        new_shape = nz, ny, facet_length, nx / facet_length
        data_rs = data.reshape(new_shape)
        data_rs = np.moveaxis(data_rs, 2, 1) # dask-safe
    else:
        new_shape = nz, facet_length, ny / facet_length, nx
        data_rs = data.reshape(new_shape)
    return data_rs

def _facets_to_faces(facets):
    all_faces = []
    for nfacet, data_facet in enumerate(facets):
        data_rs = _facet_to_faces(data_facet, nfacet)
        all_faces.append(data_rs)
    return dsa.concatenate(all_faces, axis=1)

def _faces_to_facets(data, facedim=-3):
    assert data.shape[facedim] == _nfaces
    facets = []
    for nfacet, (strides, reshape) in enumerate(zip(_facet_strides, _facet_reshape)):
        face_data = [data[(...,) + (slice(nface, nface+1), slice(None), slice(None))]
                     for nface in range(*strides)]
        if reshape:
            concat_axis = facedim + 2
        else:
            concat_axis = facedim + 1
        # todo: use duck typing for concat
        facet_data = dsa.concatenate(face_data, axis=concat_axis)
        facets.append(facet_data)
    return facets


def _rotate_scalar_facet(facet):
    facet_transposed = np.moveaxis(facet, -1, -2)
    facet_rotated = np.flip(facet_transposed, -2)
    return facet_rotated


def _facets_to_latlon_scalar(all_facets):
    rotated = (all_facets[:2]
               + [_rotate_scalar_facet(facet) for facet in all_facets[-2:]])
    # drop facet dimension
    rotated = [r[..., 0, :, :] for r in rotated]
    return dsa.concatenate(rotated, axis=-1)


def _faces_to_latlon_scalar(data):
    data_facets = _faces_to_facets(data)
    return _facets_to_latlon_scalar(data_facets)


# dask's pad function doesn't work
# it does weird things to non-pad dimensions
# need to roll our own
def shift_and_pad(a):
    a_shifted = a[..., 1:]
    pad_array = dsa.zeros_like(a[..., -2:-1])
    return dsa.concatenate([a_shifted, pad_array], axis=-1)

def transform_v_to_u(facet):
    return _rotate_scalar_facet(facet)

def transform_u_to_v(facet, metric=False):
    # "shift" u component by 1 pixel
    pad_width = (facet.ndim - 1) * (None,) + ((1, 0),)
    #facet_padded = dsa.pad(facet[..., 1:], pad_width, 'constant')
    facet_padded = shift_and_pad(facet)
    assert facet.shape == facet_padded.shape
    facet_rotated = _rotate_scalar_facet(facet_padded)
    if not metric:
        facet_rotated = -facet_rotated
    return facet_rotated

def _facets_to_latlon_vector(facets_u, facets_v, metric=False):
    # need to pad the rotated v values
    ndim = facets_u[0].ndim
    # second-to-last axis is the one to pad, plus a facet axis
    assert ndim >= 3

    # drop facet dimension
    facets_u_drop = [f[..., 0, :, :] for f in facets_u]
    facets_v_drop = [f[..., 0, :, :] for f in facets_v]

    u_rot = (facets_u_drop[:2]
             + [transform_v_to_u(facet) for facet in facets_v_drop[-2:]])
    v_rot = (facets_v_drop[:2]
             + [transform_u_to_v(facet) for facet in facets_u_drop[-2:]])

    u = dsa.concatenate(u_rot, axis=-1)
    v = dsa.concatenate(v_rot, axis=-1)
    return u, v

def _faces_to_latlon_vector(u_faces, v_faces, metric=False):
    u_facets = _faces_to_facets(u_faces)
    v_facets = _faces_to_facets(v_faces)
    u, v = _facets_to_latlon_vector(u_facets, v_facets, metric=metric)
    return u, v

def _drop_facedim(dims):
    dims = list(dims)
    dims.remove('face')
    return dims

def _add_face_to_dims(dims):
    if 'j'in dims or 'j_g' in dims:
        new_dims = dims.copy()
        j_dim = dims.index('j')
        new_dims.inset('face', j_dim)
        return new_dims
    else:
        return dims

def _faces_coords_to_latlon(ds):
    coords = ds.reset_coords().coords.to_dataset()
    ifac = 4
    jfac = 3
    dim_coords = {}
    for vname in coords.coords:
        if vname[0] == 'i':
            data = np.arange(ifac * coords.dims[vname])
        elif vname[0] == 'j':
            data = np.arange(jfac * coords.dims[vname])
        else:
            data = coords[vname].data
        var = xr.Variable(ds[vname].dims, data, ds[vname].attrs)
        dim_coords[vname] = var
    return xr.Dataset(dim_coords)

def faces_dataset_to_latlon(ds, metric_vector_pairs=[('dxC', 'dyC'), ('dyG', 'dxG')]):
    """Transform a 13-face LLC xarray Dataset into a rectancular grid,
    discarding the Arctic.

    Parameters
    ----------
    ds : xarray.Dataset
        A 13-face LLC dataset
    metric_vector_pairs : list, optional
        Pairs of variables that are positive-definite metrics located at grid
        edges.

    Returns
    -------
    out : xarray.Dataset
        Transformed rectangular dataset
    """

    coord_vars = list(ds.coords)
    ds_new = _faces_coords_to_latlon(ds)

    vector_pairs = []
    scalars = []
    vnames = list(ds.reset_coords().variables)
    for vname in vnames:
        try:
            mate = ds[vname].attrs['mate']
            vector_pairs.append((vname, mate))
            vnames.remove(mate)
        except KeyError:
            pass

    all_vector_components = [inner for outer in (vector_pairs + metric_vector_pairs)
                             for inner in outer]
    scalars = [vname for vname in vnames if vname not in all_vector_components]
    data_vars = {}

    for vname in scalars:
        if vname=='face' or vname in ds_new:
            continue
        if 'face' in ds[vname].dims:
            data = _faces_to_latlon_scalar(ds[vname].data)
            dims = _drop_facedim(ds[vname].dims)
        else:
            data = ds[vname].data
            dims = ds[vname].dims
        data_vars[vname] = xr.Variable(dims, data, ds[vname].attrs)

    for vname_u, vname_v in vector_pairs:
        data_u, data_v = _faces_to_latlon_vector(ds[vname_u].data, ds[vname_v].data)
        data_vars[vname_u] = xr.Variable(_drop_facedim(ds[vname_u].dims), data_u, ds[vname_u].attrs)
        data_vars[vname_v] = xr.Variable(_drop_facedim(ds[vname_v].dims), data_v, ds[vname_v].attrs)
    for vname_u, vname_v in metric_vector_pairs:
        data_u, data_v = _faces_to_latlon_vector(ds[vname_u].data, ds[vname_v].data, metric=True)
        data_vars[vname_u] = xr.Variable(_drop_facedim(ds[vname_u].dims), data_u, ds[vname_u].attrs)
        data_vars[vname_v] = xr.Variable(_drop_facedim(ds[vname_v].dims), data_v, ds[vname_v].attrs)


    ds_new = ds_new.update(data_vars)
    ds_new = ds_new.set_coords([c for c in coord_vars if c in ds_new])
    return ds_new


# below are data transformers

def _all_facets_to_faces(data_facets, meta):
    return {vname: _facets_to_faces(data)
            for vname, data in data_facets.items()}


def _all_facets_to_latlon(data_facets, meta):

    vector_pairs = []
    scalars = []
    vnames = list(data_facets)
    for vname in vnames:
        try:
            mate = meta['attrs']['mate']
            vector_pairs.append((vname, mate))
            vnames.remove(mate)
        except KeyError:
            pass

    all_vector_components = [inner for outer in vector_pairs for inner in outer]
    scalars = [vname for vname in vnames if vname not in all_vector_components]

    data = {}
    for vname in scalars:
        data[vname] = _facet_to_latlon_scalar(data_facets[vname])

    for vname_u, vname_v in vector_pairs:
        data_u, data_v = _facet_to_latlon_vector(data_facets[vname_u],
                                                 data_facets[vname_v])
        data[vname_u] = data_u
        data[vname_v] = data_v

    return data


######################### OLD BELOW #############################

class _LLCDataRequest:

    def __init__(self, fs, path, dtype, nk, nx,
                 klevels=[0], index=None, mask=None):
        """Create numpy data from a file

        Parameters
        ----------
        fs : fsspec.Filesystem
        path : str
        file_shape : tuple of ints
            The shape of the data in the file
        dtype : numpy.dtype
            Data type of the data in the file
        nfacet : int
            Which facet to read
        levels : int or lits of ints, optional
            Which k levels to read
        index : dict
        mask : dask.array

        Returns
        -------
        out : np.ndarray
            The data
        """

        self.fs = fs
        self.path = path
        self.dtype = dtype
        self.nk = nk
        self.nx = nx
        self.klevels = klevels
        self.mask = mask
        self.index = index


    def build_facet_chunk(self, nfacet):

        assert (nfacet >= 0) & (nfacet < _nfacets)

        try:
            # workaround for ecco data portal
            file = self.fs.open(self.path, size_policy='get')
        except TypeError:
            file = self.fs.open(self.path)
        facet_shape = _facet_shape(nfacet, self.nx)

        level_data = []
        for k in self.klevels:
            assert (k >= 0) & (k < self.nk)

            # figure out where in the file we have to read to get the data
            # for this level and facet
            if self.index:
                i = np.ravel_multi_index((k, nfacet), (self.nk, _nfacets))
                start = self.index[i]
                end = self.index[i+1]
                print('start, end', start, end)
            else:
                level_start = k * self.nx**2 * _nfaces
                facet_start, facet_end = _uncompressed_facet_index(nfacet, self.nx)
                start = level_start + facet_start
                end = level_start + facet_end

            read_offset = start * self.dtype.itemsize # in bytes
            read_length  = (end - start) * self.dtype.itemsize # in bytes
            file.seek(read_offset)
            buffer = file.read(read_length)
            data = np.frombuffer(buffer, dtype=self.dtype)
            assert len(data) == (end - start)

            if self.mask:
                this_mask = self.mask[nfacet][k].compute()
                data = _decompress(data, this_mask, self.dtype)

            # this is the shape this facet is supposed to have
            data.shape = facet_shape
            level_data.append(data)

        return np.concatenate(level_data, axis=0)

    def lazily_build_facet_chunk(self, nfacet):
        facet_shape = _facet_shape(nfacet, self.nx)
        shape = (len(self.klevels),) + facet_shape[1:]
        return dsa.from_delayed(dask.delayed(self.build_facet_chunk)(nfacet),
                                shape, self.dtype)

    def facets(self):
        return [self.lazily_build_facet_chunk(nfacet) for nfacet in range(5)]



class BaseLLCModel:
    nz = 90
    nface = 13
    dtype = np.dtype('>f4')

    def __init__(self, datastore, mask_ds):
        """Initialize model

        Parameters
        ----------
        datastore : llcreader.BaseStore
        mask_ds : zarr.Group
            Must contain variables `mask_c`, `masc_w`, `mask_s`
        """
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


    def _make_coords_faces(self):
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


    def _make_coords_latlon():
        ds = self._make_coords_faces(self):
        return _faces_coords_to_latlon(ds)


    def _get_facet_data(self, varname, iters, k=levels):
        mask, index = self.get_mask_and_index_for_variable(varname)
        # needs facets to be outer index of nested lists
        data_iters = 5 * [[],]
        for iternum in iters:
            fs, path = self.store.get_fs_and_full_path(self, varname, iternum)
            dr = _LLCDataRequest(fs, path, self.dtype, self.nk, self.nx,
                                 mask=mask, index=index, klevels=klevels)
            data_facets = dr.facets()
            for n in range(5):
                # insert a new axis for time at the beginning
                data_iters[n].append(data_facets[n][None])

        data = [dsa.concatenate(facet, axis=0) for facet in data_iters]
        return data


    def get_dataset(variables, iter_start=None, iter_stop=None,
                    iter_step=None, k_levels=None, k_chunksize=1,
                    type='faces'):
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
        type : {'faces', 'latlon'}, optional
            What type of dataset to create
        """

        iters = np.arange(iter_start, iter_stop, iter_step)

        ds = self._make_coords_faces()
        if type=='latlon':
            ds = _faces_coords_to_lalon(ds)

        if varnames is None:
            varnames = self.varnames

        # get the data in facet form
        data_facets = {vname:
                       self._get_facet_data(vname, iters, k_levels, k_chunksize),
                       for vname in varnames}

        # transform it into faces or latlon
        data_transformers = {'faces': _all_facets_to_faces,
                             'latlon': _all_facets_to_latlon}

        transformer = data_transformers[type]
        data = transformer(data_facets, _VAR_METADATA)

        variables = {}
        for vname in varnames:
            meta = _VAR_METADATA[vname]
            dims = meta['dims']
            if type=='latlon':
                dims = _add_face_to_dims[dims]
            attrs = meta['attrs']
            variables[vname] = xr.Variable(dims, data[vname], attrs)

        ds = ds.update(data_vars)
        return ds


class LLC4320Model(BaseLLCModel):
    nx = 4320
    delta_t = 25
    iter_start = 10368
    iter_stop = 1310544 + 1
    iter_step = 144
    time_units='seconds since 2011-09-10'
    varnames = ['']
