"""
Contains the base class for galaxy catalog (BaseGalaxyCatalog).
"""
__all__ = ['BaseGalaxyCatalog']


from collections import defaultdict
import numpy as np
import h5py


class BaseGalaxyCatalog(object):
    """
    Abstract base class for all galaxy catalog classes.
    """
    _required_attributes = ('cosmology',)
    _required_quantities = ('redshift',)

    _quantity_modifiers = dict()
    _native_quantities = set()
    _pre_filter_quantities = set()


    def __init__(self, **kwargs):
        self._subclass_init(**kwargs)

        # enforce the existence of required attributes
        if not all(hasattr(self, attr) for attr in self._required_attributes):
            raise ValueError("Any subclass of GalaxyCatalog must implement following attributes: {0}".format(', '.join(self._required_attributes)))

        # enforce the minimal set of quantities
        if not all(self.has_quantity(q) for q in self._required_quantities):
            raise ValueError("GalaxyCatalog must have the following quantities: {0}".format(self._required_quantities))

        # make sure all quantity modifiers are set correctly
        try:
            for q in self._quantity_modifiers:
                self._translate_quantity(q)
        except TypeError:
            raise ValueError('modifier for {} is not set correctly'.format(q))

        if not all(q in self._native_quantities for q in self._translate_quantities(self.list_all_quantities(True))):
            raise ValueError('the reader specifies quantities that are not in the catalog')

        # This is only for backward compatibility (TODO: to be removed)
        self.quantities = set(self.list_all_quantities(True))


    def list_all_quantities(self, include_native=False):
        """
        Return a set of all available quantities in this catalog
        """
        output = list(self._quantity_modifiers)
        if include_native:
            for q in self._native_quantities:
                if q not in output:
                    output.append(q)
        return output


    def has_quantity(self, quantity):
        """
        Check if a specific quantity is available in this galaxy catalog

        Parameters
        ----------
        quantity : str
            quantity name to check

        Returns
        -------
        has_quantity : bool
            True if the quantity is available; otherwise False
        """
        return all(q in self._native_quantities for q in self._translate_quantity(quantity))


    def _translate_quantity(self, quantity_requested):
        modifier = self._quantity_modifiers.get(quantity_requested)

        if modifier is None or callable(modifier):
            return {quantity_requested}

        elif isinstance(modifier, (tuple, list)) and len(modifier) > 1 and callable(modifier[0]):
            return set(modifier[1:])

        return {modifier}


    def _translate_quantities(self, quantities_requested):
        native_quantities = set()
        for q in quantities_requested:
            native_quantities.update(self._translate_quantity(q))
        return native_quantities


    def _assemble_quantity(self, quantity_requested, native_quantities_loaded):
        modifier = self._quantity_modifiers.get(quantity_requested)

        if modifier is None:
            return native_quantities_loaded[quantity_requested]

        elif callable(modifier):
            return modifier(native_quantities_loaded[quantity_requested])

        elif isinstance(modifier, (tuple, list)) and len(modifier) > 1 and callable(modifier[0]):
            return modifier[0](*(native_quantities_loaded[_] for _ in modifier[1:]))

        return native_quantities_loaded[modifier]


    @staticmethod
    def _get_mask_from_filter(filters, data, premask=None):
        mask = premask
        for f in filters:
            if mask is None:
                mask = f[0](*(data[_] for _ in f[1:]))
            else:
                mask &= f[0](*(data[_] for _ in f[1:]))
        return mask


    @staticmethod
    def _get_quantities_from_filters(filters):
        return set(q for f in filters for q in f[1:])


    def _load_quantities(self, quantities, dataset):
        native_data = {q: self._fetch_native_quantity(dataset, q) for q in self._translate_quantities(quantities)}
        return {q: self._assemble_quantity(q, native_data) for q in quantities}


    def _preprocess_requested_quantities(self, quantities):
        if quantities is None:
            quantities = self.list_all_quantities()

        if isinstance(quantities, basestring):
            quantities = {quantities}

        quantities = set(quantities)
        if not quantities:
            raise ValueError('You must set `quantities`.')

        if not all(q in self._native_quantities for q in self._translate_quantities(quantities)):
            raise ValueError('Some quantities are not available in this catalog')

        return quantities


    def _preprocess_requested_filters(self, filters):

        if isinstance(filters, dict) and ('zlo' in filters or 'zhi' in filters): # This is only for backward compatibility (TODO: to be removed)
            _zlo = float(filters.get('zlo', -np.inf))
            _zhi = float(filters.get('zhi', np.inf))
            filters = [(lambda z: (z >= _zlo) & (z <= _zhi), 'redshift')]

        if filters is None:
            filters = tuple()

        if not all(isinstance(f, (tuple, list)) and len(f) > 1 and callable(f[0]) and all(isinstance(q, basestring) for q in f[1:]) for f in filters):
            raise ValueError('`filters is not set correctly. Must be None or [(callable, str, str, ...), ...]')

        if not all(q in self._native_quantities for q in self._translate_quantities(self._get_quantities_from_filters(filters))):
            raise ValueError('Some filters are not available in this catalog')

        pre_filters = list()
        post_filters = list()
        for f in filters:
            if set(f[1:]).issubset(self._pre_filter_quantities):
                pre_filters.append(f)
            else:
                post_filters.append(f)

        return pre_filters, post_filters


    def _get_quantities_iter(self, quantities, pre_filters, post_filters):
        for dataset in self._iter_native_dataset(pre_filters):
            mask = self._get_mask_from_filter(pre_filters, self._load_quantities(self._get_quantities_from_filters(pre_filters), dataset))
            if mask is not None:
                if not mask.any():
                    continue
                if mask.all():
                    mask = None
            data = self._load_quantities(quantities.union(self._get_quantities_from_filters(post_filters)), dataset)
            mask = self._get_mask_from_filter(post_filters, data, mask)
            if mask is not None:
                for q in data:
                    data[q] = data[q][mask]
            del mask
            yield data
            del data


    def _concatenate_quantities(self, quantities, pre_filters, post_filters):
        requested_data = defaultdict(list)
        for data in self._get_quantities_iter(quantities, pre_filters, post_filters):
            for q in quantities:
                requested_data[q].append(data[q])
        return {q: np.concatenate(requested_data[q]) if requested_data[q] else np.array([]) for q in quantities}


    def get_quantities(self, quantities=None, filters=None, return_hdf5=None, return_iterator=False):
        """
        Fetch quantities from this galaxy catalog.

        Parameters
        ----------
        quantities : str or list of str or tuple of str
            quantities to fetch

        filters : list of tuple, optional
            filters to apply. Each filter should be in the format of (callable, str, str, ...)

        return_hdf5 : None or str, optional
            filename to a hdf5 file to store the return data.
            If `return_hdf5` is set, `return_iterator` is set to False.

        return_iterator : bool, optional
            if True, return an iterator that iterates over the native format, default is False

        Returns
        -------
        quantities : dict or h5py.File (when `return_hdf5` is set) or iterator (when `return_iterator` is True)
        """

        quantities = self._preprocess_requested_quantities(quantities)
        pre_filters, post_filters = self._preprocess_requested_filters(filters)

        if return_hdf5:
            with h5py.File(return_hdf5, 'w') as f:
                for q in quantities:
                    f.create_dataset(q, data=self._concatenate_quantities(q, pre_filters, post_filters)[q], chunks=True, compression="gzip", shuffle=True, fletcher32=True)
            return h5py.File(return_hdf5, 'r')

        if return_iterator:
            return self._get_quantities_iter(quantities, pre_filters, post_filters)

        return self._concatenate_quantities(quantities, pre_filters, post_filters)


    def _subclass_init(self, **kwargs):
        """ To be implemented by subclass. """
        raise NotImplementedError


    def _iter_native_dataset(self, pre_filters=None):
        """ To be implemented by subclass. Must be a generator."""
        raise NotImplementedError


    @staticmethod
    def _fetch_native_quantity(dataset, native_quantity):
        """ To be overwritten by subclass. Must return a 1-d numpy.ndarray """
        return np.asanyarray(dataset[native_quantity]).ravel()
