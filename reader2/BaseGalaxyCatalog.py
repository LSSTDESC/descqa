"""
Contains the base class for galaxy catalog (BaseGalaxyCatalog).
"""
__all__ = ['BaseGalaxyCatalog']


from collections import defaultdict
import numpy as np
from numpy.core.records import fromarrays
import h5py


def _dict_to_ndarray(d):
    return fromarrays(d.values(), np.dtype([(str(k), v.dtype) for k, v in d.iteritems()]))


class BaseGalaxyCatalog(object):
    """
    Abstract base class for all galaxy catalog classes.
    """
    _required_attributes = ('cosmology',)
    _required_quantities = ('redshift',)

    _default_quantity_modifier = None
    _quantity_modifiers = dict()
    _pre_filter_quantities = set()


    def __init__(self, **kwargs):
        self._subclass_init(**kwargs)
        self._native_quantities = set(self._generate_native_quantity_list())

        # enforce the existence of required attributes
        if not all(hasattr(self, attr) for attr in self._required_attributes):
            raise ValueError("Any subclass of GalaxyCatalog must implement following attributes: {0}".format(', '.join(self._required_attributes)))

        # enforce the minimal set of quantities
        if not self.has_quantities(self._required_quantities):
            raise ValueError("GalaxyCatalog must have the following quantities: {0}".format(self._required_quantities))

        if not all(q in self._native_quantities for q in self._translate_quantities(self.list_all_quantities(True))):
            raise ValueError('the reader specifies quantities that are not in the catalog')


    def _generate_native_quantity_list(self):
        """ To be implemented by subclass. Must return an iterator"""
        raise NotImplementedError


    def list_all_quantities(self, include_native=False):
        """
        Return a list of all available quantities in this catalog
        """
        output = list(self._quantity_modifiers)
        if include_native:
            for q in self._native_quantities:
                if q not in output:
                    output.append(q)
        return output


    def list_all_native_quantities(self):
        """
        Return a list of all available native quantities in this catalog
        """
        return list(self._native_quantities)


    def add_quantity_modifier(self, quantity, modifier, overwrite=False):
        """
        Add a quantify modifier.

        Parameters
        ----------
        quantity : str
            name of the derived quantity to add

        modifier : None or str or tuple
            If the quantity modifier is a tuple of length >=2 and the first element is a callable,
            it should be in the formate of `(callable, native quantity 1,  native quantity 2, ...)`.
            And the modifier would work as callable(native quantity 1,  native quantity 2, ...)
            If the quantity modifier is None, the quantity will be used as the native quantity name
            Otherwise, the modifier would be use directly as a native quantity name

        overwrite : bool, optional
            If False and quantity are already specified in _quantity_modifiers, raise an ValueError
        """
        if quantity in self._quantity_modifiers and not overwrite:
            raise ValueError('quantity `{}` already exists'.format(quantity))
        self._quantity_modifiers[quantity] = modifier


    def has_quantities(self, quantities, include_native=True):
        """
        Check if all quantities specified are available in this galaxy catalog

        Parameters
        ----------
        quantities : iterable
            a list of quantity names to check

        include_native : bool, optional
            whether or not to include native quantity names when checking

        Returns
        -------
        has_quantities : bool
            True if the quantities are all available; otherwise False
        """
        quantities = {quantities} if isinstance(quantities, basestring) else set(quantities)

        if include_native:
            return all(q in self._native_quantities for q in self._translate_quantities(quantities))
        else:
            return all(q in self._quantity_modifiers for q in quantities)


    def _translate_quantity(self, quantity_requested):
        modifier = self._quantity_modifiers.get(quantity_requested, self._default_quantity_modifier)

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
        modifier = self._quantity_modifiers.get(quantity_requested, self._default_quantity_modifier)

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
        if filters:
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
        if isinstance(quantities, basestring):
            quantities = {quantities}

        quantities = set(quantities)
        if not quantities:
            raise ValueError('You must set `quantities`.')

        if not all(q in self._native_quantities for q in self._translate_quantities(quantities)):
            raise ValueError('Some quantities are not available in this catalog')

        return quantities


    def _preprocess_requested_filters(self, filters):
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

        return pre_filters if pre_filters else None, post_filters


    def _get_quantities_iter(self, quantities, pre_filters, post_filters, return_ndarray=False):
        for dataset in self._iter_native_dataset(pre_filters):

            if pre_filters:
                data = self._load_quantities(self._get_quantities_from_filters(pre_filters), dataset)
                mask = self._get_mask_from_filter(pre_filters, data)
                if mask is not None:
                    if not mask.any():
                        continue
                    if mask.all():
                        mask = None
            else:
                data = dict()
                mask = None

            rest_quantities = quantities.union(self._get_quantities_from_filters(post_filters))
            for q in set(data).difference(rest_quantities):
                del data[q]
            data.update(self._load_quantities(rest_quantities.difference(set(data)), dataset))
            mask = self._get_mask_from_filter(post_filters, data, mask)
            for q in set(data).difference(quantities):
                del data[q]
            if mask is not None and not mask.all():
                for q in data:
                    data[q] = data[q][mask]
            del mask
            yield _dict_to_ndarray(data) if return_ndarray else data
            del data


    def _concatenate_quantities(self, quantities, pre_filters, post_filters):
        requested_data = defaultdict(list)
        for data in self._get_quantities_iter(quantities, pre_filters, post_filters):
            for q in quantities:
                requested_data[q].append(data[q])
        return {q: np.concatenate(requested_data[q]) if requested_data[q] else np.array([]) for q in quantities}


    def get_quantities(self, quantities, filters=None, return_ndarray=False, return_hdf5=None, return_iterator=False):
        """
        Fetch quantities from this galaxy catalog.

        Parameters
        ----------
        quantities : str or list of str or tuple of str
            quantities to fetch

        filters : list of tuple, optional
            filters to apply. Each filter should be in the format of (callable, str, str, ...)

        return_ndarray : bool, optional
            return an structured ndarray if True. default is False, return a dict object
            this option is ignored if `return_hdf5` is True

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
            return self._get_quantities_iter(quantities, pre_filters, post_filters, return_ndarray)

        d = self._concatenate_quantities(quantities, pre_filters, post_filters)
        return _dict_to_ndarray(d) if return_ndarray else d


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
