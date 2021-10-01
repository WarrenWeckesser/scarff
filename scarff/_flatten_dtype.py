# Copyright © 2021 Warren Weckesser

import itertools


def flatten_dtype(dt, join='.', index_open='_', index_close='',
                  multiindex_join=None, index_base=0):
    """
    Create a set of flattened names and data types for a structured data type.

    Parameters
    ----------
    dt : numpy `dtype` object of a structure array.
        dt *must* be a structured data type.
    join : str
        String used to join a subfield to its parent.
    index_open : str
        String use between a field name and the numeric index (or indices).
    index_close : str
        String used at the end of an index of a field.
    multiindex_join : str
        When this is not None, it is used to separate the numerical values
        of a multidimensional index.
        If multiindex_join is None, the indices of the flattened shape are
        used.

    Returns
    -------
    names : list of str
        The list of the field names of the flattened dtype.
    types : list of numpy dtypes
        The corresponding dtypes of the individual fields.

    Notes
    -----
    The function returns two lists, `names` and `types`.  To actually create a
    numpy dtype from these, use `dtype(zip(names, types))`:

    >>> names, types = flatten_dtype(my_nested_dtype)
    >>> my_flat_dtype = np.dtype(zip(names, types))

    Numpy has `numpy.lib.npyio.flatten_dtype`, but that function does
    not create names, and it does not handle a data type nested more than
    one level deep.

    Examples
    --------
    A basic example:

    >>> dt = np.dtype([('x', 'f8'), ('y', 'f8')])
    >>> names, types = flatten_dtype(dt)
    >>> zip(names, types)
    [('x', dtype('float64')), ('y', dtype('float64'))]

    A more complicated structure:

    >>> dt = np.dtype([('name', 'S12'),
    ...                ('pos', [('x', 'f8'), ('y', 'f8')]),
    ...                ('data', [('foo', 'i2', 2), ('bar', 'u2')], (2,2))])
    ...
    >>> names, types = flatten_dtype(dt)
    >>> zip(names, types)
    [('name', dtype('S12')),
     ('pos.x', dtype('float64')),
     ('pos.y', dtype('float64')),
     ('data_0.foo_0', dtype('int16')),
     ('data_0.foo_1', dtype('int16')),
     ('data_0.bar', dtype('uint16')),
     ('data_1.foo_0', dtype('int16')),
     ('data_1.foo_1', dtype('int16')),
     ('data_1.bar', dtype('uint16')),
     ('data_2.foo_0', dtype('int16')),
     ('data_2.foo_1', dtype('int16')),
     ('data_2.bar', dtype('uint16')),
     ('data_3.foo_0', dtype('int16')),
     ('data_3.foo_1', dtype('int16')),
     ('data_3.bar', dtype('uint16'))]

    Customize how fields with shapes are converted to names:

    >>> names, types = flatten_dtype(dt, index_open='[', index_close=']')
    >>> names
    ['name',
     'pos.x',
     'pos.y',
     'data[0].foo[0]',
     'data[0].foo[1]',
     'data[0].bar',
     'data[1].foo[0]',
     'data[1].foo[1]',
     'data[1].bar',
     'data[2].foo[0]',
     'data[2].foo[1]',
     'data[2].bar',
     'data[3].foo[0]',
     'data[3].foo[1]',
     'data[3].bar']

    Further customization:

    >>> names, types = flatten_dtype(dt, join='$',
    ...                              index_open='(', index_close=')',
    ...                              multiindex_join=',', index_base=1)
    ...
    >>> names
    ['name',
     'pos$x',
     'pos$y',
     'data(1,1)$foo(1)',
     'data(1,1)$foo(2)',
     'data(1,1)$bar',
     'data(1,2)$foo(1)',
     'data(1,2)$foo(2)',
     'data(1,2)$bar',
     'data(2,1)$foo(1)',
     'data(2,1)$foo(2)',
     'data(2,1)$bar',
     'data(2,2)$foo(1)',
     'data(2,2)$foo(2)',
     'data(2,2)$bar']

    """
    field_names = dt.names
    names = []
    types = []
    for field_name in field_names:
        field_dt = dt.fields[field_name][0]
        shape = field_dt.shape
        if len(shape) > 0:
            field_dt = field_dt.subdtype[0]

        if field_dt.names is not None:
            # Recurse here...
            subfield_names, subfield_types = \
                flatten_dtype(field_dt, join=join,
                              index_open=index_open, index_close=index_close,
                              multiindex_join=multiindex_join,
                              index_base=index_base)

        for k, idx in enumerate(itertools.product(*[range(j) for j in shape])):
            if len(shape) == 0:
                fname = field_name
            else:
                if len(idx) > 1 and multiindex_join is not None:
                    index_str = multiindex_join.join(str(j+index_base)
                                                     for j in idx)
                    fname = field_name + index_open + index_str + index_close
                else:
                    fname = (field_name + index_open + str(k + index_base)
                             + index_close)
            if field_dt.names is not None:
                # Prepend the current field name to the subfield names.
                current_subfield_names = [fname + join + subfield_name
                                          for subfield_name in subfield_names]
                names.extend(current_subfield_names)
                types.extend(subfield_types)
            else:
                names.append(fname)
                types.append(field_dt)

    return names, types
