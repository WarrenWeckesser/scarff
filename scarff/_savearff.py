# Copyright © 2021 Warren Weckesser

import datetime
import numpy as np
try:
    from scipy.sparse import issparse
except ImportError:
    def issparse(a):
        return False

from ._flatten_dtype import flatten_dtype
from ._javadate import java_date_format_to_strftime


__all__ = ['savearff']


def _wrap(s):
    if ' ' in s or ',' in s:
        s = '"' + s + '"'
    return s


def _wrap_rel_or_attr(s, string_name):
    if (any(ord(c) < 32 for c in s) or
            '{' in s or '}' in s or ',' in s or '%' in s):
        raise ValueError(f"invalid character in {string_name} name: {s}. "
                         "The name may not contain characters below \\u0021, "
                         "and it may not contain '{', '}', ',' or '%%'.")
    if ' ' in s:
        s = '"' + s + '"'
    return s


def _wrap_rel(s):
    return _wrap_rel_or_attr(s, 'relation')


def _wrap_attr(s):
    return _wrap_rel_or_attr(s, 'attribute')


def _attr_type(x, nominal_values, missing):
    """
    x must be a NumPy array or a SciPy sparse matrix with a numeric
    or str data type.

    nominal_values must be one of:
        False or None:  This data was not specified as a nominal column.
        True:           This data was specified to be nominal; figure out
                        the nominal values from x.
        sequence:       The data was specified to be nominal with the given
                        nominal values.  If any value in x is not in the
                        sequence, raise an error.
    """

    if nominal_values not in [False, None]:
        attr_type = 'nominal'
        actual_values = np.unique(x)
        actual_values = [value for value in actual_values
                         if (value not in missing
                             and not np.ma.is_masked(value))]
        if nominal_values is True:
            attr_values = actual_values
        else:
            # nominal_values must be a sequence.  Check that each
            # value in x is also in nominal values.
            for value in actual_values:
                if value not in nominal_values:
                    raise ValueError("Found a value %r that is not in the "
                                     "given nominal values %r" %
                                     (value, nominal_values))
            attr_values = nominal_values

        return attr_type, attr_values

    attr_values = None
    if np.issubdtype(x.dtype, np.integer):
        attr_type = 'integer'
    elif np.issubdtype(x.dtype, np.inexact):
        attr_type = 'real'
    elif np.issubdtype(x.dtype, str) or np.issubdtype(x.dtype, bytes):
        attr_type = 'string'
    elif x.dtype.type == np.datetime64:
        attr_type = 'date'
    else:
        raise ValueError("%r not handled." % (x.dtype,))

    return attr_type, attr_values


def _attr_types(a, attributes, nominal, missing):
    """
    Returns a list of attribute types, one for each column of `a`.

    An attribute type is a tuple (string, values), where `string` is one of
    'integer', 'real', 'string' or 'nominal'.  If the string is 'nominal',
    `values` is the sequence of allowed values.  Otherwise `values` is None.
    """
    nominal_seq = [nominal.get(name, False) for name in attributes]
    if issparse(a):
        attrs = [_attr_type(a, False, missing)] * a.shape[1]
    elif a.dtype.names is None:
        # `a` must be 2-d.
        if a.ndim != 2:
            raise ValueError("numpy array must be two-dimensional, "
                             "but got a.ndim = %d" % (a.ndim,))
        attrs = [_attr_type(a[:, k], nom, missing)
                 for k, nom in zip(range(a.shape[1]), nominal_seq)]
    else:
        # structured array
        if a.ndim != 1:
            raise ValueError("numpy structured array must be one-dimensional.")
        attrs = [_attr_type(a[name], nom, missing)
                 for name, nom in zip(a.dtype.names, nominal_seq)]
    return attrs


def _object_to_array(a, attributes):
    """
    `a` is expected to be a 2-d array-like sequence of sequences.
    """
    # We let numpy take care of figuring out the data types of the
    # columns by converting `a` to a structured array.
    # XXX (future enhancement) This code makes a lot of copies of the
    #     data in `a`.
    ncols = len(a[0])
    columns = [np.array(col) for col in zip(*a)]
    if any(len(row) != ncols for row in a):
        raise ValueError('all rows of `a` must have the same length')

    if attributes is None:
        attributes = ['f%i' % k for k in range(len(columns))]
    dt = np.dtype([(name, col.dtype)
                   for name, col in zip(attributes, columns)])

    new_a = np.empty(len(a), dtype=dt)
    for name, col in zip(attributes, columns):
        new_a[name] = col

    return new_a


def _myisnan(x):
    try:
        result = np.isnan(x)
    except TypeError:
        result = False
    return result


def _item_matches(item, values):
    nan_in_values = any(_myisnan(t) for t in values)
    return item in values or (_myisnan(item) and nan_in_values)


_fmt_map = {'integer': "%i",
            'real': "%s",
            'string': '"%s"',
            'nominal': '"%s"',
            'date': None}


def _format(value, value_type):
    if value_type == 'date':
        if isinstance(value, np.datetime64):
            dt = value.item()  # Convert to datetime.datetime.
        elif isinstance(value, datetime.datetime):
            dt = value
        else:
            raise RuntimeError("don't know how to format {value} as a date.")
        out = _wrap(dt.strftime(_fmt_map['date']))
    else:
        if isinstance(value, bytes):
            value = value.decode('latin1')
        out = _fmt_map[value_type] % value
    return out


def _format_weight(w):
    return '{' + _format(w, 'real') + '}'


def _is_missing(value, missing):
    return _item_matches(value, missing) or np.ma.is_masked(value)


def _write_dense_data(f, data, types, missing=None, weights=None):
    if missing is None:
        missing = []
    # XXX There are probably optimizations of the string handling
    # that can be done here...
    for i, row in enumerate(data):
        line = ','.join(('?' if _is_missing(value, missing)
                         else _format(value, typ))
                        for typ, value in zip(types, row))
        if weights is not None:
            line = ', '.join([line, _format_weight(weights[i])])
        f.write(line + '\n')


def _write_sparse_data(f, data, types, missing=None, weights=None):
    if missing is None:
        missing = []
    for i, row in enumerate(data):
        f.write('{')
        line = ', '.join((('%i ?' % k) if _is_missing(value, missing)
                          else ("%i " + _format(value, typ)) % (k,))
                         for k, (typ, value) in enumerate(zip(types, row))
                         if (np.ma.is_masked(value) or value != 0))
        f.write(line)
        f.write('}')
        if weights is not None:
            f.write(', ' + _format_weight(weights[i]))
        f.write('\n')


def _blankline(f, style):
    if style != 'compact':
        f.write('\n')


def savearff(fileobj, a, *, attributes=None, relation=None,  missing=None,
             nominal=None, fileformat=None, style=None, realformat="%s",
             dateformat=None, weights=None,
             join='.', index_open='_', index_close='',
             index_base=0, multiindex_join=None, comments=None):
    """
    Write an array to an ARFF file.

    This function does not support "relational" attribute types.
    The set of patterns accepted by the ``dateformat`` parameter is
    a subset of those accepted by the Java SimpleDateFormat.

    Parameters
    ----------
    a : array_like, or sparse matrix
        `a` can be a 2-d array-like python object, a 2-d NumPy array
        (including masked arrays), a 1-d structured numpy array, or a SciPy
        sparse matrix with a basic numeric data dtype (integer or floating
        point).  If `a` is an instance of `numpy.matrix` (the `ndarray`
        subclass) with a structured data type, the shape of `a` must be
        (m, 1).
    attributes : list of str, optional
        `attributes` can be a list of strings.  Each column of `a` is
        considered to be an attribute; `attributes` gives the names of the
        attributes.  If given, `len(attributes)` must equal the number of
        columns of `a`.  If `attributes` is not given, and `a` is a structured
        array, the field names of `a` are used.  Otherwise, when `attributes`
        is not given, the names "f0", "f1", ..., are used.
    relation : str, optional
        If given, `relation` must be a string.  It defines the value written
        in the @relation line in the ARFF file.  If `relation` is not given,
        the string "undefined" is used.
    missing : list, optional
        If a value in `a` is also in the list `missing`, it will be replaced
        in the output file with a question mark.  (`missing` must be a list,
        even if there is only one value that indicates a missing value.)
    nominal : dict, optional
        `nominal`, if given, must be a dictionary.  Including an attribute
        name as a key in the dictionary with a value that is not None or
        False declares that attribute to be a nominal type.  If the value
        associated with the key is True, the values of the nominal type will
        be the set of unique values found in the corresponding column of `a`.
        Otherwise, the value must be a collection of strings.  When the
        nominal values are given explicitly, an error will be raised if a
        value is found in the corresponding column of `a` that is not in the
        given collection of values and is not in `missing`.
    fileformat : str, optional
        `fileformat`, if given, must be either "dense" or "sparse".
        The default is "sparse" for SciPy sparse matrices, and "dense" for
        anything else.
    style : str, optional
        By default, a few comments are included in the ARFF file, including
        a timestamp, and blank lines separate the comments, relation,
        attributes and data sections.  When `style` is "compact", the
        comments and the blank lines are not written to the file.
    realformat : str, optional
        `realformat` is the format string used for *real* numeric values
        only.  The default is "%s".
    dateformat : str, optional
        If given, this string must conform to the Java SimpleDateFormat
        specification.  Only a subset of the SimpleDateFormat mini-language
        is supported.  This is the format that will be written as the format
        in @attribute line for date attributes.  If it is not given, a format
        will not be written in the @attribute lines, and the date format will
        be the default, which is the ISO-8601 combined date and time format
        `yyyy-MM-dd'T'HH:mm:ss`.
    weights : sequence of numeric values
        Instance weights.  If given, this must be a sequence of numeric
        values with the same length as the first dimension of the input
        array `a`.  If not given, no weights are written to the ARFF file.
    join : str, optional
        When `a` has a structured data type, the field names must be
        converted to flattened names.  This argument defines the string
        used to join the field names of subfields to the field name of
        the parent field.  See examples of its use in the Examples
        section below.
    index_open : str, optional
        When `a` has a structured data type with a field that is itself
        an array, the elements of that array must be interpreted as
        separate fields, and each element must be assigned an attribute
        name.  This string is used as the opening of an indexing-like
        expression in the attributes name. See examples of its use in the
        Examples section below.
    index_close : str, optional
        When `a` has a structured data type with a field that is itself
        an array, the elements of that array must be interpreted as
        separate fields, and each element must be assigned an attribute
        name.  This string is used as the closing of an indexing-like
        expression in the attributes name.   See examples of its use in
        the Examples section below.
    index_base : int, optional
        When `a` has a structured data type with a field that is itself
        an array, the elements of that array must be interpreted as
        separate fields, and each element must be assigned an attribute
        name.  Included as part of the name is an indexing operation using
        integer indices. This argument sets the numerical value of the
        first index.  The default is 0.  See examples of its use in the
        Examples section below.
    multiindex_join : str, optional
        When `a` has a structured data type with a field that is itself
        an array, and that array is multidimensional, the elements of
        that array must be interpreted as separate fields, and each
        element must be assigned an attribute name.  Included as part
        of the name is an indexing operation with multidimensional indexing.
        This string is used to join the integer indices.  The default
        behavior is to generate a name that includes indices of the
        flattened subfield.  See examples of its use in the Examples
        section below.
    comments : list of str
        The strings in this list are written as comments at the top
        of the output file.  The comment character and a space is added
        automatically to the beginning of each string in `comments`.

    See Also
    --------
    scipy.arff.loadarff

    Examples
    --------
    >>> import io
    >>> import sys
    >>> from scarff import savearff

    The following will print the lines in `s` will a vertical bar at the
    beginning.  This will help distinguish blank lines in the output from
    other blank lines in this docstring.

    >>> def bprint(s):
    ...     for line in s.splitlines():
    ...         print("|%s" % line)
    ...

    The first example is a numpy array containing a `nan` value.

    >>> x = np.array([[1.5, 2.25, np.nan], [5.5, 6.75, 7.5]])

    The `missing` argument is used to convert the `nan`  to '?' in the
    output file.  Note that the relation name is "undefined", and the
    attributes names were automatically set to "f0", "f1" and "f2".

    >>> f = io.StringIO()
    >>> savearff(f, x, missing=[np.nan])
    >>> bprint(f.getvalue())
    |@relation undefined
    |
    |@attribute f0 real
    |@attribute f1 real
    |@attribute f2 real
    |
    |@data
    |1.5,2.25,?
    |5.5,6.75,7.5


    The next example is a NumPy matrix.  The example shows the use of
    the `relation`, `attributes` and `missing` arguments.

    (From here on, the argument `style="compact"` is used, and the
    ARFF file will be printed to `sys.stdout`.)

    >>> M = np.matrix([[1, 2, 3], [-1, 5, 6]])
    >>> savearff(sys.stdout, M, relation='foo', attributes=['a', 'b', 'c'],
    ...          missing=[-1], style="compact")
    @relation foo
    @attribute a integer
    @attribute b integer
    @attribute c integer
    @data
    1,2,3
    ?,5,6

    This example demonstrates saving an array with a structured data type.

    >>> dt = np.dtype([('r', np.float64), ('code', np.int32), ('s', 'U8')])
    >>> w = np.array([(1.25, 42, 'FOO'), (2.31, 19, 'BAR'), (3.33, 17, 'FOO')],
    ...              dtype=dt)

    Note that the attribute names and types are taken from the array's
    data type.

    >>> savearff(sys.stdout, w, relation='XYZZY', style="compact")
    @relation XYZZY
    @attribute r real
    @attribute code integer
    @attribute s string
    @data
    1.25,42,"FOO"
    2.31,19,"BAR"
    3.33,17,"FOO"

    ARFF attributes can have a nominal type, where a field is one of a
    fixed set of values.  The following example uses the same array `w` to
    show how to tell `savearff` that the "s" attribute should be saved as a
    nominal attribute.  By passing in a dictionary with key "s" whose value
    is simply True, `savearff` will use the unique values found in the
    field as the nominal values.  (The value in the dictionary may also be
    set to a predefined list of strings, if the allowed values are known
    already.)

    >>> savearff(sys.stdout, w, relation='XYZZY', nominal=dict(s=True),
    ...          style="compact")
    @relation XYZZY
    @attribute r real
    @attribute code integer
    @attribute s {BAR,FOO}
    @data
    1.25,42,"FOO"
    2.31,19,"BAR"
    3.33,17,"FOO"

    The following example shows how the names of the attributes are
    generated when the structured data type has a nested structure,
    and when there are fields that have dimensions.  In this data type,
    `loc` has two subfields, `x` and `y`, and the `matrix` field is an
    array with shape (2, 2).

    >>> dt = np.dtype([('loc', [('x', np.float32), ('y', np.float32)]),
    ...                ('matrix', np.float32, (2, 2))])
    >>> x = np.array([((1.0, 2.0), [[0.25, 0.75], [0.75, 0.25]]),
    ...               ((3.0, 4.0), [[0.75, 0.25], [0.5, 0.5]])], dtype=dt)

    Note that the components of `loc` are given the attributes names
    "loc.x" and "loc.y".  The four elements of the `matrix` field were
    given the names "matrix_0", "matrix_1", "matrix_2" and "matrix_3".

    >>> savearff(sys.stdout, x, style="compact")
    @relation undefined
    @attribute loc.x real
    @attribute loc.y real
    @attribute matrix_0 real
    @attribute matrix_1 real
    @attribute matrix_2 real
    @attribute matrix_3 real
    @data
    1,2,0.25,0.75,0.75,0.25
    3,4,0.75,0.25,0.5,0.5

    Here the same array is used, but several options are specified
    to customize how the attribute names of the structured array fields
    are generated.

    >>> savearff(sys.stdout, x, join='$', multiindex_join=';', index_base=1,
    ...          index_open='(', index_close=')', style="compact")
    @relation undefined
    @attribute loc$x real
    @attribute loc$y real
    @attribute matrix(1;1) real
    @attribute matrix(1;2) real
    @attribute matrix(2;1) real
    @attribute matrix(2;2) real
    @data
    1,2,0.25,0.75,0.75,0.25
    3,4,0.75,0.25,0.5,0.5

    (It is tempting to use `multiindex_join=','`, but this will result in
    an error.  A comma is not allowed in the attribute names of an ARFF
    file.)

    The last example demonstrates that `savearff` accepts SciPy sparse
    matrices.  This sparse matrix `s` has three nonzero values.

    >>> from scipy.sparse import csr_matrix
    >>> s = csr_matrix(((10, 20, 30), ((0, 2, 2), (2, 0, 1))), shape=(3, 4))
    >>> s.A
    array([[ 0,  0, 10,  0],
           [ 0,  0,  0,  0],
           [20, 30,  0,  0]])

    When given a sparse matrix, the ARFF file is written in the sparse
    format by default. (Use `fileformat="dense"` to write the file in the
    dense format.  To write other array types in the sparse format, use
    `fileformat="sparse"`.)

    >>> savearff(sys.stdout, s, relation='sparse example', style="compact")
    @relation "sparse example"
    @attribute f0 integer
    @attribute f1 integer
    @attribute f2 integer
    @attribute f3 integer
    @data
    {2 10}
    {}
    {0 20, 1 30}

    """
    # XXX The implementation of savearff is not optimized for speed or
    # memory use.  Depending on the type of object passed in, the code
    # might end up making multiple temporary copies of the input array.

    if relation is None:
        relation = 'undefined'

    if nominal is None:
        nominal = {}

    if missing is None:
        missing = []

    if fileformat is None:
        if issparse(a):
            fileformat = 'sparse'
        else:
            fileformat = 'dense'
    elif fileformat not in ['sparse', 'dense']:
        raise ValueError(f"unknown fileformat {fileformat}")

    if dateformat is None:
        # The SimpleDateFormat for the ISO 8601 combined data and time format:
        sdf = "yyyy-MM-dd'T'HH:mm:ss"
    else:
        sdf = dateformat
    strftime_fmt = java_date_format_to_strftime(sdf)
    _fmt_map['date'] = strftime_fmt

    # These arguments are only used by passing them on to flatten_dtype().
    flatten_kwargs = dict(join=join,
                          index_open=index_open, index_close=index_close,
                          multiindex_join=multiindex_join,
                          index_base=index_base)

    _fmt_map['real'] = realformat

    if isinstance(a, np.ndarray):
        if isinstance(a, np.matrix):
            if a.dtype.names is not None:
                if a.shape[1] != 1:
                    raise ValueError("a numpy.matrix instance with a "
                                     "structured dtype must have shape "
                                     "(m, 1).")
                else:
                    # Convert the numpy `matrix` to a 1-d numpy `ndarray`.
                    a = a.getA1()
            else:
                # Convert to a numpy ndarray.
                a = a.getA()
        if a.dtype == object:
            a = _object_to_array(a, attributes)
        if a.dtype.names is not None:
            # `a` has a structured dtype.  Flatten the dtype, if necessary.
            _names, _types = flatten_dtype(a.dtype, **flatten_kwargs)
            flat_dtype = np.dtype(list(zip(_names, _types)))
            # XXX Is using `view` dangerous?  What if the fields have
            # unexpected offsets?
            a = a.view(flat_dtype)
    elif not issparse(a):
        # `a` is some other python object (not a numpy array or scipy
        # sparse matrix).  Convert it to a structured array.
        a = _object_to_array(a, attributes)

    if weights is not None:
        if issparse(a):
            n = a.shape[0]
        else:
            n = len(a)
        if len(weights) != n:
            raise ValueError('The length of weights must equal the number of '
                             'rows of the input array `a`.')

    if attributes is None:
        if a.dtype.names is None:
            attributes = ["f" + str(k) for k in range(a.shape[1])]
        else:
            attributes = a.dtype.names

    for key in nominal:
        if key not in attributes:
            raise ValueError(f"key {key} of `nominal` is not one of the "
                             f"attributes: {attributes}.")

    attr_types = _attr_types(a, attributes, nominal, missing)
    types = [t for t, _ in attr_types]

    # XXX More validation needed: check that len(attributes) equals
    #     the number of columns of `a`.

    # ---------------------------------------------------------------
    # Done validating and munging the input, now write the file.
    # ---------------------------------------------------------------

    if isinstance(fileobj, str):
        f = open(fileobj, "w")
    else:
        f = fileobj

    if comments:
        for line in comments:
            f.write(f"% {line}\n")
            _blankline(f, style)

    # Write the @relation line.
    f.write(f'@relation {_wrap_rel(relation)}\n')

    _blankline(f, style)

    # Write the @attribute lines.
    for name, attr_type in zip(attributes, attr_types):
        typ, values = attr_type
        f.write(f'@attribute {_wrap_attr(name)} ')
        if typ == 'date':
            f.write('date')
            if dateformat is not None:
                f.write(' ' + _wrap(dateformat) + '\n')
            else:
                f.write('\n')
        else:
            if values is None:
                f.write('%s\n' % (typ,))
            else:
                values_str = ','.join(_wrap(name) for name in values)
                line = '{' + values_str + '}'
                f.write(line + '\n')

    _blankline(f, style)

    # Write the @data section.
    f.write("@data\n")

    if fileformat == "sparse":
        write_data = _write_sparse_data
    else:
        write_data = _write_dense_data

    if issparse(a):
        # SciPy sparse array.  Densify one row at a time.
        for k in range(a.shape[0]):
            row = a.getrow(k).A
            w = None if weights is None else [weights[k]]
            write_data(f, row, types=types, missing=missing, weights=w)
    else:
        # Numpy array
        write_data(f, a, types=types, missing=missing, weights=weights)

    if f != fileobj:
        f.close()
