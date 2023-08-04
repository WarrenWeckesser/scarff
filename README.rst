scarff
======

An ARFF file writer that handles NumPy arrays and SciPy sparse matrices.

Limitations:

* ``relational`` attributes are not supported.
* The ``dateformat`` parameter accepts a format string that defines
  the output format for ``date`` attributes.  ARFF uses the Java
  SimpleDateFormat specification for the format string.  Only a subset
  of the SimpleDateFormat patterns are supported by ``savearff``.


Examples
--------

Initial imports::

    >>> import sys
    >>> import numpy as np
    >>> from scarff import savearff

**NumPy array of integers**

``a`` is a 2-d array of integers.  The default attribute names generated
by ``savearff`` for each column are ``f0``, ``f1``, etc.  Here we
override that default and assign each column an attribute name with the
``attributes`` parameter::

    >>> a = np.array([[1, 2, 3], [9, 7, 6], [2, 2, 8], [4, 2, 3]])
    >>> savearff(sys.stdout, a, attributes=['x0', 'y0', 'z0'],
    ...          relation='points')
    @relation points

    @attribute x0 integer
    @attribute y0 integer
    @attribute z0 integer

    @data
    1,2,3
    9,7,6
    2,2,8
    4,2,3

**NumPy array with a structured dtype**

In this example, we have a structured array with a data type
that has four fields.  ``savearff`` takes the attribute names
from the names of the fields in the data type.  This example
also shows the use of a ``date`` attribute::

    >>> dt = np.dtype([('id', int), ('strength', float), ('key', 'U8'),
    ...                ('timestamp', 'datetime64[s]')])
    >>> m = np.array([(233, 1.75, 'QXX34', '2011-05-04T13:12:04'),
    ...               (154, 3.25, 'QXX99', '2011-05-04T13:47:43'),
    ...               (199, 2.16, 'QXZ55', '2011-05-04T14:41:02'),
    ...               (198, 2.32, 'QXZ59', '2011-05-04T15:28:19')], dtype=dt)
    >>> savearff(sys.stdout, m, relation='measurements',
    ...          dateformat='yyyy-MM-dd HH:mm:ss')
    @relation measurements

    @attribute id integer
    @attribute strength real
    @attribute key string
    @attribute timestamp date "yyyy-MM-dd HH:mm:ss"

    @data
    233,1.75,"QXX34","2011-05-04 13:12:04"
    154,3.25,"QXX99","2011-05-04 13:47:43"
    199,2.16,"QXZ55","2011-05-04 14:41:02"
    198,2.32,"QXZ59","2011-05-04 15:28:19"

**Nominal attributes**

ARFF files can have "nominal" attributes, in which the possible
values are restricted to a given set.  The ``nominal`` parameter
of ``savearff`` allows a column to be designated as a nominal
attribute.  The set of possible values can be derived from the
set of unique values found in the column, or can be given explicitly.
For example, here we use ``nominal={'color': True}`` to indicate that
the ``color`` attribute is nominal; the set of possible values will
be the set of unique values found in the data (in this case, ``black``,
``green`` and ``red``)::

    >>> things = [[10, 20, 'a', 'green'],
    ...           [30, 40, 'b', 'red'],
    ...           [50, 60, 'b', 'red'],
    ...           [70, 80, 'c', 'black'],
    ...           [19, 29, 'c', 'red']]
    >>> savearff(sys.stdout, things, relation='THINGS',
    ...          attributes=['x', 'y', 'code', 'color'],
    ...          nominal={'color': True})
    @relation THINGS

    @attribute x integer
    @attribute y integer
    @attribute code string
    @attribute color {black,green,red}

    @data
    10,20,"a","green"
    30,40,"b","red"
    50,60,"b","red"
    70,80,"c","black"
    19,29,"c","red"

The set of possible values can be given explicitly::

    >>> savearff(sys.stdout, things, relation='THINGS',
    ...          attributes=['x', 'y', 'code', 'color'],
    ...          nominal={'color': ['red', 'green', 'blue', 'black', 'white']})
    @relation THINGS

    @attribute x integer
    @attribute y integer
    @attribute code string
    @attribute color {red,green,blue,black,white}

    @data
    10,20,"a","green"
    30,40,"b","red"
    50,60,"b","red"
    70,80,"c","black"
    19,29,"c","red"

**SciPy sparse matrix**

SciPy is not a required dependency of ``scarff``, but ``savearff``
will recognize SciPy sparse matrices and write them to the ARFF file
using the sparse format by default::

    >>> from scipy.sparse import csc_matrix
    >>> data = [10, 20, 30, 40, 50, 60]
    >>> rows = [0, 2, 2, 3, 5, 5]
    >>> cols = [3, 1, 2, 2, 3, 4]
    >>> s = csc_matrix((data, (rows, cols)), shape=(7, 5))
    >>> s.A
    array([[ 0,  0,  0, 10,  0],
           [ 0,  0,  0,  0,  0],
           [ 0, 20, 30,  0,  0],
           [ 0,  0, 40,  0,  0],
           [ 0,  0,  0,  0,  0],
           [ 0,  0,  0, 50, 60],
           [ 0,  0,  0,  0,  0]])
    >>> savearff(sys.stdout, s, relation='links',
    ...          attributes=['a', 'b', 'c', 'd', 'e'])
    @relation links

    @attribute a integer
    @attribute b integer
    @attribute c integer
    @attribute d integer
    @attribute e integer

    @data
    {3 10}
    {}
    {1 20, 2 30}
    {2 40}
    {}
    {3 50, 4 60}
    {}

**Sparse format with a NumPy array**

A regular NumPy array can be written in the sparse format by giving
the argument ``fileformat='sparse'``::

    >>> sp = np.array([[0, 0, 99, 0, 0],
    ...                [29, 0, 0, 0, 19],
    ...                [0, 0, 0, 0, 0],
    ...                [0, 89, 0, 0, 0]])
    >>> savearff(sys.stdout, sp, fileformat='sparse',
    ...          relation='sparse example')
    @relation "sparse example"

    @attribute f0 integer
    @attribute f1 integer
    @attribute f2 integer
    @attribute f3 integer
    @attribute f4 integer

    @data
    {2 99}
    {0 29, 4 19}
    {}
    {1 89}

**Missing data**

The ``missing`` parameter allows values to be specified that
correspond to missing values.  These will appear as ``?`` in the
``@data`` section of the ARFF file.

In this example, the value 999.25 indicates a missing value::

    >>> x = np.array([[1.75, 7.93, 18.31],
    ...               [2.44, 6.62, 32.11],
    ...               [2.51, 2.25, 999.25],
    ...               [2.64, 2.33, 999.25],
    ...               [2.75, 2.83, 999.25]])
    >>> savearff(sys.stdout, x, missing=[999.25], relation='readings')
    @relation readings

    @attribute f0 real
    @attribute f1 real
    @attribute f2 real

    @data
    1.75,7.93,18.31
    2.44,6.62,32.11
    2.51,2.25,?
    2.64,2.33,?
    2.75,2.83,?

**NumPy masked array**

``savearff`` recognizes NumPy masked arrays.  Masked values in
the input array will be written as ``?`` in the ``@data`` section::

    >>> flux = np.ma.masked_array([[3.4, 2.1, 0.0, 3.4],
    ...                            [3.2, 4.8, 0.5, 3.7],
    ...                            [3.3, 2.8, 0.0, 4.1]],
    ...                           mask=[[0, 0, 1, 0],
    ...                                 [0, 0, 0, 0],
    ...                                 [0, 0, 1, 0]])
    >>> flux
    masked_array(
      data=[[3.4, 2.1, --, 3.4],
            [3.2, 4.8, 0.5, 3.7],
            [3.3, 2.8, --, 4.1]],
      mask=[[False, False,  True, False],
            [False, False, False, False],
            [False, False,  True, False]],
      fill_value=1e+20)
    >>> savearff(sys.stdout, flux, relation='flux capacitance')
    @relation "flux capacitance"

    @attribute f0 real
    @attribute f1 real
    @attribute f2 real
    @attribute f3 real

    @data
    3.4,2.1,?,3.4
    3.2,4.8,0.5,3.7
    3.3,2.8,?,4.1

**NumPy array with nested data type**

This example uses a NumPy array with a structured data type with nested
and array elements in the structure.  ``savearff`` flattens the data type
and derives attribute names from the structured data type; note how the
field names in the structured data type are used to produce the attribute
names in the output::

    >>> dt = np.dtype([('key', 'U4'),
    ...                ('position', [('x', np.float32), ('y', np.float32)]),
    ...                ('values', np.float32, 3)])
    >>> records = np.array([('A234', (1.9, -3.0), (6, 7, 2)),
    ...                     ('A555', (2.8, 0.6), (4, 2.5, 3)),
    ...                     ('B431', (2.7, 8.6), (4, 2.8, 0.2))], dtype=dt)
    >>> savearff(sys.stdout, records, relation='records')
    @relation records

    @attribute key string
    @attribute position.x real
    @attribute position.y real
    @attribute values_0 real
    @attribute values_1 real
    @attribute values_2 real

    @data
    "A234",1.9,-3,6,7,2
    "A555",2.8,0.6,4,2.5,3
    "B431",2.7,8.6,4,2.8,0.2

The above example demonstrates the default method for converting
structured data type field names to attribute names. ``savearff``
has several options to change how the names are generated.
For example::

    >>> savearff(sys.stdout, records, relation='records',
    ...          join='$', index_base=1, index_open='(', index_close=')')
    @relation records

    @attribute key string
    @attribute position$x real
    @attribute position$y real
    @attribute values(1) real
    @attribute values(2) real
    @attribute values(3) real

    @data
    "A234",1.9,-3,6,7,2
    "A555",2.8,0.6,4,2.5,3
    "B431",2.7,8.6,4,2.8,0.2

**Instance weights**

The ARFF format provides the option of saving an "instance weight" with
each instance (i.e. each row) of the data.  ``savearff`` accepts a
``weights`` argument containing a sequence of numbers.  The length of
``weights`` must equal the number of rows to be written in the ``@DATA``
section.  The weights are written to the file as an additional column in
the ``@DATA`` section, with the values enclosed in curly brackets.

For example::

    >>> dt = np.dtype([('id', int), ('x', float), ('y', float)])
    >>> samples = np.array([(300, 1.5, 1.8),
    ...                     (300, 0.8, 2.4),
    ...                     (304, 2.4, 0.5),
    ...                     (304, 3.2, 0.2)], dtype=dt)
    >>> weights = np.array([2, 2, 1, 1])
    >>> savearff(sys.stdout, samples, relation='samples', weights=weights)
    @relation samples

    @attribute id integer
    @attribute x real
    @attribute y real

    @data
    300,1.5,1.8, {2}
    300,0.8,2.4, {2}
    304,2.4,0.5, {1}
    304,3.2,0.2, {1}
