scarff
======

An ARFF file writer that handles NumPy arrays and SciPy sparse matrices.

A few examples
--------------

Initial imports::

    >>> import sys
    >>> import numpy as np
    >>> from scarff import savearff

A 2-d array of integers; we'll assign each column an attribute name
with the ``attributes`` parameter::

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

A structured array, with four fields; the attribute names are
taken from the names of the fields in the data type::

    >>> dt = np.dtype([('id', int), ('strength', float), ('key', 'U8'),
    ...                ('time', np.dtype(np.datetime64, 's'))
    >>> m = np.array([(233, 1.75, 'QXX34', '2011-05-04T13:12:04'),
    ...               (154, 3.25, 'QXX99', '2011-05-04T13:47:43'),
    ...               (199, 2.16, 'QXZ55', '2011-05-04T14:41:02'),
    ...               (198, 2.32, 'QXZ59', '2011-05-04T15:28:19')], dtype=dt)
    >>> savearff(sys.stdout, m, relation='measurements',
    ...          dateformat='yyyy-mm-dd hh:mm:ss')
    @relation measurements

    @attribute id integer
    @attribute strength real
    @attribute key string
    @attribute time date "yyyy-mm-dd hh:mm:ss"

    @data
    233,1.75,"QXX34","2011-12-04 01:12:04"
    154,3.25,"QXX99","2011-47-04 01:47:43"
    199,2.16,"QXZ55","2011-41-04 02:41:02"
    198,2.32,"QXZ59","2011-28-04 03:28:19"

A SciPy sparse matrix::

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

A NumPy masked array; note that the masked values are converted
to ``?`` in the ``@data`` section of the output::

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

A NumPy array with a structured data type with nested and array elements
in the structure.  ``savearff`` flattens the data type and derives
attribute names from the structured data type; note how the field names
in the structured data type are used to produce the attribute names in
the output::

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

Currently ``savearff`` does not implement the generation of "relational"
attributes.
