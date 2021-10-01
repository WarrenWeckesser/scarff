scarff
======

An ARFF file writer that handles NumPy arrays and SciPy sparse matrices.

A few examples
--------------

```
>>> import sys
>>> import numpy as np
>>> from scarff import savearff
```

A 2-d array of integers; we'll assign each column an attribute name
with the `attributes` parameter:

```
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
```

A structured array, with four fields; the attribute names are
taken from the names of the fields in the data type:

```
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
```

A SciPy sparse matrix:

```
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
```
