# Copyright © 2021 Warren Weckesser


import io
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_array_equal
try:
    _have_scipy = True
    from scipy.sparse import csr_matrix
except ImportError:
    _have_scipy = False
from scarff import savearff
from .parsearff import parsearff


class TestSaveArff:

    def test_simple_array(self):
        x = np.array([[10, 20], [30, 40]])
        f = io.StringIO()
        savearff(f, x)
        contents = f.getvalue()

        relation, attributes, types, data, weights = parsearff(contents)
        assert_equal(relation, 'undefined')
        assert_equal(attributes, ['f0', 'f1'])
        assert_equal(types, ['integer', 'integer'])
        assert_array_equal(data, [[str(k) for k in row] for row in x])
        assert weights == [None, None]

        dt = np.dtype([('a', np.float64), ('b', np.float64)])
        x = np.array([(1.0, 2.0), (10.0, 20.0), (80.0, 90.0)], dtype=dt)
        f = io.StringIO()
        savearff(f, x, relation="XYZZY")
        contents = f.getvalue()

        relation, attributes, types, data, weights = parsearff(contents)
        assert_equal(relation, 'XYZZY')
        assert_equal(attributes, ['a', 'b'])
        assert_equal(types, ['real', 'real'])
        assert_array_equal(data, [[("%g" % value) for value in row]
                                  for row in x])
        assert weights == [None]*len(x)

    @pytest.mark.skipif(not _have_scipy, reason="SciPy not installed")
    def test_sparse_matrix(self):
        s = csr_matrix([[10, 0, 0, 20], [0, 0, 0, 0], [0, 30, 0, 0]])

        attrs = ['a', 'b', 'c', 'd']
        rel = 'special'
        for fileformat in ['sparse', 'dense']:
            f = io.StringIO()
            savearff(f, s, attributes=attrs, relation=rel,
                     fileformat=fileformat)
            contents = f.getvalue()

            relation, attributes, types, data, weights = parsearff(contents)
            assert_equal(relation, rel)
            assert_equal(attributes, attrs)
            assert_equal(types, ['integer']*4)
            assert_equal(data, [[str(k) for k in row] for row in s.A])
            assert weights == [None]*s.shape[0]

    def test_nested_dtype(self):
        dt = np.dtype([('code', 'u2'),
                       ('pos', [('x', 'f8'), ('y', 'f8')]),
                       ('color', [('red', 'u1'),
                                  ('green', 'u1'),
                                  ('blue', 'u1')])])
        a = np.array([(2112, (1.5, 2.5), (31, 127, 255)),
                      (2113, (3.0, 4.0), (99, 98, 97)),
                      (2114, (4.5, 5.5), (0, 255, 127))], dtype=dt)

        rel = 'plate of shrimp'
        f = io.StringIO()
        savearff(f, a, relation=rel)
        contents = f.getvalue()

        relation, attributes, types, data, weights = parsearff(contents)
        assert_equal(relation, rel)
        expected_names = ['code', 'pos.x', 'pos.y',
                          'color.red', 'color.green', 'color.blue']
        assert_equal(attributes, expected_names)
        expected_types = ['integer', 'real', 'real',
                          'integer', 'integer', 'integer']
        assert_equal(types, expected_types)
        expected_data = [[str(row['code'])] +
                         [("%g" % v) for v in row['pos']] +
                         [("%g" % v) for v in row['color']]
                         for row in a]
        assert_equal(data, expected_data)
        assert weights == [None]*len(a)

    def test_shapes_chars_etc(self):
        dt = np.dtype([('a', 'u2', (2, 2)),
                       ('b', [('x', 'f8', 3), ('y', 'f8', 2)], 2)])

        a = np.array([([[10, 11], [12, 13]], [([14, 15, 16], [17, 18]),
                                              ([19, 20, 21], [22, 23])]),
                      ([[24, 25], [26, 27]], [([28, 29, 30], [31, 32]),
                                              ([33, 34, 35], [36, 37])])],
                     dtype=dt)
        rel = 'qwerty'
        f = io.StringIO()
        savearff(f, a, relation=rel)
        contents = f.getvalue()

        relation, attributes, types, data, weights = parsearff(contents)
        assert_equal(relation, rel)
        expected_names = [
            'a_0', 'a_1', 'a_2', 'a_3',
            'b_0.x_0', 'b_0.x_1', 'b_0.x_2', 'b_0.y_0', 'b_0.y_1',
            'b_1.x_0', 'b_1.x_1', 'b_1.x_2', 'b_1.y_0', 'b_1.y_1']
        assert_equal(attributes, expected_names)
        expected_types = ['integer']*4 + ['real']*10
        assert_equal(types, expected_types)
        str_values = [str(k) for k in range(10, 38)]
        expected_data = [str_values[:14], str_values[14:]]
        assert_equal(data, expected_data)
        assert weights == [None]*len(a)

        f = io.StringIO()
        savearff(f, a, relation=rel, join='$',
                 index_open='[', index_close=']')
        contents = f.getvalue()

        relation, attributes, types, data, weights = parsearff(contents)
        expected_names = [
            'a[0]', 'a[1]', 'a[2]', 'a[3]',
            'b[0]$x[0]', 'b[0]$x[1]', 'b[0]$x[2]', 'b[0]$y[0]', 'b[0]$y[1]',
            'b[1]$x[0]', 'b[1]$x[1]', 'b[1]$x[2]', 'b[1]$y[0]', 'b[1]$y[1]']
        assert_equal(attributes, expected_names)
        assert weights == [None]*len(a)

        f = io.StringIO()
        savearff(f, a, relation=rel, join='$',
                 index_open='(', index_close=')',
                 multiindex_join=';', index_base=1)
        contents = f.getvalue()

        relation, attributes, types, data, weights = parsearff(contents)
        expected_names = [
            'a(1;1)', 'a(1;2)', 'a(2;1)', 'a(2;2)',
            'b(1)$x(1)', 'b(1)$x(2)', 'b(1)$x(3)', 'b(1)$y(1)', 'b(1)$y(2)',
            'b(2)$x(1)', 'b(2)$x(2)', 'b(2)$x(3)', 'b(2)$y(1)', 'b(2)$y(2)']
        assert_equal(attributes, expected_names)
        assert weights == [None]*len(a)

    def test_masked_array(self):
        m = np.ma.masked_array([[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                               mask=[[0, 0, 1], [0, 0, 0], [0, 1, 0]])
        rel = 'MASKED ARRAY'
        f = io.StringIO()
        savearff(f, m, relation=rel)
        contents = f.getvalue()

        relation, attributes, types, data, weights = parsearff(contents)
        assert_equal(relation, rel)
        assert_equal(attributes, ['f0', 'f1', 'f2'])
        assert_equal(types, ['integer']*3)
        assert weights == [None]*m.shape[0]

        # Temporarily change the display of masked values from '--' to '?'.
        saved_disp = np.ma.masked_print_option.display()
        np.ma.masked_print_option.set_display('?')
        assert_equal(data, [[str(k) for k in row] for row in m])
        # Restore the display of masked values.
        np.ma.masked_print_option.set_display(saved_disp)

    def test_masked_structured_array(self):
        dt = np.dtype([('x', np.float64), ('y', np.float64), ('a', np.int64)])
        m = np.ma.masked_array([(1.0, 2.0, 10),
                                (3.0, 4.0, 20),
                                (5.0, 6.0, 30)],
                               mask=[(0, 0, 0),
                                     (1, 0, 0),
                                     (0, 0, 1)],
                               dtype=dt)
        rel = u'MASKED STRUCTURED ARRAY!'
        f = io.StringIO()
        savearff(f, m, relation=rel)
        contents = f.getvalue()

        relation, attributes, types, data, weights = parsearff(contents)
        assert_equal(relation, rel)
        assert_equal(attributes, m.dtype.names)
        assert_equal(types, ['real', 'real', 'integer'])
        assert_equal(data, [[("%g" % (k,) if not np.ma.is_masked(k) else '?')
                             for k in row]
                            for row in m])
        assert weights == [None]*m.shape[0]

    def test_object_array(self):
        x = [[10, 20, 1.5], [30, 40, 2.5], [50, 60, 3.5]]

        rel = 'LIST OF LISTS'
        attrs = ['item 1', 'item 2', 'item 3']
        f = io.StringIO()
        savearff(f, x, relation=rel, attributes=attrs)
        contents = f.getvalue()

        relation, attributes, types, data, weights = parsearff(contents)
        assert_equal(relation, rel)
        assert_equal(attributes, attrs)
        assert_equal(types, ['integer', 'integer', 'real'])
        assert_equal(data, [[str(k) for k in row] for row in x])
        assert weights == [None]*len(x)

    def test_no_data(self):
        dt = np.dtype([('x', np.float64), ('y', np.float64), ('a', np.int64)])
        nodata = np.array([], dtype=dt)
        rel = 'NO DATA!'
        f = io.StringIO()
        savearff(f, nodata, relation=rel)
        contents = f.getvalue()

        relation, attributes, types, data, weights = parsearff(contents)
        assert_equal(relation, rel)
        assert_equal(attributes, nodata.dtype.names)
        assert_equal(types, ['real', 'real', 'integer'])
        assert_equal(data, [])
        assert weights == []

    def test_nominal(self):
        # This also tests an input that is a list of lists of python objects
        # rather than a numpy array.
        x = [['foo', 10, 20, 'a'],
             ['bar', 30, 40, 'b'],
             ['cat', 50, 60, 'b'],
             ['dog', 70, 80, 'c'],
             ['foo', 19, 29, 'c']]
        rel = "something"

        f = io.StringIO()
        savearff(f, x, relation=rel, nominal=dict(f0=True))
        contents = f.getvalue()

        relation, attributes, types, data, weights = parsearff(contents)
        assert_equal(relation, rel)
        assert_equal(attributes, ['f0', 'f1', 'f2', 'f3'])
        assert_equal(types[1:], ['integer', 'integer', 'string'])
        assert_equal(types[0], ('nominal', ['bar', 'cat', 'dog', 'foo']))
        assert_equal(data, [[str(k) for k in row] for row in x])
        assert weights == [None]*len(x)

        f = io.StringIO()
        f0vals = ['foo', 'bar', 'cat', 'dog']
        savearff(f, x, relation=rel, nominal=dict(f0=f0vals))
        contents = f.getvalue()

        relation, attributes, types, data, weights = parsearff(contents)
        assert_equal(relation, rel)
        assert_equal(attributes, ['f0', 'f1', 'f2', 'f3'])
        assert_equal(types[1:], ['integer', 'integer', 'string'])
        assert_equal(types[0], ('nominal', f0vals))
        assert_equal(data, [[str(k) for k in row] for row in x])
        assert weights == [None]*len(x)

        f = io.StringIO()
        savearff(f, x, relation=rel, missing=['foo'], nominal=dict(f0=True))
        contents = f.getvalue()

        relation, attributes, types, data, weights = parsearff(contents)
        assert_equal(relation, rel)
        assert_equal(attributes, ['f0', 'f1', 'f2', 'f3'])
        assert_equal(types[1:], ['integer', 'integer', 'string'])
        assert_equal(types[0], ('nominal', ['bar', 'cat', 'dog']))
        assert_equal(data, [[('?' if k == 'foo' else str(k)) for k in row]
                            for row in x])
        assert weights == [None]*len(x)

    @pytest.mark.parametrize('dateformat',
                             [None, "MM/dd/yy ('American style')"])
    def test_date(self, dateformat):
        d0 = np.datetime64('2004-01-10T17:02:49', 's')
        d1 = np.datetime64('2004-01-11T15:16:09', 's')
        a = np.array([[d0]*3,
                      [d1]*3])
        f = io.StringIO()
        savearff(f, a, dateformat=dateformat)
        contents = f.getvalue()

        relation, attributes, types, data, weights = parsearff(contents)
        assert relation == 'undefined'
        assert attributes == ['f0', 'f1', 'f2']
        if dateformat is None:
            assert types == [('date', None)]*3
            assert data == [[str(d0)]*3,
                            [str(d1)]*3]
        else:
            assert types == [('date', dateformat)]*3
            assert data == [['01/10/04 (American style)']*3,
                            ['01/11/04 (American style)']*3]
        assert weights == [None]*a.shape[0]

    def test_weights(self):
        x = [[10, 20],
             [11, 21],
             [12, 22]]
        f = io.StringIO()
        savearff(f, x, relation='spiffy', weights=[0.5, 0.125, 0.375])
        contents = f.getvalue()

        relation, attributes, types, data, weights = parsearff(contents)
        assert relation == 'spiffy'
        assert attributes == ['f0', 'f1']
        assert types == ['integer', 'integer']
        assert data == [['10', '20'], ['11', '21'], ['12', '22']]
        assert weights == ['0.5', '0.125', '0.375']

    def test_weights_sparse(self):
        x = [[0, 0],
             [1, 0],
             [0, 2],
             [3, 4],
             [0, 0]]
        f = io.StringIO()
        savearff(f, x, relation='counts', weights=[1, 2, 0, 3, 4],
                 fileformat='sparse', style='compact', comments=['Well then'])
        contents = f.getvalue()

        relation, attributes, types, data, weights = parsearff(contents)
        assert relation == 'counts'
        assert attributes == ['f0', 'f1']
        assert types == ['integer', 'integer']
        assert data == [['0', '0'],
                        ['1', '0'],
                        ['0', '2'],
                        ['3', '4'],
                        ['0', '0']]
        assert weights == ['1', '2', '0', '3', '4']
