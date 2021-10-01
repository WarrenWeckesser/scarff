# Copyright © 2021 Warren Weckesser

import numpy as np
from .._flatten_dtype import flatten_dtype


def test_basic1():
    dt = np.dtype([('x', np.float32)])
    names, types = flatten_dtype(dt)
    assert names == ['x']
    assert types == [np.dtype('f')]


def test_basic2():
    dt = np.dtype([('x', 'f8'), ('y', 'f8')])
    names, types = flatten_dtype(dt)
    assert names == ['x', 'y']
    assert types == [np.dtype('f8'), np.dtype('f8')]


def test_nested1():
    dt = np.dtype([('z', 'd', 2), ('w', [('a', 'S8'), ('b', 'S8')])])
    names, types = flatten_dtype(dt)
    assert names == ['z_0', 'z_1', 'w.a', 'w.b']
    assert types == [np.dtype('d'), np.dtype('d'),
                     np.dtype('S8'), np.dtype('S8')]


def test_nested2():
    dt = np.dtype([('name', 'S12'),
                   ('pos', [('x', 'f8'), ('y', 'f8')]),
                   ('data', [('foo', 'i2', 2), ('bar', 'u2')], (2, 2))])
    names, types = flatten_dtype(dt)
    assert names == ['name', 'pos.x', 'pos.y',
                     'data_0.foo_0', 'data_0.foo_1', 'data_0.bar',
                     'data_1.foo_0', 'data_1.foo_1', 'data_1.bar',
                     'data_2.foo_0', 'data_2.foo_1', 'data_2.bar',
                     'data_3.foo_0', 'data_3.foo_1', 'data_3.bar']
    assert types == ([np.dtype('S12')]
                     + [np.dtype('f8'), np.dtype('f8')]
                     + [np.dtype('i2'), np.dtype('i2'), np.dtype('u2')]
                     + [np.dtype('i2'), np.dtype('i2'), np.dtype('u2')]
                     + [np.dtype('i2'), np.dtype('i2'), np.dtype('u2')]
                     + [np.dtype('i2'), np.dtype('i2'), np.dtype('u2')])


def test_nested3():
    dt = np.dtype([('name', 'S12'),
                   ('pos', [('x', 'f8'), ('y', 'f8')]),
                   ('data', [('foo', 'i2', 2), ('bar', 'u2')], (2, 2))])
    names, types = flatten_dtype(dt, index_open='[', index_close=']')
    assert names == ['name', 'pos.x', 'pos.y',
                     'data[0].foo[0]', 'data[0].foo[1]', 'data[0].bar',
                     'data[1].foo[0]', 'data[1].foo[1]', 'data[1].bar',
                     'data[2].foo[0]', 'data[2].foo[1]', 'data[2].bar',
                     'data[3].foo[0]', 'data[3].foo[1]', 'data[3].bar']
    assert types == ([np.dtype('S12')]
                     + [np.dtype('f8'), np.dtype('f8')]
                     + [np.dtype('i2'), np.dtype('i2'), np.dtype('u2')]
                     + [np.dtype('i2'), np.dtype('i2'), np.dtype('u2')]
                     + [np.dtype('i2'), np.dtype('i2'), np.dtype('u2')]
                     + [np.dtype('i2'), np.dtype('i2'), np.dtype('u2')])


def test_nested4():
    dt = np.dtype([('name', 'S12'),
                   ('pos', [('x', 'f8'), ('y', 'f8')]),
                   ('data', [('foo', 'i2', 2), ('bar', 'u2')], (2, 2))])
    names, types = flatten_dtype(dt, join='$', index_open='(', index_close=')',
                                 multiindex_join=',', index_base=1)
    assert names == ['name', 'pos$x', 'pos$y',
                     'data(1,1)$foo(1)', 'data(1,1)$foo(2)', 'data(1,1)$bar',
                     'data(1,2)$foo(1)', 'data(1,2)$foo(2)', 'data(1,2)$bar',
                     'data(2,1)$foo(1)', 'data(2,1)$foo(2)', 'data(2,1)$bar',
                     'data(2,2)$foo(1)', 'data(2,2)$foo(2)', 'data(2,2)$bar']
    assert types == ([np.dtype('S12')]
                     + [np.dtype('f8'), np.dtype('f8')]
                     + [np.dtype('i2'), np.dtype('i2'), np.dtype('u2')]
                     + [np.dtype('i2'), np.dtype('i2'), np.dtype('u2')]
                     + [np.dtype('i2'), np.dtype('i2'), np.dtype('u2')]
                     + [np.dtype('i2'), np.dtype('i2'), np.dtype('u2')])
