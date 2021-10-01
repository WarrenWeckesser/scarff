# Copyright © 2021 Warren Weckesser

import pytest
from .._javadate import java_date_format_to_strftime


@pytest.mark.parametrize('java_fmt, strftime_fmt',
                         [('yy', '%y'),
                          ('yyyy', '%Y'),
                          ('MMM', '%b'),
                          ('MMMM', '%B'),
                          ('dd', '%d'),
                          ('ddd', '0%d'),
                          ('EEE', '%a'),
                          ('EEEEE', '%A'),
                          ('a', '%p'),
                          ('HH', '%H'),
                          ('HHH', '0%H'),
                          ('hh', '%I'),
                          ('hhh', '0%I'),
                          ('mm', '%M'),
                          ('mmm', '0%M'),
                          ('ss', '%S'),
                          ('sss', '0%S'),
                          ('z', '%Z'),
                          ('zzz', '%Z'),
                          ('Z', '%z')])
def test_basic(java_fmt, strftime_fmt):
    fmt = java_date_format_to_strftime(java_fmt)
    assert fmt == strftime_fmt
