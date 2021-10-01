# Copyright © 2021 Warren Weckesser

import string


def _parse_java_simple_date_format(fmt):
    """
    Split a SimpleDateFormat into literal strings and format codes with counts.

    Examples
    --------
    >>> _parse_java_simple_date_format("'Date:' EEEEE, MMM dd, ''yy")
    ['Date: ', ('E', 5), ', ', ('M', 3), ' ', ('d', 2), ", '", ('y', 2)]

    """
    out = []
    quoted = False
    prev_c = None
    prev_count = 0
    literal_text = ''
    k = 0
    while k < len(fmt):
        c = fmt[k]
        k += 1
        if not quoted and c == "'" and k < len(fmt) and fmt[k] == "'":
            # Repeated single quote.
            if prev_c is not None:
                out.append((prev_c, prev_count))
                prev_c = None
                prev_count = 0
            literal_text += c
            k += 1
            continue
        if c == "'":
            if not quoted:
                if prev_c is not None:
                    out.append((prev_c, prev_count))
                    prev_c = None
                    prev_count = 0
                if literal_text:
                    out.append(literal_text)
                    literal_text = ''
            quoted = not quoted
            continue
        if quoted:
            literal_text += c
            continue
        if c not in string.ascii_letters:
            if prev_c is not None:
                out.append((prev_c, prev_count))
                prev_c = None
                prev_count = 0
            literal_text += c
            continue
        if c not in 'GyMdhHmsSEDFwWakKzZ':
            raise ValueError(f"unknown format character {c}")
        if literal_text != '':
            out.append(literal_text)
            literal_text = ''
        if prev_c is not None and c != prev_c:
            out.append((prev_c, prev_count))
            prev_count = 0
        prev_c = c
        prev_count += 1
    else:
        if quoted:
            raise ValueError("missing closing quote; input ends "
                             f"with '{literal_text}")
        if literal_text != '':
            out.append(literal_text)
        elif prev_c is not None:
            out.append((prev_c, prev_count))
    return out


# XXX A correct translation of the SimpleDateFormat string into
# format string for strftime isn't really feasible.
# The Java format documentation says that for "number" fields, the
# number of repetitions of the character specifies the minimum number
# of digits, using zero padding.  The strftime format string doesn't
# have an equivalent.  To stick with the idea of just converting the
# format string, the function could either (1) raise an error when the
# input can't be translated exactly, or (2) create a format string
# that is "pretty close".
#
# Overall, a better alternative is probably to define a formatting
# function that actually does the formatting of a time argument
# directly from the Java SimpleDateFormat string.
# E.g. something like
#     def strftime_java(java_format, time_value)
#         ...
#
# For now, the following takes approach (1): try to create
# a format string that matches the requested format exactly, and
# raise an error if that is not possible.  The implementation
# is currently very conservative--if a detail seems ambiguous, the
# code raise an error.  For example, the (possibly unofficial)
# documentation copied above shows an example where 'yyyyy' generates
# '1996'.  But the number of characters is supposed to indicate the
# minimum number of digts, with zero padding.  So why doesn't
# 'yyyyy' generate '01996'?  That might be part of the "special"
# handling of the year.  For now, this code only handles 'yy' and
# 'yyyy'.

def java_date_format_to_strftime(fmt):
    """
    Convert a Java SimpleDateFormat into a Python strftime format string.

    Only a subset of the SimpleDateForamt mini-language is supported.

    Examples
    --------
    >>> java_fmt = "EEEEE, MMM dd, yyyy HH:mm:ss (65%, '''check''')"
    >>> fmt = java_date_format_to_strftime(java_fmt)
    >>> fmt
    "%A, %B %d, %Y %H:%M:%S (65%%, 'check')"

    >>> import time
    >>> t = time.struct_time((2015, 11, 15, 8, 20, 48, 6, 319, 0))
    >>> time.strftime(fmt, t)
    "Sunday, Nov 15, 2015 08:20:48 (65%, 'check')"

    """
    p = _parse_java_simple_date_format(fmt)
    out = ''
    for item in p:
        if not isinstance(item, tuple):
            out += item.replace('%', '%%')
            continue
        c, count = item
        if c in "DFGKkSWw":
            raise ValueError("format %r not implemented" % (c,))
        if c == 'h':
            # Hour in am/pm, 1-12
            if count == 1:
                raise ValueError("single 'h' not implemented")
            if count > 2:
                padding = "0"*(count - 2)
                out += padding
            out += "%I"
        elif c == 'H':
            # Hour in day, 0-23
            if count == 1:
                raise ValueError("single 'H' not implemented")
            if count > 2:
                padding = "0"*(count - 2)
                out += padding
            out += "%H"
        elif c == 'm':
            # Minute, 0-59
            if count == 1:
                raise ValueError("single 'm' not implemented")
            if count > 2:
                padding = "0"*(count - 2)
                out += padding
            out += "%M"
        elif c == 's':
            # Second, 0-59
            if count == 1:
                raise ValueError("single 's' not implemented")
            if count > 2:
                padding = "0"*(count - 2)
                out += padding
            out += "%S"
        elif c == 'a':
            # AM/PM text
            out += "%p"
        elif c == 'd':
            # Day in month, 1-31
            if count == 1:
                raise ValueError("single 'd' not implemented")
            if count > 2:
                padding = "0"*(count - 2)
                out += padding
            out += "%d"
        elif c == 'M':
            # Month in year, 1-12
            if count == 1:
                raise ValueError("single 'M' not implemented")
            if count == 2:
                out += "%m"
            elif count == 3:
                out += "%b"
            else:
                out += "%B"
        elif c == 'y':
            # Year
            if count != 2 and count != 4:
                raise ValueError(f"'{c*count}' not implemented; use either"
                                 " 'yy' or 'yyyy'.")
            if count == 2:
                out += "%y"
            else:
                out += "%Y"
        elif c == 'E':
            # Day in week (text)
            if count <= 3:
                out += "%a"
            else:
                out += "%A"
        elif c == 'z':
            # Time zone (text)
            if count > 3:
                raise ValueError(f"'{c*count}' not implemented; use 'zzz'.")
            out += "%Z"
        elif c == 'Z':
            # Time zone offset (e.g. "-0800")
            if count > 1:
                raise ValueError(f"'{c*count}' not implemented; use 'Z'.")
            out += "%z"
        else:
            raise RuntimeError(f"Unhandled format code and count: {item}. "
                               "(This should not happen.)")
    return out
