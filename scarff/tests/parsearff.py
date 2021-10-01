# Copyright © 2021 Warren Weckesser

import re


def comma_split(s, maxsplit=0, stripquotes=True):
    comma_split_re = r'(?!\B"[^"]*),(?![^"]*"\B)'
    parts = re.split(comma_split_re, s, maxsplit=maxsplit)
    if stripquotes:
        parts = [(part.strip('"')
                  if part.startswith('"') and part.endswith('"')
                  else part) for part in parts]
    return parts


def space_split(s, maxsplit=0, stripquotes=True):
    space_split_re = r'(?!\B"[^"]*) +(?![^"]*"\B)'
    parts = re.split(space_split_re, s, maxsplit=maxsplit)
    if stripquotes:
        parts = [(part.strip('"')
                  if part.startswith('"') and part.endswith('"')
                  else part) for part in parts]
    return parts


def parsearff(s):
    """
    Parse a string containing an ARFF file.

    This is *not* a full-blown, robust parser, but it is sufficient for
    testing the code that writes numpy arrays to an ARFF file.
    """
    lines = s.splitlines()

    # Read the header...
    relation = None
    attributes = []
    types = []
    for linenumber, line in enumerate(lines):
        if line.startswith('%'):
            continue
        if line.strip() == '':
            continue
        uline = line.upper()
        if uline.startswith('@DATA'):
            break
        if uline.startswith('@RELATION'):
            if relation is not None:
                raise ValueError("More than one @RELATION found")
            relation = space_split(line)[1]
            continue
        if uline.startswith('@ATTRIBUTE'):
            parts = space_split(line)
            print(f"parts = {parts}")
            if len(parts) < 3:
                raise ValueError(f"Line {linenumber}: incomplete attribute")
            attr, typ = parts[1:3]
            if typ == 'date':
                maxlen = 4
            else:
                maxlen = 3
            if len(parts) > maxlen:
                raise ValueError(f"Line {linenumber}: invalid attribute line")
            if attr in attributes:
                raise ValueError(f"The attribute {attr} appears more "
                                 "than once")
            attributes.append(attr)
            typ = typ.strip()
            if typ.startswith('{') and typ.endswith('}'):
                typ = ('nominal', comma_split(typ[1:-1]))
            elif typ.upper() == 'DATE' and len(parts) == 4:
                typ = ('date', parts[3])
            types.append(typ)
    else:
        raise ValueError("Missing @DATA section")

    if relation is None:
        raise ValueError("Missing @RELATION")
    if len(attributes) == 0:
        raise ValueError("No attributes found")

    # Read the @DATA section...
    data = []
    for linenumber, line in enumerate(lines[linenumber+1:]):
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            # sparse format
            values = ['0']*len(attributes)
            line = line[1:-1].strip()
            if line != '':
                pairs = [space_split(q.strip()) for q in comma_split(line)]
                for pair in pairs:
                    fieldnum = int(pair[0])
                    values[fieldnum] = pair[1]
        else:
            values = comma_split(line)
            if len(values) != len(attributes):
                raise ValueError("Line %i: Incorrect number of fields; "
                                 "expected %i, got %i" % (linenumber+1,
                                                          len(attributes),
                                                          len(values)))
        data.append(values)

    return relation, attributes, types, data
