# Copyright © 2021 Warren Weckesser

import re
from dataclasses import dataclass


@dataclass
class Relational:
    """
    Data associated with a relational attribute.

    The class holds the name, attribute names and attribute
    types of a relation attribute.
    """
    name: str
    attributes: list
    types: list


def _comma_split(s, maxsplit=0, stripquotes=True):
    comma_split_re = r'(?!\B"[^"]*),(?![^"]*"\B)'
    parts = re.split(comma_split_re, s, maxsplit=maxsplit)
    if stripquotes:
        parts = [(part.strip('"')
                  if part.startswith('"') and part.endswith('"')
                  else part) for part in parts]
    return parts


def _space_split(s, maxsplit=0, stripquotes=True):
    space_split_re = r'(?!\B"[^"]*) +(?![^"]*"\B)'
    parts = re.split(space_split_re, s, maxsplit=maxsplit)
    if stripquotes:
        parts = [(part.strip('"')
                  if part.startswith('"') and part.endswith('"')
                  else part) for part in parts]
    return parts


def _qfind(s, char):
    """
    Find char in s, but ignore occurrences of char inside double-quotes.
    """
    if char == '"':
        raise ValueError("can't search for a quote character with qfind")
    in_double_quotes = False
    prev_c = ''
    for k, c in enumerate(s):
        if c == '"' and (not in_double_quotes
                         or in_double_quotes and prev_c != '\\'):
            in_double_quotes = not in_double_quotes
        elif not in_double_quotes and c == char:
            return k
        prev_c = c
    return -1


def _parse_attribute(line, attributes, relational_name, linenumber):
    parts = _space_split(line, maxsplit=2)
    if len(parts) < 3:
        raise ValueError(f"line {linenumber}: incomplete attribute")

    name, type_spec = parts[1:3]
    if name in attributes:
        more = ('' if relational_name is None
                else f" in the relational attribute '{relational_name}'")
        raise ValueError(f"line {linenumber}: the attribute '{name}' "
                         "appears more than once" + more)
    type_spec = type_spec.strip()
    utype_spec = type_spec.upper()

    if utype_spec.startswith('DATE'):
        parts = _space_split(type_spec)
        if len(parts) > 2:
            raise ValueError(f"line {linenumber}: invalid DATE type")
        if len(parts) == 1:
            fmt = None
        else:
            fmt = parts[1].strip()
        typ = ('date', fmt)
    elif type_spec.startswith('{'):
        if not type_spec.endswith('}'):
            raise ValueError(f'line {linenumber}: nominal type missing '
                             'closing }')
        typ = ('nominal', [w.strip() for w in _comma_split(type_spec[1:-1])])
    elif utype_spec in ['NUMERIC', 'INTEGER', 'REAL', 'STRING', 'RELATIONAL']:
        typ = type_spec
    else:
        raise ValueError(f"line {linenumber}: invalid date type for "
                         f"attribute '{name}'")

    return name, typ


def _parse_data(line, attributes, linenumber):
    "Parse one line of non-sparse data."
    values = _comma_split(line)
    weight = None
    if len(values) > 0:
        last = values[-1].strip()
        if last.startswith('{') and last.endswith('}'):
            # Last value is an instance weight.
            weight = last[1:-1]
            values = values[:-1]
    if len(values) != len(attributes):
        raise ValueError("line %i: Incorrect number of fields; "
                         "expected %i, got %i" % (linenumber+1,
                                                  len(attributes),
                                                  len(values)))
    return values, weight


def _parse_sparse_data(line, attributes, linenumber):
    "Parse one line of sparse data."
    values = ['0']*len(attributes)
    close_brace_index = _qfind(line, '}')
    if close_brace_index == -1:
        raise ValueError(f'line {linenumber}: unmatch opening {{')
    trailing = line[close_brace_index+1:].strip()
    line = line[1:close_brace_index].strip()
    if line != '':
        pairs = [_space_split(q.strip()) for q in _comma_split(line)]
        for pair in pairs:
            fieldnum = int(pair[0])
            values[fieldnum] = pair[1]
    weight = None
    if trailing != '':
        if trailing[0] == ',':
            trailing = trailing[1:].strip()
            if trailing.startswith('{') and trailing.endswith('}'):
                weight = trailing[1:-1]
        if weight is None:
            raise ValueError(f'line {linenumber}: unexpected text '
                             'after sparse data (incomplete instance weight?)')
    return values, weight


def parsearff(s):
    """
    Parse a string containing an ARFF file.

    This is not a complete ARFF parser.  It returns only collections
    of strings, and sparse data is "densified".  It also might not handle
    quoting and escaping of quote characters in strings and names correctly.
    It is sufficient for testing that savearff writes the expected text to
    the ARFF file.
    """
    lines = s.splitlines()

    # - - - - - - - - - - - - - - -
    # Read the header
    # - - - - - - - - - - - - - - -

    # Get the @RELATION
    relation = None
    for linenumber, line in enumerate(lines, start=1):
        line = line.strip()
        if line.startswith('%') or line == '':
            continue
        uline = line.upper()
        if uline.startswith('@RELATION'):
            parts = _space_split(line)
            if len(parts) == 1:
                raise ValueError(f'line {linenumber}: @RELATION missing name')
            if len(parts) > 2:
                raise ValueError(f'line {linenumber}: invalid @RELATION line')
            relation = parts[1]
            break
        raise ValueError(f"line {linenumber}: expected to find the @RELATION "
                         f"but got '{line}'")

    # Get the @ATTRIBUTEs
    attributes = []  # List of attribute names.
    types = []
    relational = None
    for linenumber, line in enumerate(lines[linenumber:], start=linenumber+1):
        line = line.strip()
        if line.startswith('%') or line == '':
            continue
        uline = line.upper()
        if uline.startswith('@DATA'):
            if relational is not None:
                raise ValueError(f"line {linenumber}: @DATA encountered "
                                 f"before the @END for relational "
                                 f"attribute '{relational.name}'")
            break
        if uline.startswith('@ATTRIBUTE'):
            if relational is not None:
                # Currently processing the attributes of a relational
                # attribute.
                attr, typ = _parse_attribute(line, relational.attributes,
                                             relational.name,
                                             linenumber)
                if typ == 'relational':
                    raise ValueError(f"line {linenumber}: nested relational "
                                     f"attributes are not supported; can't "
                                     f"create relational attribute '{attr}' "
                                     f"within relational attribute "
                                     f"'{relational.name}'")
                relational.attributes.append(attr)
                relational.types.append(typ)
            else:
                attr, typ = _parse_attribute(line, attributes, None,
                                             linenumber)
                if typ == 'relational':
                    # Create the Relational instance to hold the relational
                    # attributes and types as they are parsed.
                    relational = Relational(name=attr, attributes=[], types=[])
                else:
                    attributes.append(attr)
                    types.append(typ)
        elif uline.startswith('@END'):
            if relational is None:
                raise ValueError(f'line {linenumber}: unexpected @END')
            parts = _space_split(line)
            if len(parts) == 1:
                raise ValueError(f'line {linenumber}: @END missing name')
            if len(parts) > 2:
                raise ValueError(f'line {linenumber}: invalid @END line')
            end_name = parts[1]
            if end_name != relational.name:
                raise ValueError(f"line: {linenumber}: @END name '{end_name}' "
                                 f"does not match the @RELATIONAL name "
                                 f"'{relational.name}'")
            attributes.append(relational.name)
            types.append(('relational', (relational.attributes,
                                         relational.types)))
            relational = None
        else:
            raise ValueError(f'line {linenumber}: unexpected input.  Expected '
                             '@ATTRIBUTE or @DATA')
    else:
        raise ValueError("Missing @DATA section")

    if relation is None:
        raise ValueError("Missing @RELATION")
    if len(attributes) == 0:
        raise ValueError("No attributes found")

    # - - - - - - - - - - - - - - -
    # Read the @DATA section
    # - - - - - - - - - - - - - - -

    data = []
    weights = []
    for linenumber, line in enumerate(lines[linenumber:], start=linenumber+1):
        line = line.strip()
        if line.startswith('{'):
            # Sparse format
            values, weight = _parse_sparse_data(line, attributes, linenumber)
        else:
            # Regular (i.e. non-sparse) data
            values, weight = _parse_data(line, attributes, linenumber)
        data.append(values)
        weights.append(weight)

    return relation, attributes, types, data, weights
