"""
DP3-style arguments: an optional parset file as the first argument, then key=value overrides
(DP3 base/DP3.cc, common/ParameterSetImpl.cc), and the per-station value syntax of
antenna_averaging_factors (DP3 common/ValuePerStationParsing.h).
"""
import fnmatch


def strip_quotes(v):
    v = v.strip()
    while len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        v = v[1:-1].strip()
    return v


def split_top(s, sep=","):
    """Split at `sep` outside brackets."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [c.strip() for c in out]


def parse_list(v):
    """'[a,b,[c,d]]' -> ['a', 'b', '[c,d]']; a bare value -> [value]."""
    v = strip_quotes(v)
    if v.startswith("[") and v.endswith("]"):
        v = v[1:-1]
    return [strip_quotes(x) for x in split_top(v)] if v.strip() else []


def read_args(argv):
    """
    DP3 command-line arguments -> dict of key -> value (strings, quotes removed).

    The first argument is a parset file if it has no '='; later key=value arguments override it.
    """
    kv = {}
    args = list(argv)
    if args and "=" not in args[0]:
        with open(args.pop(0)) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    kv[k.strip()] = strip_quotes(v)
    for a in args:
        if "=" not in a:
            raise ValueError(f"argument without '=': {a!r}")
        k, v = a.split("=", 1)
        kv[k.strip()] = strip_quotes(v)
    return kv


def solve_step_prefix(kv):
    """The prefix ('<step>.') of the single ddecal step in steps=[...]."""
    steps = parse_list(kv.get("steps", "[]"))
    if len(steps) != 1:
        raise ValueError(f"exactly one step (the ddecal step) is supported, got steps={steps}")
    name = steps[0]
    typ = kv.get(name + ".type", name.rstrip("0123456789")).lower()
    if typ not in ("ddecal", "ddecalibrate"):
        raise ValueError(f"step {name} has type {typ}; only ddecal is supported")
    return name + "."


def antenna_groups(value):
    """
    antennaconstraint '[[a,b],[c,d]]' -> list of lists of antenna names (wildcards allowed).
    """
    return [[strip_quotes(n) for n in parse_list(g)] for g in parse_list(value)]


def _match(pattern, names):
    neg = pattern[:1] in ("!", "^")
    p = pattern[1:] if neg else pattern
    hit = {n for n in names if fnmatch.fnmatchcase(n, p)}
    return (set(names) - hit) if neg else hit


def value_per_station(value, names, default, conv=int):
    """
    Per-station values from DP3's syntax: ['pattern:value', '[a,b]:value', ...], patterns with
    wildcards and !/^ negation; later items override earlier ones.

    Returns
    -------
    list
        One value per name.
    """
    out = {n: default for n in names}
    for item in parse_list(value):
        pat, _, val = item.rpartition(":")
        if not pat:
            raise ValueError(f"no ':value' in {item!r}")
        pats = parse_list(pat) if pat.strip().startswith("[") else [pat]
        hit = set()
        for p in pats:
            hit |= _match(p, names)
        for n in hit:
            out[n] = conv(val)
    return [out[n] for n in names]
