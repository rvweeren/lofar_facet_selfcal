"""Discovery for the historical VLA calibrator-based calibration (--vlacal): band and spectral windows,
field roles from the calibrator codes (FIELD CODE) and positions, the phase calibrator of each target,
the parallactic angle coverage of the calibrators, and the calibration level."""
import json

import astropy.units as units
import numpy as np
from astropy.coordinates import SkyCoord
from casacore.measures import measures
import casacore.quanta as qa
from casacore.tables import table  # type: ignore

from submods.calibrator_models import POLARISED_CALIBRATOR_MODELS, SETJY_CALIBRATORS
from submods.runtime_utils import terminal_print

# bands of the first version (GHz ranges of the historical VLA receivers)
VLA_BANDS = {'L': (1.0, 2.0), 'C': (4.0, 8.0), 'X': (8.0, 12.0)}
# polarisation angle calibrators with a valid model per band (3C138: Perley et al. 2026 model up to 4 GHz)
POLCAL_BANDS = {'3C286': ('L', 'C', 'X'), '3C138': ('L',)}
# unpolarised leakage calibrators (J2000 degrees) and the bands where they are unpolarised to < ~0.1%
UNPOLARISED_CALIBRATORS = {'3C84': ((49.950667, 41.511695), ('L', 'C', 'X')),
                           'OQ208': ((211.751642, 28.454081), ('L', 'C', 'X')),
                           '3C147': ((85.650575, 49.852009), ('L',))}
CIRCULAR_CORR = [5, 6, 7, 8]  # RR, RL, LR, LL


def vla_band(freq):
    """VLA band letter (L, C, X) of a frequency in Hz."""
    for name, (lo, hi) in VLA_BANDS.items():
        if lo * 1e9 <= freq < hi * 1e9:
            return name
    raise ValueError(f'--vlacal supports the L, C and X bands, not {freq / 1e9:.3f} GHz')


def _match(direction, catalogue, tolerance=0.05):
    """Name of the catalogue source (name: (ra, dec) deg) within tolerance (deg) of direction (rad), or None."""
    coord = SkyCoord(direction[0] * units.radian, direction[1] * units.radian, frame='icrs')
    for name, (ra, dec) in catalogue.items():
        if coord.separation(SkyCoord(ra * units.deg, dec * units.deg, frame='icrs')) < tolerance * units.deg:
            return name
    return None


def separation_deg(dir1, dir2):
    """Angular separation (deg) of two directions (RA, Dec in rad)."""
    c1 = SkyCoord(dir1[0] * units.radian, dir1[1] * units.radian, frame='icrs')
    c2 = SkyCoord(dir2[0] * units.radian, dir2[1] * units.radian, frame='icrs')
    return float(c1.separation(c2).deg)


def discover_spws(ms):
    """The spectral windows with data: spw id, DATA_DESC_ID, mean frequency, channels and channel width."""
    with table(ms, ack=False) as t:
        ddids = np.unique(t.getcol('DATA_DESC_ID'))
    with table(ms + '::DATA_DESCRIPTION', ack=False) as t:
        spw_of_ddid = t.getcol('SPECTRAL_WINDOW_ID')
        pol_of_ddid = t.getcol('POLARIZATION_ID')
    with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
        chan_freq = [t.getcell('CHAN_FREQ', i) for i in range(t.nrows())]
        chan_width = [t.getcell('CHAN_WIDTH', i) for i in range(t.nrows())]
        names = t.getcol('NAME')
    with table(ms + '::POLARIZATION', ack=False) as t:
        corr = [list(t.getcell('CORR_TYPE', i)) for i in range(t.nrows())]
    spws = []
    for ddid in ddids:
        spw = int(spw_of_ddid[ddid])
        spws.append(dict(spw=spw, ddid=int(ddid), freq=float(np.mean(chan_freq[spw])), nchan=len(chan_freq[spw]),
                         chanwidth=float(abs(chan_width[spw][0])), name=str(names[spw]),
                         corr=[int(c) for c in corr[pol_of_ddid[ddid]]]))
    return spws


def discover_fields(ms, spws):
    """
    Fields, calibrator codes, scans and calibrator matches of a (multi-field) VLA Measurement Set, for the
    spectral windows spws (from discover_spws; e.g. those of one band of a band-switching observation).

    Returns
    -------
    dict
        Per field name with data in spws: field_id, direction (rad), code (FIELD CODE, the VLA calibrator
        code; blank for targets), scans, scans_spw (the scans per spectral window: in a band-switching
        observation a scan observes only some of them), hours on source, and the matched flux,
        polarisation angle and unpolarised calibrator names (or None).
    """
    with table(ms + '::FIELD', ack=False) as t:
        names = list(t.getcol('NAME'))
        dirs = t.getcol('DELAY_DIR')[:, 0, :]
        codes = list(t.getcol('CODE')) if 'CODE' in t.colnames() else [None] * len(names)
    with table(ms, ack=False) as t:
        field_id = t.getcol('FIELD_ID')
        scan = t.getcol('SCAN_NUMBER')
        time = t.getcol('TIME')
        interval = t.getcol('INTERVAL')
        ddid = t.getcol('DATA_DESC_ID')
    fields = {}
    for fid, name in enumerate(names):
        scans_spw = {str(s['spw']): [int(x) for x in np.unique(scan[(field_id == fid) & (ddid == s['ddid'])])]
                     for s in spws}
        scans_spw = {s: v for s, v in scans_spw.items() if v}
        sel = (field_id == fid) & np.isin(ddid, [s['ddid'] for s in spws])
        if not np.any(sel):
            continue
        utime, index = np.unique(time[sel], return_index=True)
        fields[name] = dict(field_id=fid, direction=[float(d) for d in dirs[fid]],
                            code=None if codes[fid] is None else str(codes[fid]),
                            scans=[int(s) for s in np.unique(scan[sel])], scans_spw=scans_spw,
                            hours=float(np.sum(interval[sel][index]) / 3600.),
                            flux_calibrator=_match(dirs[fid], SETJY_CALIBRATORS),
                            polarisation_calibrator=_match(dirs[fid], {k: SETJY_CALIBRATORS[k]
                                                                       for k in POLARISED_CALIBRATOR_MODELS}),
                            unpolarised_calibrator=_match(dirs[fid], {k: v[0] for k, v in
                                                                      UNPOLARISED_CALIBRATORS.items()}))
    return fields


def assign_roles(fields, band, args, all_names=()):
    """
    Flux, phase, polarisation angle and leakage calibrators and targets, from the options or from the
    calibrator positions and the VLA calibrator codes (FIELD CODE: non-blank for calibrators).

    Returns
    -------
    dict
        Lists of field names under 'fluxcal', 'phasecal', 'polcal', 'leakcal' (explicit or unpolarised
        leakage calibrators; the phase calibrators are added in choose_level) and 'targets'.
    """
    have_codes = all(f['code'] is not None for f in fields.values())

    def is_cal(name):
        return have_codes and fields[name]['code'].strip() != ''

    roles = {}
    roles['fluxcal'] = args['vlacal_fluxcal'] or [n for n in fields if fields[n]['flux_calibrator'] is not None]
    roles['polcal'] = args['vlacal_polcal'] or [n for n in fields if band in
                                                POLCAL_BANDS.get(fields[n]['polarisation_calibrator'], ())]
    roles['phasecal'] = args['vlacal_phasecal'] or [n for n in fields if is_cal(n) and
                                                    n not in roles['fluxcal'] + roles['polcal']]
    roles['leakcal'] = args['vlacal_leakcal'] or [n for n in fields if band in
                                                  UNPOLARISED_CALIBRATORS.get(fields[n]['unpolarised_calibrator'],
                                                                              (None, ()))[1]]
    calibrators = set(roles['fluxcal'] + roles['polcal'] + roles['phasecal'] + roles['leakcal'])
    roles['targets'] = args['vlacal_targets'] or [n for n in fields if have_codes and not is_cal(n)
                                                  and n not in calibrators]
    for role, names in roles.items():
        for name in list(names):
            if name in fields:
                continue
            if name in all_names:  # in the MS, but not observed in this band
                terminal_print(f'WARNING: {role} field {name} has no data in the {band} band, it is left out')
                names.remove(name)
            else:
                raise ValueError(f'{role} field {name} is not in the MS, fields: {list(all_names) or list(fields)}')
    unmodelled = [n for n in roles['fluxcal'] if fields[n]['flux_calibrator'] is None]
    if unmodelled:
        raise ValueError(f'No flux density model for {unmodelled}, available: {list(SETJY_CALIBRATORS)}')
    if not roles['fluxcal']:
        raise ValueError('No flux density calibrator (3C48, 3C138, 3C147 or 3C286) found, set --vlacal-fluxcal')
    for name in roles['polcal']:
        if fields[name]['polarisation_calibrator'] is None:
            raise ValueError(f'No polarisation model for --vlacal-polcal field {name}, available: '
                             f'{list(POLARISED_CALIBRATOR_MODELS)}')
    if not have_codes and not (args['vlacal_targets'] and (args['vlacal_phasecal'] is not None)):
        raise ValueError('The FIELD table has no CODE column, set --vlacal-phasecal and --vlacal-targets')
    if not roles['targets']:
        terminal_print('WARNING: no target fields found (fields with a blank calibrator code), set --vlacal-targets')
    return roles


def pair_targets(fields, roles):
    """The phase calibrator nearest on the sky of each target (None without phase calibrators)."""
    pairing = {}
    for target in roles['targets']:
        if not roles['phasecal']:
            pairing[target] = None
            continue
        seps = {p: separation_deg(fields[target]['direction'], fields[p]['direction']) for p in roles['phasecal']}
        pairing[target] = min(seps, key=seps.get)
    return pairing


def scan_geometry(ms, fields, name):
    """
    Elevation and parallactic angle (deg, at the array centre) at the middle of each scan of a field.

    Returns
    -------
    list of dict
        Per scan: scan, time, elevation, parang.
    """
    fid = fields[name]['field_id']
    with table(ms, ack=False) as t:
        scan = t.getcol('SCAN_NUMBER')
        sel = (t.getcol('FIELD_ID') == fid) & np.isin(scan, fields[name]['scans'])  # the scans of this band
        scan = scan[sel]
        time = t.getcol('TIME')[sel]
    with table(ms + '::ANTENNA', ack=False) as t:
        pos = np.mean(t.getcol('POSITION'), axis=0)
    ra, dec = fields[name]['direction']
    dm = measures()
    direction = dm.direction('J2000', qa.quantity(ra, 'rad'), qa.quantity(dec, 'rad'))
    zenith = dm.direction('AZEL', qa.quantity(0., 'deg'), qa.quantity(90., 'deg'))
    dm.do_frame(dm.position('ITRF', *(qa.quantity(p, 'm') for p in pos)))
    out = []
    for s in np.unique(scan):
        tmid = float(np.median(time[scan == s]))
        dm.do_frame(dm.epoch('UTC', qa.quantity(tmid, 's')))
        elevation = np.degrees(dm.measure(direction, 'AZEL')['m1']['value'])
        chi = dm.posangle(dm.measure(direction, 'APP'), zenith).get_value('deg')
        out.append(dict(scan=int(s), time=tmid, elevation=float(elevation), parang=float(chi)))
    return out


def parang_range(geometry):
    """Range (deg) of the parallactic angle over the scans (unwrapped)."""
    if not geometry:
        return 0.
    chi = np.degrees(np.unwrap(np.radians([g['parang'] for g in geometry])))
    return float(chi.max() - chi.min())


def choose_level(roles, fields, geometry, spws, args):
    """
    Calibration level and the leakage calibrators.

    Leakage is solvable on an unpolarised calibrator (source polarisation fixed at 0) or on a calibrator
    observed over at least --vlacal-min-parang degrees of parallactic angle (its polarisation is solved
    with the leakage: it rotates with the parallactic angle, the leakage does not). The phase
    calibrators qualify by default; explicit --vlacal-leakcal fields are used as given.

    Returns
    -------
    (str, dict)
        Level ('none', 'leakage' or 'full') and per leakage calibrator whether its polarisation is solved.
    """
    fourpol = all(s['corr'] == CIRCULAR_CORR for s in spws)
    leak = {}
    for name in roles['leakcal']:
        leak[name] = dict(fitpol=fields[name]['unpolarised_calibrator'] is None,
                          parang_range=parang_range(geometry.get(name, [])))
    if not args['vlacal_leakcal']:
        for name in roles['phasecal']:
            prange = parang_range(geometry.get(name, []))
            if name not in leak and prange >= args['vlacal_min_parang']:
                leak[name] = dict(fitpol=True, parang_range=prange)
    for name, item in leak.items():
        if item['fitpol'] and item['parang_range'] < args['vlacal_min_parang']:
            terminal_print(f'WARNING: leakage calibrator {name} covers only {item["parang_range"]:.0f} deg of parallactic '
                           f'angle (< {args["vlacal_min_parang"]:.0f}): its polarisation and the leakage are poorly '
                           'separated')
    level = args['vlacal_level']
    possible = 'none'
    if fourpol and leak:
        possible = 'full' if roles['polcal'] else 'leakage'
    if level == 'auto':
        level = possible
    order = ['none', 'leakage', 'full']
    if order.index(level) > order.index(possible):
        reason = ('the data do not have all four circular correlations' if not fourpol else
                  'no leakage calibrator (an unpolarised calibrator, or a calibrator with at least '
                  f'{args["vlacal_min_parang"]:.0f} deg parallactic angle coverage)' if not leak else
                  'no polarisation angle calibrator (3C286, or 3C138 at L-band)')
        raise ValueError(f'--vlacal-level {level} is not possible: {reason}')
    if level == 'none':
        leak = {}
    return level, leak


def discover(ms, args, outfile='misc/vlacal_discovery.json'):
    """
    Run the discovery and write it to a json file.

    Returns
    -------
    dict
        ms, band, spws, antennas, transition (EVLA antennas present), fields, roles, pairing, geometry
        (calibrator scans), level and leakage calibrators.
    """
    with table(ms + '::OBSERVATION', ack=False) as t:
        telescope = t.getcol('TELESCOPE_NAME')[0]
    with table(ms + '::ANTENNA', ack=False) as t:
        antennas = list(t.getcol('NAME'))
    if telescope.upper() not in ('VLA', 'EVLA'):
        raise ValueError(f'--vlacal is for VLA data, the MS is from {telescope}')
    vla_ants = [a for a in antennas if a.upper().startswith('VA')]
    evla_ants = [a for a in antennas if a.upper().startswith('EA')]
    if not vla_ants:
        raise ValueError('--vlacal is for historical VLA data (VA antennas, CASA importvla); this MS has only EVLA '
                         'antennas: use the standard facetselfcal VLA workflow')
    allspws = discover_spws(ms)
    bands = sorted({vla_band(s['freq']) for s in allspws})
    band = args['vlacal_band']
    if band is None:
        if len(bands) > 1:
            raise ValueError(f'The MS has data in the {", ".join(bands)} bands (band switching): choose one with '
                             '--vlacal-band and run each band in its own directory')
        band = bands[0]
    elif band not in bands:
        raise ValueError(f'--vlacal-band {band}: the MS has data in the {", ".join(bands)} band(s)')
    spws = [s for s in allspws if vla_band(s['freq']) == band]
    with table(ms + '::FIELD', ack=False) as t:
        all_names = list(t.getcol('NAME'))
    fields = discover_fields(ms, spws)
    roles = assign_roles(fields, band, args, all_names=all_names)
    pairing = pair_targets(fields, roles)
    calibrators = sorted(set(roles['fluxcal'] + roles['phasecal'] + roles['polcal'] + roles['leakcal']))
    geometry = {name: scan_geometry(ms, fields, name) for name in calibrators}
    level, leakcal = choose_level(roles, fields, geometry, spws, args)
    info = dict(ms=ms, band=band, spws=spws, antennas=antennas, transition=bool(evla_ants), evla_antennas=evla_ants,
                fields=fields, roles=roles, pairing=pairing, geometry=geometry, level=level, leakcal=leakcal,
                level_requested=args['vlacal_level'], bands=bands)

    terminal_print(f'VLA {band} band, {len(spws)} spectral window(s): '
                   + ', '.join(f"spw{s['spw']} {s['freq'] / 1e6:.1f} MHz x {s['nchan']} ch" for s in spws)
                   + f'; calibration level: {level}')
    if evla_ants:
        terminal_print(f'EVLA-transition data: {len(evla_ants)} EVLA antennas ({", ".join(evla_ants)})'
                       + (', the VLA-EVLA baselines are flagged' if args['vlacal_flag_mixed_baselines'] else ''))
    for name, f in fields.items():
        role = [r for r, names in roles.items() if name in names]
        extra = f', phase calibrator {pairing[name]}' if name in pairing else ''
        terminal_print(f"  {name:14s} code '{f['code']}' {','.join(role) or '-':16s} {f['hours']:.2f} h, "
                       f"scans {f['scans']}{extra}")
    for name in calibrators:
        terminal_print(f'  {name}: parallactic angle range {parang_range(geometry[name]):.0f} deg, elevation '
                       f"{min(g['elevation'] for g in geometry[name]):.0f}-"
                       f"{max(g['elevation'] for g in geometry[name]):.0f} deg")
    for name, item in leakcal.items():
        terminal_print(f"  leakage calibrator {name}: {'polarisation solved' if item['fitpol'] else 'unpolarised'}")
    with open(outfile, 'w') as fp:
        json.dump(info, fp, indent=1)
    return info
