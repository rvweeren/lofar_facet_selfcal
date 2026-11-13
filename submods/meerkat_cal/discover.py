"""Discovery for the MeerKAT calibrator-based calibration: band, field roles from the scan intents and the
calibrator positions, calibration level, and checks of the polarisation calibrator scans."""
import json
import os

import astropy.units as units
import numpy as np
from astropy.coordinates import SkyCoord
from casacore.measures import measures
import casacore.quanta as qa
from casacore.tables import table  # type: ignore

from submods.calibrator_models import POLARISED_CALIBRATOR_MODELS, SETJY_CALIBRATORS
from submods.runtime_utils import terminal_print

# bandpass/flux calibrators with a facetselfcal sky model (J2000 degrees); J1939-6342 is unpolarised,
# J0408-6545 is weakly polarised
BANDPASS_CALIBRATORS = {'J1939-6342': (294.854275, -63.712674),
                        'J0408-6545': (62.084875, -65.752517)}
UNPOLARISED_CALIBRATORS = ['J1939-6342']


def meerkat_band(freqs):
    """
    MeerKAT band of a set of channel frequencies.

    The S-band sub-bands follow the centre frequencies used elsewhere in facetselfcal
    (S0 < 2.3 GHz, S1 < 2.52 GHz, S2 < 2.73 GHz, S3 < 2.95 GHz, else S4).

    Parameters
    ----------
    freqs : array_like
        Channel frequencies in Hz.

    Returns
    -------
    str
        'UHF', 'L', 'S0', 'S1', 'S2', 'S3' or 'S4'.
    """
    midfreq = np.mean(freqs)
    if midfreq < 1.0e9:
        return 'UHF'
    if midfreq < 1.7e9:
        return 'L'
    for name, upper in [('S0', 2.3e9), ('S1', 2.52e9), ('S2', 2.73e9), ('S3', 2.95e9)]:
        if midfreq < upper:
            return name
    return 'S4'


def _match_position(direction, catalogue, tolerance=0.05):
    """Name of the catalogue source within tolerance (deg) of direction (rad), or None."""
    coord = SkyCoord(direction[0]*units.radian, direction[1]*units.radian, frame='icrs')
    for name, (ra, dec) in catalogue.items():
        if coord.separation(SkyCoord(ra*units.deg, dec*units.deg, frame='icrs')) < tolerance*units.deg:
            return name
    return None


def discover_fields(ms):
    """
    Fields, scans and scan intents of a (multi-field) MeerKAT Measurement Set.

    Parameters
    ----------
    ms : str
        Path to the Measurement Set.

    Returns
    -------
    dict
        Per field name: field_id, direction (rad), scans, locations (the MS, field_id and scans, see
        field_locations), hours on source, intents, and the matched bandpass or polarisation calibrator
        name (or None).
    """
    with table(ms + '::FIELD', ack=False) as t:
        names = list(t.getcol('NAME'))
        dirs = t.getcol('DELAY_DIR')[:, 0, :]
    with table(ms + '::STATE', ack=False) as t:
        obs_mode = list(t.getcol('OBS_MODE')) if t.nrows() > 0 else []
    with table(ms, ack=False) as t:
        field_id = t.getcol('FIELD_ID')
        state_id = t.getcol('STATE_ID')
        scan = t.getcol('SCAN_NUMBER')
        time = t.getcol('TIME')
        interval = t.getcol('INTERVAL')
    fields = {}
    for fid, name in enumerate(names):
        sel = field_id == fid
        if not np.any(sel):
            continue
        utime, index = np.unique(time[sel], return_index=True)
        intents = set()
        for sid in np.unique(state_id[sel]):
            if 0 <= sid < len(obs_mode):
                intents.update(x.strip() for x in obs_mode[sid].split(','))
        scans = [int(s) for s in np.unique(scan[sel])]
        fields[name] = dict(field_id=fid, direction=list(dirs[fid]), scans=scans,
                            locations=[dict(ms=ms, field_id=fid, scans=scans)],
                            hours=float(np.sum(interval[sel][index]) / 3600.),
                            intents=sorted(intents),
                            bandpass_calibrator=_match_position(dirs[fid], BANDPASS_CALIBRATORS),
                            polarisation_calibrator=_match_position(dirs[fid], {k: SETJY_CALIBRATORS[k] for k in POLARISED_CALIBRATOR_MODELS}))
    return fields


def field_locations(field, archive_ms=None):
    """
    Where the data of a field are: a list of (ms, field_id, scans), one per input MS that contains the
    field (already split input), or the archive MS (discovery files written before split input was supported
    have no 'locations').
    """
    if 'locations' in field:
        return [(loc['ms'], loc['field_id'], loc['scans']) for loc in field['locations']]
    return [(archive_ms, field['field_id'], field['scans'])]


def scan_location(info, name, scan):
    """The input MS and FIELD_ID of a scan of a field (see field_locations)."""
    for ms, fid, scans in field_locations(info['fields'][name], info.get('ms')):
        if scan in scans:
            return ms, fid
    raise KeyError(f'scan {scan} of {name} is not in the input MSs')


def input_mslist(info):
    """All input MSs: the archive MS, or the already split MSs."""
    return info.get('mslist') or [info['ms']]


def split_input(args):
    """Whether the input is already split MSs (one of the --meerkatcal-*-ms role options is given)."""
    return any(args.get(f'meerkatcal_{role}_ms') for role in ('bpcal', 'phasecal', 'polcal'))


def merge_fields(per_ms):
    """
    Merge the discover_fields results of several MSs by field name (a field can be spread over several
    MSs, e.g. one MS per scan). A scan number may occur in only one MS per field.

    Parameters
    ----------
    per_ms : list of dict
        discover_fields results.

    Returns
    -------
    dict
        Fields as discover_fields, with the locations of all MSs.
    """
    fields = {}
    for found in per_ms:
        for name, f in found.items():
            if name not in fields:
                fields[name] = f
                continue
            m = fields[name]
            dup = sorted(set(m['scans']) & set(f['scans']))
            if dup:
                raise ValueError(f"Field {name}: scans {dup} are in more than one input MS "
                                 f"({m['locations'][0]['ms']}, {f['locations'][0]['ms']})")
            if np.max(np.abs(np.subtract(m['direction'], f['direction']))) > 1e-6:
                raise ValueError(f'Field {name} has different directions in the input MSs')
            m['scans'] = sorted(m['scans'] + f['scans'])
            m['hours'] += f['hours']
            m['intents'] = sorted(set(m['intents']) | set(f['intents']))
            m['locations'] += f['locations']
    return fields


def check_split_inputs(mslist, weightgain=False, maxspan=24.):
    """
    Check that already split MSs are raw data of one MeerKAT observation: the same channel frequencies and
    antennas, four correlations, feeds not swapped yet (RECEPTOR_ANGLE not zero, as in the archive MS) and
    all data within maxspan hours. Raises ValueError otherwise. Different OBSERVATION time ranges give a
    warning. Without WEIGHT_SPECTRUM there is a warning (DP3 uses WEIGHT), or an
    error for --meerkatcal-weightgain (which needs the per-channel archive weights).
    """
    from submods.meerkat_cal.prepare import has_receptor_angle
    if len(set(mslist)) != len(mslist):
        raise ValueError(f'An MS is given more than once: {mslist}')
    ref, ranges = None, []
    for ms in mslist:
        with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
            if t.nrows() != 1:
                raise ValueError(f'{ms} has {t.nrows()} spectral windows, one is supported')
            freqs = t.getcol('CHAN_FREQ')[0]
        with table(ms + '::ANTENNA', ack=False) as t:
            ants = list(t.getcol('NAME'))
        with table(ms + '::POLARIZATION', ack=False) as t:
            ncorr = int(t.getcol('NUM_CORR')[0])
        if ncorr != 4:
            raise ValueError(f'{ms} has {ncorr} correlations, all four (XX, XY, YX, YY) are needed')
        if not has_receptor_angle(ms):
            raise ValueError(f'{ms} has RECEPTOR_ANGLE 0: the feeds were swapped already (not raw archive data)')
        if ref is None:
            ref = (ms, freqs, ants)
        elif len(freqs) != len(ref[1]) or np.max(np.abs(freqs - ref[1])) > 1.:
            raise ValueError(f'{ms} and {ref[0]} have different channel frequencies (average or split them alike)')
        elif ants != ref[2]:
            raise ValueError(f'{ms} and {ref[0]} have different antennas')
        with table(ms, ack=False) as t:
            if 'WEIGHT_SPECTRUM' not in t.colnames():
                if weightgain:
                    raise ValueError(f'--meerkatcal-weightgain needs WEIGHT_SPECTRUM, {ms} has none')
                terminal_print(f'WARNING: {ms} has no WEIGHT_SPECTRUM, the WEIGHT column is used')
            time = t.getcol('TIME')
        ranges.append((float(time.min()), float(time.max()), ms))
    span = max(r[1] for r in ranges) - min(r[0] for r in ranges)
    if span > maxspan * 3600.:
        raise ValueError(f'The input MSs span {span / 3600.:.1f} h: they must be from one observation')
    # the OBSERVATION table (time range of the whole observation) is copied unchanged by the split tools;
    # a difference is only a warning, a tool might rewrite it
    obsranges = set()
    for ms in mslist:
        with table(ms + '::OBSERVATION', ack=False) as t:
            obsranges.add(tuple(np.round(t.getcol('TIME_RANGE')[0] / 60.)) if t.nrows() > 0 else None)
    if len(obsranges) > 1:
        terminal_print('WARNING: the input MSs have different OBSERVATION time ranges, check that they are '
                       'from one observation')


def split_roles(mslists):
    """
    Fields and roles of already split MSs: every field in an MS given with a role option gets that role
    (mslists: role -> list of MSs, roles bpcal, phasecal, polcal and targets).

    Returns
    -------
    (dict, dict)
        Fields (merge_fields) and roles (lists of field names per role).
    """
    roles, per_ms, owner = {}, [], {}
    for role, mss in mslists.items():
        roles[role] = []
        for ms in mss:
            found = discover_fields(ms)
            if not found:
                raise ValueError(f'{ms} contains no data')
            per_ms.append(found)
            for name in found:
                if owner.setdefault(name, role) != role:
                    raise ValueError(f'Field {name} is in a {owner[name]} MS and in a {role} MS ({ms})')
                if name not in roles[role]:
                    roles[role].append(name)
    fields = merge_fields(per_ms)
    check_roles(roles, fields)
    unmatched = [n for n in roles['polcal'] if fields[n]['polarisation_calibrator'] is None]
    if unmatched:
        raise ValueError(f'No polarisation model for {unmatched} (3C286 and 3C138 are supported, matched by position)')
    return fields, roles


def assign_roles(fields, args):
    """
    Bandpass, phase and polarisation calibrators and targets, from the options or the scan intents
    and calibrator positions.

    Parameters
    ----------
    fields : dict
        Output of discover_fields.
    args : dict
        facetselfcal options (meerkatcal_bpcal, meerkatcal_phasecal, meerkatcal_polcal, meerkatcal_targets).

    Returns
    -------
    dict
        Lists of field names under 'bpcal', 'phasecal', 'polcal' and 'targets'.
    """
    def has(name, intent):
        return any(i.startswith(intent) for i in fields[name]['intents'])

    roles = {}
    roles['bpcal'] = args['meerkatcal_bpcal'] or [n for n in fields if fields[n]['bandpass_calibrator'] is not None]
    roles['polcal'] = args['meerkatcal_polcal'] or [n for n in fields if fields[n]['polarisation_calibrator'] is not None]
    roles['phasecal'] = args['meerkatcal_phasecal'] or [n for n in fields if has(n, 'CALIBRATE_PHASE')
                                                         and n not in roles['bpcal'] + roles['polcal']]
    roles['targets'] = args['meerkatcal_targets'] or [n for n in fields if has(n, 'TARGET')]
    check_roles(roles, fields)
    return roles


def check_roles(roles, fields):
    """Every role field exists, and there is a bandpass calibrator with a model."""
    for role, names in roles.items():
        for name in names:
            if name not in fields:
                raise ValueError(f'{role} field {name} is not in the MS, fields: {list(fields)}')
    unmodelled = [n for n in roles['bpcal'] if fields[n]['bandpass_calibrator'] is None]
    if unmodelled:
        raise ValueError(f'No bandpass calibrator model for {unmodelled}, available: {list(BANDPASS_CALIBRATORS)}')
    if not roles['bpcal']:
        raise ValueError('No bandpass calibrator (J1939-6342 or J0408-6545) found, set --meerkatcal-bpcal')
    return roles


def choose_level(roles, fields, args, has_autocorrelations):
    """
    Calibration level: the requested one, or with 'auto' the highest the observed calibrators allow.

    Parameters
    ----------
    roles : dict
        Output of assign_roles.
    fields : dict
        Output of discover_fields.
    args : dict
        facetselfcal options (meerkatcal_level, meerkatcal_angle).
    has_autocorrelations : bool
        Whether the MS has autocorrelations (needed for the noise diode angle calibration).

    Returns
    -------
    (str, str)
        Level ('standard', 'leakage' or 'full') and the angle source ('celestial', 'diode', 'nullv'
        or None).
    """
    unpolarised = [n for n in roles['bpcal'] if fields[n]['bandpass_calibrator'] in UNPOLARISED_CALIBRATORS]
    angle = args['meerkatcal_angle']
    if angle == 'auto':
        angle = 'celestial' if roles['polcal'] else ('diode' if has_autocorrelations else None)
    level = args['meerkatcal_level']
    if level == 'auto':
        level = 'full' if angle is not None else 'leakage'
    if level in ('leakage', 'full') and not unpolarised:
        terminal_print('WARNING: no unpolarised bandpass calibrator (J1939-6342), the leakage is solved on '
                       + ', '.join(roles['bpcal']) + ' which is weakly polarised')
    if level == 'full' and angle == 'celestial' and not roles['polcal']:
        raise ValueError('Polarisation angle calibration with a celestial calibrator requested, but no 3C286 or 3C138 field')
    if level == 'full' and angle == 'diode' and not has_autocorrelations:
        raise ValueError('Noise diode angle calibration needs autocorrelations, the MS has none')
    if level != 'full':
        angle = None
    return level, angle


def has_autocorrelations(ms, nrow=20000):
    """Whether the first rows of the MS contain autocorrelations."""
    with table(ms, ack=False) as t:
        return bool(np.any(t.getcol('ANTENNA1', 0, nrow) == t.getcol('ANTENNA2', 0, nrow)))


def polcal_scan_checks(ms, field_name, fields, source):
    """
    Elevation, parallactic angle and feed-frame cross-hand power of each polarisation calibrator scan.

    The cross-hand (XY, YX) power of a linearly polarised source in the feed frame is
    p I |sin(2 (EVPA - chi))|, with chi the parallactic angle: scans near EVPA - chi = 0 or 90 deg
    carry no power for the cross-hand phase solve.

    Parameters
    ----------
    ms : str
        Path to the archive Measurement Set (the field's own MSs are used for already split input,
        see field_locations).
    field_name : str
        Name of the polarisation calibrator field.
    fields : dict
        Output of discover_fields or merge_fields.
    source : str
        '3C286' or '3C138'.

    Returns
    -------
    list of dict
        Per scan: scan number, mid time, elevation (deg), parallactic angle (deg) and the fraction of
        the polarised flux in the feed-frame cross-hands at the band centre.
    """
    scan, time = [], []
    for loc_ms, fid, _ in field_locations(fields[field_name], ms):
        with table(loc_ms, ack=False) as t:
            sel = t.getcol('FIELD_ID') == fid
            scan.append(t.getcol('SCAN_NUMBER')[sel])
            time.append(t.getcol('TIME')[sel])
    scan, time = np.concatenate(scan), np.concatenate(time)
    ms = field_locations(fields[field_name], ms)[0][0]
    with table(ms + '::ANTENNA', ack=False) as t:
        pos = np.mean(t.getcol('POSITION'), axis=0)
    with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
        freq = np.mean(t.getcol('CHAN_FREQ')[0]) / 1e9
    evpa = POLARISED_CALIBRATOR_MODELS[source][1](np.array([freq]))[0]
    ra, dec = fields[field_name]['direction']
    dm = measures()
    direction = dm.direction('J2000', qa.quantity(ra, 'rad'), qa.quantity(dec, 'rad'))
    zenith = dm.direction('AZEL', qa.quantity(0., 'deg'), qa.quantity(90., 'deg'))
    dm.do_frame(dm.position('ITRF', *(qa.quantity(p, 'm') for p in pos)))
    out = []
    for s in np.unique(scan):
        tmid = float(np.median(time[scan == s]))
        dm.do_frame(dm.epoch('UTC', qa.quantity(tmid, 's')))
        azel = dm.measure(direction, 'AZEL')
        elevation = np.degrees(azel['m1']['value'])
        chi = dm.posangle(dm.measure(direction, 'APP'), zenith).get_value('rad')
        out.append(dict(scan=int(s), time=tmid, elevation=float(elevation), parang=float(np.degrees(chi)),
                        crosshand_fraction=float(abs(np.sin(2 * (evpa - chi))))))
    return out


def discover(mslist, args, outfile='misc/meerkatcal_discovery.json'):
    """
    Run the discovery and write it to a json file.

    Parameters
    ----------
    mslist : list of str
        The archive Measurement Set (one MS with all fields), or with already split input (split_input:
        the calibrator MSs given with --meerkatcal-bpcal-ms, --meerkatcal-phasecal-ms and
        --meerkatcal-polcal-ms) the target MSs.
    args : dict
        facetselfcal options.
    outfile : str, optional
        Output json file.

    Returns
    -------
    dict
        ms (the archive MS, None for split input), mslist (all input MSs), band, fields, roles, level,
        angle, autocorrelations and polcal_scans.
    """
    if split_input(args):
        mslists = {role: [os.path.abspath(m.rstrip('/')) for m in (args[f'meerkatcal_{role}_ms'] or [])]
                   for role in ('bpcal', 'phasecal', 'polcal')}
        mslists['targets'] = list(mslist)
        allms = [m for mss in mslists.values() for m in mss]
        check_split_inputs(allms, weightgain=args.get('meerkatcal_weightgain', False))
        fields, roles = split_roles(mslists)
        ms = None
        calms = mslists['bpcal'] + mslists['phasecal'] + mslists['polcal']
        autocorr = any(has_autocorrelations(m) for m in calms)
    else:
        ms = mslist[0]
        allms = [ms]
        fields = discover_fields(ms)
        roles = assign_roles(fields, args)
        autocorr = has_autocorrelations(ms)
    with table(allms[0] + '::SPECTRAL_WINDOW', ack=False) as t:
        band = meerkat_band(t.getcol('CHAN_FREQ')[0])
    level, angle = choose_level(roles, fields, args, autocorr)
    polcal_scans = {}
    for name in roles['polcal']:
        polcal_scans[name] = polcal_scan_checks(ms, name, fields, fields[name]['polarisation_calibrator'])
    info = dict(ms=ms, mslist=allms, band=band, fields=fields, roles=roles, level=level, angle=angle,
                autocorrelations=autocorr, polcal_scans=polcal_scans)
    if ms is None:
        terminal_print('Already split input MSs:')
        for role in ('bpcal', 'phasecal', 'polcal', 'targets'):
            for name in roles[role]:
                terminal_print(f"  {role:9s} {name}: " + ', '.join(m for m, _, _ in field_locations(fields[name])))

    terminal_print(f'MeerKAT {band} band, calibration level: {level}' + (f', angle from: {angle}' if angle else ''))
    for name, f in fields.items():
        role = [r for r, names in roles.items() if name in names]
        terminal_print(f"  {name:14s} {','.join(role) or '-':10s} {f['hours']:.2f} h, scans {f['scans']}, intents {f['intents']}")
    for name, scans in polcal_scans.items():
        for s in scans:
            terminal_print(f"  {name} scan {s['scan']}: elevation {s['elevation']:.1f} deg, parallactic angle "
                           f"{s['parang']:.1f} deg, feed-frame cross-hand fraction {s['crosshand_fraction']:.2f}")
        if max(s['crosshand_fraction'] for s in scans) < 0.3:
            terminal_print(f'WARNING: little cross-hand power on {name} in all scans, the angle calibration will be poor')
    if not autocorr:
        terminal_print('No autocorrelations in the MS: the noise diode angle calibration is not possible')
    with open(outfile, 'w') as f:
        json.dump(info, f, indent=1)
    return info
