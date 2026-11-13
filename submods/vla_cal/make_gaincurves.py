"""
One-off export of the CASA VLA antenna gain curves (casa-data nrao/VLA/GainCurves) to
facetselfcal/data/vla_gaincurves.json, used by submods/vla_cal/gaincurve.py.

Usage: python make_gaincurves.py /path/to/casa-data/nrao/VLA/GainCurves [output.json]

Each row of the table holds, for a band (frequency range), a validity time range and an antenna
number (0: the default for antennas without their own curve), the coefficients c_n of the voltage
gain g(za) = sum_n c_n za^n, with za the zenith angle in degrees, for the R and L receptors. CASA
applycal divides a visibility by g_i g_j (checked against gencal(caltype='gc') on test data).
"""
import json
import os
import sys

from casacore.tables import table  # type: ignore

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           'facetselfcal', 'data', 'vla_gaincurves.json')


def export(gaincurves, outfile=DEFAULT_OUT):
    rows = []
    with table(gaincurves, ack=False) as t:
        for row in t:
            if row['GAINTYPE'] != 'ZA':
                raise ValueError('Unsupported GAINTYPE ' + row['GAINTYPE'])
            rows.append(dict(band=row['BANDNAME'], bfreq=float(row['BFREQ']), efreq=float(row['EFREQ']),
                             btime=float(row['BTIME']), etime=float(row['ETIME']), antenna=int(row['ANTENNA']),
                             gain=[[float(c) for c in pol] for pol in row['GAIN']]))
    with open(outfile, 'w') as f:
        json.dump(dict(source='casa-data: ' + os.path.join(*os.path.normpath(gaincurves).split(os.sep)[-3:]), convention='voltage gain g(za) = sum_n c_n za_deg^n, '
                       'per receptor [R, L]; antenna 0 is the default', rows=rows), f, indent=0)
    print(f'Wrote {len(rows)} gain curves to {outfile}')


if __name__ == '__main__':
    export(sys.argv[1], *(sys.argv[2:3] or [DEFAULT_OUT]))
