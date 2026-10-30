"""
This code is based on MockPyStep.py from https://dp3.readthedocs.io/en/latest/steps/PythonStep.html
Parallactic angle (P-Jones) correction for alt-az dishes, the DP3 counterpart of CASA applycal(parang=True).
The angles are computed as in QuartiCal (quartical/data_handling/angles.py).

To use this script in DPPP/DP3, you can save this file in the same folder as from where you run your DPPP command.
Or you save this python file somewhere you like and run:
export PYTHONPATH=/somewhere/you/like:$PYTHONPATH

Example:
DP3 msin=my.ms msout=. msout.datacolumn=CORRECTED_DATA steps=[pa] \
    pa.type=PythonDPPP pa.python.module=parang pa.python.class=ParAng pa.mode=correct

"""
from subprocess import check_output
import re
import numpy as np
import shutil
import sys
from casacore.tables import table
from casacore.measures import measures
import casacore.quanta as qa

#hacky way to figure out the DPPP/DP3 version (important to run this script properly)
DP3name = shutil.which('DP3')
if not DP3name:
    DP3name = shutil.which('DPPP')
try:
    rgx = r'[0-9]+(\.[0-9]+)+'
    grep_version_string = str(check_output(DP3name+' --version', shell=True), 'utf-8')
    DP3_VERSION = float(re.search(rgx, grep_version_string).group()[0:3])
except AttributeError:
    print('WARNING: grep of DP3 version failed.')
    DP3_VERSION=0.0 # set to default

if DP3_VERSION > 5.3:
    from dp3 import Fields

try:
    from dppp import DPStep as Step
except:
    if DP3_VERSION >= 6:
        from dp3.pydp3 import Step
    else:
        from dp3 import Step


def get_parset_value(parset, key, default, kind='string'):
    """Read a parset key, trying the old (getString/getInt) and new (get_string/get_int) DP3 bindings."""
    names = {'string': ('getString', 'get_string'), 'int': ('getInt', 'get_int')}[kind]
    for name in names:
        try:
            return getattr(parset, name)(key)
        except AttributeError:
            continue  # DP3 Python bindings have been renamed.
        except RuntimeError:
            return default  # key not present
    return default


class ParAng(Step):
    r"""
    Apply (mode=corrupt) or remove (mode=correct) the parallactic angle rotation of the feeds
    of alt-az mounted dishes, per antenna and per time slot.

    The feed rotation chi = parallactic angle (+ FEED RECEPTOR_ANGLE if receptorangle=1) enters as

        V_obs = P_p V_sky P_q^H

    with (QuartiCal's rotation_factory)

                  / exp(-i chi)      0      \                / cos(chi)  sin(chi) \
    circular: P = |                         |,   linear: P = |                     |
                  \     0        exp(i chi) /                \ -sin(chi) cos(chi) /

    mode=correct applies P^-1 to the data (as CASA applycal parang=True does),
    mode=corrupt applies P (for example to a sky-frame model column).

    Parset keys (prefix pystep.):
      mode          correct (default) or corrupt
      receptorangle 1 (default): add FEED RECEPTOR_ANGLE to the parallactic angle (as CASA does);
                    0: ignore it. Note that MeerKAT MSs have RECEPTOR_ANGLE = -90 deg.
      feedtype      linear or circular; only needed when the correlations in the DP3 stream
                    differ from those in the MS (e.g. after a polconv step). Default: from the MS.
      swapfeeds     0 (default) or 1: also swap the X and Y feeds, F = [[0,1],[1,0]] (linear feeds only),
                    and ignore the FEED RECEPTOR_ANGLE. This is the MeerKAT convention fix as in
                    mepi (lib_ms.py swap_feeds). Rotating by MeerKAT's RECEPTOR_ANGLE = -90 deg instead
                    (what CASA does) also exchanges XX and YY, but flips the sign of XY and YX (so of U and V).
                    correct applies P^-1 F^-1 (swap, then derotate), corrupt applies F P.
      frame         apparent (default): parallactic angle with respect to the apparent north pole
                    of date, as CASA does; J2000: with respect to the J2000 north pole, as QuartiCal
                    does. These differ by precession/nutation, about 0.2 deg for current dates.

    Note: removing P from DATA before a fulljones/leakage solve makes the leakage (D) terms
    time variable, since P and D do not commute. For leakage calibration corrupt the model instead.
    """

    def __init__(self, parset, prefix):
        """
        Set up the step (constructor). Read the parset here.

        Args:
          parset: Parameter set for the entire pipeline
          prefix: Prefix for this step."
        """

        super().__init__()

        self.mode = get_parset_value(parset, prefix + "mode", "correct").lower()
        if self.mode not in ('correct', 'corrupt'):
            sys.exit("ParAng: mode should be correct or corrupt, not " + self.mode)
        self.use_receptorangle = bool(get_parset_value(parset, prefix + "receptorangle", 1, kind='int'))
        self.feedtype = get_parset_value(parset, prefix + "feedtype", "").lower()
        if self.feedtype not in ('', 'linear', 'circular'):
            sys.exit("ParAng: feedtype should be linear or circular, not " + self.feedtype)
        self.swapfeeds = bool(get_parset_value(parset, prefix + "swapfeeds", 0, kind='int'))
        if self.swapfeeds:
            self.use_receptorangle = False
        self.frame = get_parset_value(parset, prefix + "frame", "apparent").lower()
        if self.frame not in ('apparent', 'j2000'):
            sys.exit("ParAng: frame should be apparent or J2000, not " + self.frame)

        self.chi_min = np.inf
        self.chi_max = -np.inf
        self.fetch_uvw = True

    def get_required_fields(self):
        if DP3_VERSION>5.3:
            return (Fields.DATA | Fields.FLAGS | Fields.WEIGHTS | Fields.UVW)
        else:
            pass

    def get_provided_fields(self):
        if DP3_VERSION>5.3:
            return (Fields.DATA | Fields.FLAGS | Fields.WEIGHTS | Fields.UVW)
        else:
            pass

    def update_info(self, dpinfo):
        """
        Process metadata. This will be called before any call to process.
        Collects antennas, mounts, receptor angles, feed type and the phase center,
        and sets up the casacore measures needed for the parallactic angles.

        Args:
          dpinfo: DPInfo object with all metadata, see docs in pydp3.cc
        """

        super().update_info(dpinfo)

        ms = dpinfo.ms_name
        self.ant1 = np.array(dpinfo.first_antenna_indices)
        self.ant2 = np.array(dpinfo.second_antenna_indices)
        self.ncorr = dpinfo.n_correlations
        ant_names = list(dpinfo.antenna_names)
        ant_positions = np.array(dpinfo.antenna_positions)
        ra, dec = dpinfo.phase_center

        with table(ms + '::OBSERVATION', ack=False, readonly=True) as t:
            telescope = t.getcol('TELESCOPE_NAME')[0]
        if telescope.upper() == 'LOFAR':
            sys.exit("ParAng: LOFAR is an aperture array, use the applybeam step instead.")

        if not self.feedtype:
            with table(ms + '::POLARIZATION', ack=False, readonly=True) as t:
                corr_type = t.getcol('CORR_TYPE')[0]
            if np.all(np.isin(corr_type, [5, 6, 7, 8])):
                self.feedtype = 'circular'
            elif np.all(np.isin(corr_type, [9, 10, 11, 12])):
                self.feedtype = 'linear'
            else:
                sys.exit("ParAng: unsupported correlation types " + str(corr_type))
        if self.swapfeeds and self.feedtype != 'linear':
            sys.exit("ParAng: swapfeeds is only supported for linear feeds.")
        if self.ncorr != 4 and not (self.feedtype == 'circular' and self.ncorr == 2):
            sys.exit("ParAng: need 4 correlations (or RR,LL for circular feeds), got " + str(self.ncorr))

        # mounts and receptor angles, matched to the DP3 antennas by name
        with table(ms + '::ANTENNA', ack=False, readonly=True) as t:
            ms_ant_names = list(t.getcol('NAME'))
            ms_mounts = [m.upper() for m in t.getcol('MOUNT')]
        with table(ms + '::FEED', ack=False, readonly=True) as t:
            feed_ant = t.getcol('ANTENNA_ID')
            feed_angle = t.getcol('RECEPTOR_ANGLE')
        receptor_angle = np.zeros(len(ant_names))
        self.altaz = np.ones(len(ant_names), dtype=bool)
        for i, name in enumerate(ant_names):
            if name not in ms_ant_names:
                print('ParAng WARNING: antenna ' + name + ' not in the MS ANTENNA table, using ALT-AZ and receptor angle 0')
                continue
            ms_index = ms_ant_names.index(name)
            mount = ms_mounts[ms_index]
            if mount == 'EQUATORIAL':
                self.altaz[i] = False
            elif mount not in ('ALT-AZ', 'ALTAZ', 'ALT_AZ', ''):
                sys.exit("ParAng: mount type " + mount + " of antenna " + name + " is not supported.")
            rows = np.where(feed_ant == ms_index)[0]
            if len(rows) > 0:
                receptor_angle[i] = feed_angle[rows[0], 0]
        self.receptor_angle = receptor_angle if self.use_receptorangle else np.zeros(len(ant_names))

        self.dm = measures()
        self.field = self.dm.direction('J2000', qa.quantity(ra, 'rad'), qa.quantity(dec, 'rad'))
        self.zenith = self.dm.direction('AZEL', qa.quantity(0., 'deg'), qa.quantity(90., 'deg'))
        self.positions = [self.dm.position('ITRF', *(qa.quantity(p, 'm') for p in pos)) for pos in ant_positions]

    def show(self):
        """Print a summary of the step and its settings"""

        print("\nParAng")
        print("  mode:          " + self.mode + (" (remove P-Jones)" if self.mode == 'correct' else " (apply P-Jones)"))
        print("  feed type:     " + self.feedtype)
        print("  frame:         " + self.frame)
        print("  receptorangle: " + str(self.use_receptorangle))
        print("  swapfeeds:     " + str(self.swapfeeds))
        if self.use_receptorangle and np.any(self.receptor_angle != 0):
            print("  NOTE: adding non-zero FEED RECEPTOR_ANGLE (deg): " + str(np.unique(np.round(np.degrees(self.receptor_angle), 3))))
        if not np.all(self.altaz):
            print("  " + str(np.sum(~self.altaz)) + " equatorially mounted antennas, no parallactic rotation for these")

    def parallactic_angles(self, time):
        """Parallactic angle (rad) per antenna at time (MJD seconds), 0 for equatorial mounts."""
        self.dm.do_frame(self.dm.epoch('UTC', qa.quantity(time, 's')))
        chi = np.zeros(len(self.positions))
        for i, pos in enumerate(self.positions):
            if self.altaz[i]:
                self.dm.do_frame(pos)
                field = self.dm.measure(self.field, 'APP') if self.frame == 'apparent' else self.field
                chi[i] = self.dm.posangle(field, self.zenith).get_value('rad')
        return chi

    def process(self, dpbuffer):
        """
        Process one time slot of data. This function MUST call process_next_step.

        Args:
          dpbuffer: DPBuffer object which can contain data, flags and weights
                    for one time slot.
        """

        data = np.array(dpbuffer.get_data(), copy=False, dtype=np.complex64)

        chi_ant = self.parallactic_angles(dpbuffer.get_time())
        self.chi_min = min(self.chi_min, np.min(chi_ant[self.altaz]) if np.any(self.altaz) else 0.)
        self.chi_max = max(self.chi_max, np.max(chi_ant[self.altaz]) if np.any(self.altaz) else 0.)
        chi_ant += self.receptor_angle
        if self.mode == 'correct':
            chi_ant = -chi_ant  # P^-1(chi) = P(-chi)
        chi_p = chi_ant[self.ant1][:, None]
        chi_q = chi_ant[self.ant2][:, None]

        if self.feedtype == 'circular':
            # P diagonal: only phases, flags and weights are unaffected
            a_p = np.exp(-1j * chi_p)
            a_q = np.exp(-1j * chi_q)
            if self.ncorr == 4:
                newdata = np.stack([a_p * np.conj(a_q) * data[:, :, 0],  # RR
                                    a_p * a_q * data[:, :, 1],  # RL
                                    np.conj(a_p * a_q) * data[:, :, 2],  # LR
                                    np.conj(a_p) * a_q * data[:, :, 3]], axis=-1)  # LL
            else:
                newdata = np.stack([a_p * np.conj(a_q) * data[:, :, 0],  # RR
                                    np.conj(a_p) * a_q * data[:, :, 1]], axis=-1)  # LL
        else:
            # rotation mixes XX,XY,YX,YY: flag all correlations if one is flagged
            flags = np.array(dpbuffer.get_flags(), copy=False)
            anyflag = np.any(flags, axis=-1)
            flags[...] = anyflag[:, :, None]
            data[anyflag] = 0.
            rot_p = np.array([[np.cos(chi_p[:, 0]), np.sin(chi_p[:, 0])],
                              [-np.sin(chi_p[:, 0]), np.cos(chi_p[:, 0])]]).transpose(2, 0, 1)
            rot_q = np.array([[np.cos(chi_q[:, 0]), np.sin(chi_q[:, 0])],
                              [-np.sin(chi_q[:, 0]), np.cos(chi_q[:, 0])]]).transpose(2, 0, 1)
            if self.swapfeeds:
                # correct: P^-1 F (swap the columns), corrupt: F P (swap the rows)
                axis = 2 if self.mode == 'correct' else 1
                rot_p = np.flip(rot_p, axis=axis)
                rot_q = np.flip(rot_q, axis=axis)
                weights = np.array(dpbuffer.get_weights(), copy=False)
                weights[...] = weights[:, :, ::-1].copy()  # XX,XY,YX,YY -> YY,YX,XY,XX
            vis = data.reshape(data.shape[0], data.shape[1], 2, 2)
            newdata = np.einsum('bij,bfjk,blk->bfil', rot_p, vis, rot_q).reshape(data.shape)  # P_p V P_q^T

        data[...] = newdata

        # Send processed data to the next step
        if DP3_VERSION>5.3:
            next_step = self.get_next_step()
            if next_step is not None:
                next_step.process(dpbuffer)
        else:
            self.process_next_step(dpbuffer)

    def finish(self):
        """
        If there is any remaining data, process it. This can be useful if the
        step accumulates multiple time slots.
        """

        print('ParAng: ' + ('removed' if self.mode == 'correct' else 'applied') +
              ' parallactic angle rotation (' + self.feedtype + ' feeds), parallactic angle range ' +
              '{:.2f} to {:.2f} deg'.format(np.degrees(self.chi_min), np.degrees(self.chi_max)), flush=True)
        pass
