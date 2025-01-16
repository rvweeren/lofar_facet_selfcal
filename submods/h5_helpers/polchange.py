# Standard library imports
import sys
import math
from collections import OrderedDict
import psutil

# Third-party imports
from numpy import (zeros, ones, complex128, array, exp, angle, finfo, abs, memmap, ndarray)
from losoto.h5parm import h5parm
from losoto.lib_operations import reorderAxes
import tables

lin2circ_math = r"""
Convert linear polarization to circular polarization
-----------------------------
RR = XX - iXY + iYX + YY
RL = XX + iXY + iYX - YY
LR = XX - iXY - iYX - YY
LL = XX + iXY - iYX + YY
-----------------------------
"""

circ2lin_math = r"""
Convert circular polarization to linear polarization
-----------------------------
XX = RR + RL + LR + LL
XY = iRR - iRL + iLR - iLL
YX = -iRR - iRL + iLR + iLL
YY = RR - RL - LR + LL
-----------------------------
"""


def overwrite_table(T, solset, table, values, title=None):
    """
    Create table for given solset, opened with the package tables.
    Best to use for antenna or source table.

    :param T: Table opeend with tables
    :param solset: solution set of the table (f.ex. sol000)
    :param table: table name (f.ex. antenna or source)
    :param values: new values
    :param title: title of new table
    """

    # Check if the file is opened with 'tables'
    try:
        T.root
    except AttributeError:
        sys.exit('ERROR: Given table is not opened with the "tables" package (https://pypi.org/project/tables/).')

    # Warning if solset does not follow the usual naming convention
    if not solset.startswith('sol'):
        print('WARNING: Solution set name should start with "sol".')

    # Get the solution set and remove the existing table
    ss = T.root._f_get_child(solset)
    if hasattr(ss, table):
        ss._f_get_child(table)._f_remove()

    # Handle specific cases for source and antenna tables
    if table == 'source':
        values = array(values, dtype=[('name', 'S128'), ('dir', '<f4', (2,))])
        title = title or 'Source names and directions'
    elif table == 'antenna':
        values = array(values, dtype=[('name', 'S16'), ('position', '<f4', (3,))])
        title = title or 'Antenna names and positions'
    else:
        # Ensure 'values' is a valid numpy array
        if not isinstance(values, ndarray):
            values = array(values)

    # Create the new table
    T.create_table(ss, table, values, title=title)

    return


class PolChange:
    """
    This Python class helps to convert polarization from linear to circular or vice versa.
    """

    def __init__(self, h5_in, h5_out):
        """
        :param h5_in: h5 input name
        :param h5_out: h5 output name
        """

        self.h5in_name = h5_in
        self.h5out_name = h5_out
        self.h5_in = h5parm(h5_in, readonly=True)
        self.h5_out = h5parm(h5_out, readonly=False)
        self.axes_names = ['time', 'freq', 'ant', 'dir', 'pol']

    @staticmethod
    def lin2circ(G):
        """
        Convert linear polarization to circular polarization

        RR = XX - iXY + iYX + YY
        RL = XX + iXY + iYX - YY
        LR = XX - iXY - iYX - YY
        LL = XX + iXY - iYX + YY

        :param G: Linear polarized Gain

        :return: Circular polarized Gain
        """

        MEM_AVAIL = psutil.virtual_memory().available * 0.95
        G_SHAPE = G.shape[0:-1] + (4,)
        # Total memory required in bytes.
        MEM_NEEDED = math.prod(G_SHAPE) * 16

        # Performance hit for small H5parms, so prefer to stay in RAM.
        if MEM_NEEDED < MEM_AVAIL:
            G_new = zeros(G_SHAPE).astype(complex128)
        else:
            print("H5parm too large for in-memory conversion. Using memory-mapped approach.")
            G_new = memmap("tempG_new.dat", dtype=complex128, mode="w+", shape=G_SHAPE)

        G_new[..., 0] = (G[..., 0] + G[..., -1])
        G_new[..., 1] = (G[..., 0] - G[..., -1])
        G_new[..., 2] = (G[..., 0] - G[..., -1])
        G_new[..., 3] = (G[..., 0] + G[..., -1])

        if G.shape[-1] == 4:
            G_new[..., 0] += 1j * (G[..., 2] - G[..., 1])
            G_new[..., 1] += 1j * (G[..., 2] + G[..., 1])
            G_new[..., 2] -= 1j * (G[..., 2] + G[..., 1])
            G_new[..., 3] += 1j * (G[..., 1] - G[..., 2])

        G_new /= 2
        G_new[abs(G_new) < 10 * finfo(float).eps] = 0

        return G_new

    @staticmethod
    def circ2lin(G):
        """
        Convert circular polarization to linear polarization

        XX = RR + RL + LR + LL
        XY = iRR - iRL + iLR - iLL
        YX = -iRR - iRL + iLR + iLL
        YY = RR - RL - LR + LL

        :param G: Circular polarized Gain

        :return: linear polarized Gain
        """

        MEM_AVAIL = psutil.virtual_memory().available * 0.95
        G_SHAPE = G.shape[0:-1] + (4,)
        # Total memory required in bytes.
        MEM_NEEDED = math.prod(G_SHAPE) * 16

        # Performance hit for small H5parms, so prefer to stay in RAM.
        if MEM_NEEDED < MEM_AVAIL:
            G_new = zeros(G_SHAPE).astype(complex128)
        else:
            print("H5parm too large for in-memory conversion. Using memory-mapped approach.")
            G_new = memmap("tempG_new.dat", dtype=complex128, mode="w+", shape=G_SHAPE)

        G_new[..., 0] = (G[..., 0] + G[..., -1])
        G_new[..., 1] = 1j * (G[..., 0] - G[..., -1])
        G_new[..., 2] = 1j * (G[..., -1] - G[..., 0])
        G_new[..., 3] = (G[..., 0] + G[..., -1])

        if G.shape[-1] == 4:
            G_new[..., 0] += (G[..., 2] + G[..., 1])
            G_new[..., 1] += 1j * (G[..., 2] - G[..., 1])
            G_new[..., 2] += 1j * (G[..., 2] - G[..., 1])
            G_new[..., 3] -= (G[..., 1] + G[..., 2])

        G_new /= 2
        G_new[abs(G_new) < 10 * finfo(float).eps] = 0

        return G_new

    @staticmethod
    def add_polarization(values, dim_pol):
        """
        Add extra polarization if there is no polarization

        :param values: values which need to get a polarization
        :param dim_pol: number of dimensions

        :return: input values with extra polarization axis
        """

        values_new = ones(values.shape + (dim_pol,))
        for i in range(dim_pol):
            values_new[..., i] = values

        return values_new

    def create_template(self, soltab):
        """
        Make template of the gains with only ones

        :param soltab: solution table (phase, amplitude)
        """

        self.G, self.axes_vals = array([]), OrderedDict()
        for ss in self.h5_in.getSolsetNames():
            for st in self.h5_in.getSolset(ss).getSoltabNames():
                solutiontable = self.h5_in.getSolset(ss).getSoltab(st)
                if soltab in st:
                    try:
                        if 'pol' in solutiontable.getAxesNames():
                            values = reorderAxes(solutiontable.getValues()[0],
                                                 solutiontable.getAxesNames(),
                                                 self.axes_names)
                            self.G = ones(values.shape).astype(complex128)
                        else:
                            values = reorderAxes(solutiontable.getValues()[0],
                                                 solutiontable.getAxesNames(),
                                                 self.axes_names[0:-1])
                            self.G = ones(values.shape + (2,)).astype(complex128)
                    except:
                        sys.exit('ERROR: Received ' + str(solutiontable.getAxesNames()) +
                                 ', but expect at least [time, freq, ant, dir] or [time, freq, ant, dir, pol]')

                    self.axes_vals = {'time': solutiontable.getAxisValues('time'),
                                      'freq': solutiontable.getAxisValues('freq'),
                                      'ant': solutiontable.getAxisValues('ant'),
                                      'dir': solutiontable.getAxisValues('dir'),
                                      'pol': ['XX', 'XY', 'YX', 'YY']}
                    break

        print('Value shape {soltab} before --> {shape}'.format(soltab=soltab, shape=self.G.shape))

        return self

    def add_tec(self, solutiontable):
        """
        Add TEC

        :param solutiontable: the solution table for the TEC
        """

        tec_axes_names = [ax for ax in self.axes_names if solutiontable.getAxesNames()]
        tec = reorderAxes(solutiontable.getValues()[0], solutiontable.getAxesNames(), tec_axes_names)
        if 'freq' in solutiontable.getAxesNames():
            axes_vals_tec = {'time': solutiontable.getAxisValues('time'),
                             'freq': solutiontable.getAxisValues('freq'),
                             'ant': solutiontable.getAxisValues('ant'),
                             'dir': solutiontable.getAxisValues('dir')}
        else:
            axes_vals_tec = {'dir': solutiontable.getAxisValues('dir'),
                             'ant': solutiontable.getAxisValues('ant'),
                             'time': solutiontable.getAxisValues('time')}
        if 'pol' in solutiontable.getAxesNames():
            if tec.shape[-1] == 2:
                axes_vals_tec.update({'pol': ['XX', 'YY']})
            elif tec.shape[-1] == 4:
                axes_vals_tec.update({'pol': ['XX', 'XY', 'YX', 'YY']})
        axes_vals_tec = [v[1] for v in
                         sorted(axes_vals_tec.items(), key=lambda pair: self.axes_names.index(pair[0]))]
        self.solsetout.makeSoltab('tec', axesNames=tec_axes_names, axesVals=axes_vals_tec, vals=tec,
                                  weights=ones(tec.shape))

        return self

    def create_new_gain_table(self, lin2circ, circ2lin):
        """
        Create new gain tables with polarization conversion.

        :param lin2circ: boolean for linear to circular conversion
        :param circ2lin: boolean for circular to linear conversion
        """

        for ss in self.h5_in.getSolsetNames():

            self.solsetout = self.h5_out.makeSolset(ss)

            for n, st in enumerate(self.h5_in.getSolset(ss).getSoltabNames()):
                solutiontable = self.h5_in.getSolset(ss).getSoltab(st)

                print('{ss}/{st} from {h5}'.format(ss=ss, st=st, h5=self.h5in_name))
                if 'phase' in st:
                    if 'pol' in solutiontable.getAxesNames():
                        values = reorderAxes(solutiontable.getValues()[0], solutiontable.getAxesNames(),
                                             self.axes_names)
                        self.G *= exp(values * 1j)
                    else:
                        values = reorderAxes(solutiontable.getValues()[0], solutiontable.getAxesNames(),
                                             self.axes_names[0:-1])
                        self.G *= exp(self.add_polarization(values, 2) * 1j)

                elif 'amplitude' in st:
                    if 'pol' in solutiontable.getAxesNames():
                        values = reorderAxes(solutiontable.getValues()[0], solutiontable.getAxesNames(),
                                             self.axes_names)
                        self.G *= values
                    else:
                        values = reorderAxes(solutiontable.getValues()[0], solutiontable.getAxesNames(),
                                             self.axes_names[0:-1])
                        self.G *= self.add_polarization(values, 2)

                elif 'tec' in st:
                    self.add_tec(solutiontable)
                else:
                    print("WARNING: didn't include {st} in this h5_merger.py version yet".format(st=st) +
                          "\nPlease create a ticket on github if this needs to be changed")

                if n == 0:
                    weights = ones(values.shape)
                weight = solutiontable.getValues(weight=True, retAxesVals=False)
                if 'pol' in solutiontable.getAxesNames():
                    weight = reorderAxes(weight, solutiontable.getAxesNames(), self.axes_names)
                else:
                    weight = reorderAxes(weight, solutiontable.getAxesNames(), self.axes_names[0:-1])
                weights *= weight

            if lin2circ:
                print(lin2circ_math)
                G_new = self.lin2circ(self.G)
            elif circ2lin:
                print(circ2lin_math)
                G_new = self.circ2lin(self.G)
            else:
                sys.exit('ERROR: No conversion given.')
            print('Value shape after --> {shape}'.format(shape=G_new.shape))

            phase = angle(G_new)
            amplitude = abs(G_new)

            # upsample weights
            if phase.shape != weights.shape:
                new_weights = ones(phase.shape)
                for s in range(new_weights.shape[-1]):
                    if len(weights.shape) == 4:
                        new_weights[..., s] *= weights[:]
                    elif len(weights.shape) == 5:
                        new_weights[..., s] *= weights[..., 0]
                weights = new_weights

            self.axes_vals = [v[1] for v in
                              sorted(self.axes_vals.items(), key=lambda pair: self.axes_names.index(pair[0]))]

            self.solsetout.makeSoltab('phase', axesNames=self.axes_names, axesVals=self.axes_vals, vals=phase,
                                      weights=weights)
            print('Created new phase solutions')

            self.solsetout.makeSoltab('amplitude', axesNames=self.axes_names, axesVals=self.axes_vals, vals=amplitude,
                                      weights=weights)
            print('Created new amplitude solutions')

            # convert the polarization names, such that it is clear if the h5 is in circular or linear polarization
            for solset in self.h5_out.getSolsetNames():
                ss = self.h5_out.getSolset(solset)
                for soltab in ss.getSoltabNames():
                    st = ss.getSoltab(soltab)
                    if 'pol' in st.getAxesNames():
                        pols = st.getAxisValues('pol')
                        if len(pols) == 2:
                            if lin2circ:
                                st.setAxisValues('pol', ['RR', 'LL'])
                            elif circ2lin:
                                st.setAxisValues('pol', ['XX', 'YY'])
                        if len(pols) == 4:
                            if lin2circ:
                                st.setAxisValues('pol', ['RR', 'RL', 'LR', 'LL'])
                            elif circ2lin:
                                st.setAxisValues('pol', ['XX', 'XY', 'YX', 'YY'])

        self.h5_in.close()
        self.h5_out.close()

        return self

    def add_antenna_source_tables(self):
        """
        Add antenna and source table to output file
        """

        with tables.open_file(self.h5in_name) as T:
            with tables.open_file(self.h5out_name, 'r+') as H:
                for solset in T.root._v_groups.keys():
                    ss = T.root._f_get_child(solset)
                    overwrite_table(H, solset, 'antenna', ss.antenna[:])
                    overwrite_table(H, solset, 'source', ss.source[:])

        return
