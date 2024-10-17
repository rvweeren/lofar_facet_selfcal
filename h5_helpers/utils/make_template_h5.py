import tables
import os
import numpy as np
from argparse import ArgumentParser
import ast


class Template:
    """
    Find closest h5 direction in merged h5
    """
    def __init__(self, h5_in, template_name):

        os.system(' '.join(['cp', h5_in, template_name+'.tmp']))

        print(f'Created {template_name}')

        self.name_out = template_name+'.tmp'

    def make_template(self, coord = [0., 0.]):
        """
        Make template h5 with 1 direction
        """
        # open output h5, which will be modified
        with tables.open_file(self.name_out, 'r+') as self.h5:

            # loop over solsets (example: sol000, sol001, ...)
            for solset in self.h5.root._v_groups.keys():
                ss = self.h5.root._f_get_child(solset)
                ss._f_get_child('source')._f_remove()
                values = np.array([(b'Dir00', coord)], dtype=[('name', 'S128'), ('dir', '<f4', (2,))])
                title = 'Source names and directions'
                self.h5.create_table(ss, 'source', values, title=title)

                # loop over soltabs (example: phase000, amplitude000, ...)
                for soltab in ss._v_groups.keys():

                    if 'phase' in soltab or 'amplitude' in soltab:
                        st = ss._f_get_child(soltab)
                        for axes in ['val', 'weight']:
                            AXES = st._f_get_child(axes).attrs['AXES']
                            dir_idx = AXES.decode('utf8').split(',').index('dir')
                            shape = list(st._f_get_child(axes)[:].shape)
                            shape[dir_idx] = 1

                            # only phases are zeros, others are ones
                            if 'phase' in soltab and axes != 'weight':
                                newvals = np.zeros(shape)
                            elif 'amplitude' in soltab or axes == 'weight':
                                newvals = np.ones(shape)
                            else:
                                newvals = np.zeros(shape)

                            # get correct value type
                            valtype = str(st._f_get_child(axes).dtype)
                            if '16' in valtype:
                                atomtype = tables.Float16Atom()
                            elif '32' in valtype:
                                atomtype = tables.Float32Atom()
                            elif '64' in valtype:
                                atomtype = tables.Float64Atom()
                            else:
                                atomtype = tables.Float64Atom()

                            # create new value/weight table
                            st._f_get_child(axes)._f_remove()
                            self.h5.create_array(st, axes, newvals.astype(valtype), atom=atomtype)
                            st._f_get_child(axes).attrs['AXES'] = AXES

                        # modify direction axes
                        st._f_get_child('dir')._f_remove()
                        self.h5.create_array(st, 'dir', np.array([b'Dir00']).astype('|S5'))
                    else:
                        print(f"WARNING: {soltab} to template? --> Script is not advanced enough to add {soltab} to template")

        print(f'Repack {self.name_out} -> {self.name_out.replace(".tmp","")}')
        os.system(f'h5repack {self.name_out} {self.name_out.replace(".tmp","")} && rm {self.name_out}')

        return self


def parse_float_list(string):
    # Remove the square brackets and split the string by commas

    if '[[' in string:
        return ast.literal_eval(string)
    else:
        string = string.strip('[]')
        # Convert each element to a float and return as a list
        return [float(num) for num in string.split(',') if num]


def parse_args():
    """
    Command line argument parser

    :return: parsed arguments
    """

    parser = ArgumentParser(description="Make template h5parm with one direction based on existing h5parm.")
    parser.add_argument('--h5_in', help='Input h5parm', required=True)
    parser.add_argument('--h5_out', help='Output h5parm', required=True)
    parser.add_argument('--outcoor', help='Output coordinates, example: [1.2,3.1]', type=str)

    return parser.parse_args()


def main():
    """Main function"""

    args = parse_args()

    if args.outcoor is not None:
        outcoor = parse_float_list(args.outcoor)
    else:
        outcoor = [0.0, 0.0]

    T = Template(args.h5_in, args.h5_out)
    T.make_template(outcoor)


if __name__ == '__main__':
    main()
