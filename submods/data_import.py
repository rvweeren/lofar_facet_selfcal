"""Importing data into Measurement Sets: GMRT LTA to UVFITS, GMRT UVFITS to
MS, and downloading the MWA beam model."""
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

from submods.runtime_utils import get_logger, run, terminal_print


def download_MWA_beam_model(dest_dir='mwapy/data'):
    """
    Download and extract the MWA embedded element beam model pattern.

    Downloads the MWA coefficient archive from support.astron.nl if the file
    ``mwa_full_embedded_element_pattern.h5`` is not already present, uncompresses
    and extracts it, and places it into the target directory (default: ``mwapy/data/``).

    Parameters
    ----------
    dest_dir : str, optional
        Destination directory to store the beam model HDF5 file.
        Default is 'mwapy/data'.

    Returns
    -------
    str
        Path to the installed beam model file.
    """
    archive_bz2 = 'MWA_COEFF.tar.bz2'
    archive_tar = 'MWA_COEFF.tar'
    h5_name = 'mwa_full_embedded_element_pattern.h5'
    target_h5 = os.path.join(dest_dir, h5_name)

    if os.path.exists(target_h5):
        get_logger().info(f"MWA beam model already exists at {target_h5}")
        return target_h5

    if os.path.exists(h5_name):
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(h5_name, target_h5)
        get_logger().info(f"Moved existing {h5_name} to {target_h5}")
        return target_h5

    url = 'https://support.astron.nl/software/ci_data/EveryBeam/mwa_full_embedded_element_pattern.tar.bz2'
    terminal_print(f"Downloading MWA beam model from {url}...")
    get_logger().info(f"Downloading MWA beam model from {url}")

    if shutil.which('wget'):
        subprocess.run(['wget', '-q', url, '-O', archive_bz2], check=True)
    else:
        import urllib.request
        urllib.request.urlretrieve(url, archive_bz2)

    if shutil.which('bunzip2'):
        subprocess.run(['bunzip2', '-f', archive_bz2], check=True)
    else:
        import bz2
        with bz2.BZ2File(archive_bz2, 'rb') as f_in, open(archive_tar, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        if os.path.exists(archive_bz2):
            os.remove(archive_bz2)

    if shutil.which('tar'):
        subprocess.run(['tar', 'xf', archive_tar], check=True)
    else:
        import tarfile
        with tarfile.open(archive_tar, 'r:') as tar:
            tar.extractall('.')

    os.makedirs(dest_dir, exist_ok=True)

    if os.path.exists(h5_name):
        shutil.move(h5_name, target_h5)

    if os.path.exists(archive_tar):
        os.remove(archive_tar)

    terminal_print(f"MWA beam model installed to {target_h5}")
    get_logger().info(f"MWA beam model installed to {target_h5}")
    return target_h5


def gmrt_uvfits2ms(uvfits, msout, flagfile=''): 
    """
    Convert GMRT uvfits file to CASA Measurement Set format using casapy importgmrt.
    Parameters
    ----------
    uvfits : str
        Path to the input GMRT uvfits file.
    msout : str
        Path to the output Measurement Set.
    flagfile : str, optional
        Path to the flag file (default is an empty string). If provided, this file will be used during the import process.
    Notes
    -----
    This function constructs and runs a command to execute the casapy importgmrt task via a helper script.
    """
    
    datapathc = os.path.dirname(os.path.abspath(__file__))
    submodpathc = '/'.join(datapathc.split('/')[0:-1])+'/submods'

    if not os.path.isfile(uvfits):
        raise Exception('Input uvfits file ' + uvfits + ' does not exist')

    cmdcasa = f'python {submodpathc}/casa_importgmrt.py '
    cmdcasa += '--uvfits=' + uvfits + ' --msout=' + msout + ' --flagfile=' + flagfile + ' '
    if os.path.isdir(msout):
        raise Exception('Output MS ' + msout + ' already exists, please remove it first')
    run(cmdcasa)


def convert_lta_to_uvfits(lta_file_name, uvfits_file_name=None, target_list=[],
                          stokes_list=['RR', 'RL', 'LR', 'LL'], keep_all_cals=True, max_scans=256,
                          scan_offset=0, correction_list=[], version='2.03', flag_scans=[],
                          bpcal_names=[]):
    """
    Convert LTA file to UVFITS format. Original code by Huib Intema, modified and made compatible with Python 3.
    For more information on SPAM and LTA to UVFITS conversion, see:
    https://www.intema.nl/doku.php?id=huibintema:spam:start (Intema et al., 2009 2009, A&A, 501, 1185).
    Original code reference: https://ui.adsabs.harvard.edu/abs/2014ascl.soft08006I/abstract

    Parameters
    ----------
    lta_file_name : str
        Path to the input LTA file.
    uvfits_file_name : str, optional
        Path to the output UVFITS file. If None, it will be set to the input LTA file name with '.UVFITS' appended.
    target_list : list of str, optional
        List of target source names to include. Default is an empty list (include all).
    stokes_list : list of str, optional
        List of Stokes parameters to include. Default is ['RR', 'RL', 'LR', 'LL'].
    keep_all_cals : bool, optional
        Whether to keep all calibration scans. Default is True.
    max_scans : int, optional
        Maximum number of scans to process. Default is 256.
    scan_offset : int, optional
        Offset to apply to scan numbers. Default is 0.
    correction_list : list of str, optional
        List of corrections to apply. Default is an empty list.
    version : str, optional
        Version of the listscan/gvfits tools to use. Default is '2.03'.
    flag_scans : list of int, optional
        List of scan numbers to flag. Default is an empty list.
    bpcal_names : list of str, optional
        List of bandpass calibrator source names. Default is an empty list.

    Returns
    -------
    source_list
        list of str
        List of source names included in the UVFITS file.

    Notes
    -----
    Copied from SPAM Huib Intema's original code and modified for Python 3.
    For listscan/gvfits versions, see
    https://ftp.strw.leidenuniv.nl/intema/spam/
    """
    
    # for standalone usage
    datapathc = os.path.dirname(os.path.abspath(__file__))
    
    # copy nessesary files to current directory
    gmrtdatapath = '/'.join(datapathc.split('/')[0:-1])+'/facetselfcal/gmrt'
    shutil.copy(f'{gmrtdatapath}/listscan-{version}', '.')
    shutil.copy(f'{gmrtdatapath}/gvfits-{version}', '.')
    libsgmrt = glob.glob(f'{gmrtdatapath}/lib/*.so*')
    os.makedirs('lib', exist_ok=True)
    for lib in libsgmrt:
        shutil.copy(lib, 'lib/')

    # make excutable
    for executable in (f'listscan-{version}', f'gvfits-{version}'):
        executable_path = Path(executable)
        executable_path.chmod(executable_path.stat().st_mode | 0o111)

    # version of listscan/gvfits
    listscan = './listscan-%s' % (version)
    gvfits = './gvfits-%s' % (version)

    # add lib to LD_LIBRARY_PATH
    # export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SPAM_PATH}/lib
    if 'LD_LIBRARY_PATH' in os.environ:
        os.environ['LD_LIBRARY_PATH'] = os.path.abspath('lib') + ':' + os.environ['LD_LIBRARY_PATH']
    else:
        os.environ['LD_LIBRARY_PATH'] = os.path.abspath('lib')

    # hard-coded
    pad_char_offset = 127
    record_length = 336
    name_length = 12
    name_offset_start = -5 * 16
    name_offset_end = 14 * 16
    
    # check presence of files
    if not os.path.isfile(lta_file_name):
        raise Exception('LTA file %s does not exist' % (lta_file_name))
    if uvfits_file_name is None:
        uvfits_file_name = os.path.basename(lta_file_name) + '.UVFITS'
    if os.path.isfile(uvfits_file_name):
        raise Exception('UVFITS file %s already exist' % (uvfits_file_name))
    terminal_print('UVFITS file:', uvfits_file_name)
   
    # list contents of LTA file
    file_name = lta_file_name.split('/')[-1]
    extension = file_name.split('.')[-1]
    if extension in ['lta', 'ltb']:
        extension = ''
    else:
        extension = '.' + extension
        file_name = file_name[0:-len(extension)]
    ltab = file_name.split('.')[-1]
    if ltab not in ['lta', 'ltb']:
        raise Exception('LTA file name %s extension not recognized' % (lta_file_name))
    file_name = file_name[:file_name.index('.' + ltab)]
    lsin_file_name = file_name + '.log'
    if os.path.isfile(lsin_file_name):
        os.remove(lsin_file_name)
    plan_file_name = file_name + '.plan'
    if os.path.isfile(plan_file_name):
        os.remove(plan_file_name)
  
    if sys.stdout.name == '<stdout>':
        result = subprocess.run([listscan, lta_file_name]).returncode
    else:
        sys.stdout.flush()
        sys.stderr.flush()
        with open(sys.stdout.name, 'a') as logfile:
            result = subprocess.run([listscan, lta_file_name], stdout=logfile, stderr=subprocess.STDOUT).returncode
        sys.stdout.flush()
        sys.stderr.flush()
    if result != 0:
        raise Exception('LTA to UVFITS conversion failed for %s' % (lta_file_name))

    # get overview of scans
    scan_list = []
    source_list = []
    with open(lsin_file_name, mode='r') as lsin_file:
        lsin_list = lsin_file.readlines()
    
    for line in lsin_list:
        words = line.split()
        if len(words) == 0:
            continue
        if words[0] != 'Scan':
            continue
        if len(words) == 7:
            words = words[0:2] + [''] + words[2:]
        if len(words) != 8:
            raise Exception('unknown formatted line in listscan log')
        scan_list.append(int(words[1]))
        source_list.append(words[2])
    
    with open(plan_file_name, mode='rb') as plan_file:
        plan_data = plan_file.read()
    
    index_list = []
    index = -name_length
    pad_char = None
    for source in source_list:
        if len(source) > 0:
            dindex = plan_data[index + name_length:].find(source.encode() if isinstance(source, str) else source)
            if dindex == -1:
                raise Exception('mismatch between listscan log and plan files')
            index = index + name_length + dindex
        else:
            index = index + record_length
        index_list.append(index)
        if pad_char is None:
            char_val = plan_data[index + name_length - 1]
            if isinstance(char_val, int):
                char_val = chr(char_val)
            if char_val in ['\x00', ' ']:
                pad_char = char_val
    
    # fix listscan files if needed
    if len(correction_list) > 0:
        for scan_id, old_name, new_name in correction_list:
            if (scan_id >= 0) and (scan_id not in scan_list):
                raise Exception('scan id %d not found in observation' % (scan_id))
            if scan_id >= 0:
                if source_list[scan_list.index(scan_id)] != old_name:
                    raise Exception('source name %s does not correspond to scan id %d' %
                                    (old_name, scan_id))
            else:
                if old_name not in source_list:
                    raise Exception('source name %s not found in observation' % (old_name))
    
    if len(correction_list) > 0:
        for scan_id, old_name, new_name in correction_list:
            if scan_id >= 0:
                for j, line in enumerate(lsin_list):
                    words = line.split()
                    if len(words) == 0:
                        continue
                    if words[0] != 'Scan':
                        continue
                    if len(words) == 7:
                        words = words[0:2] + [''] + words[2:]
                    if int(words[1]) != scan_id:
                        continue
                    if len(words[2]) > 0:
                        index = line.index(words[2])
                    else:
                        index = 10
                    lsin_list[j] = (line[:index] + new_name.ljust(name_length) +
                                    line[index + name_length:])
                    break
                old_index = index_list[scan_list.index(scan_id)]
                try:
                    new_index = index_list[source_list.index(new_name)]
                except ValueError:
                    terminal_print('WARNING: source %s to replace %s is not found in source list, skipping' %
                          (new_name, old_name))
                    flag_scans.append(scan_id)
                    continue
                plan_data = (plan_data[:old_index + name_offset_start] +
                             plan_data[new_index + name_offset_start:new_index + name_offset_end] +
                             plan_data[old_index + name_offset_end:])
                source_list[scan_id] = new_name
            else:
                i = -1
                while old_name in source_list[i + 1:]:
                    i = i + 1 + source_list[i + 1:].index(old_name)
                    for j, line in enumerate(lsin_list):
                        words = line.split()
                        if len(words) == 0:
                            continue
                        if words[0] != 'Scan':
                            continue
                        if int(words[1]) != scan_list[i]:
                            continue
                        index = line.index(words[2])
                        lsin_list[j] = (line[:index] + new_name.ljust(name_length) +
                                        line[index + name_length:])
                        break
                    old_index = index_list[i]
                    new_index = index_list[source_list.index(new_name)]
                    plan_data = (plan_data[:old_index + name_offset_start] +
                                 plan_data[new_index + name_offset_start:new_index + name_offset_end] +
                                 plan_data[old_index + name_offset_end:])
                    source_list[scan_list[i]] = new_name
        
        with open(plan_file_name, mode='wb') as plan_file:
            plan_file.write(plan_data)
    
    # filter contents of LTA file
    if len(bpcal_names) == 0:
        bpcal_names = ['3C48', '3C138', '3C147', '3C196', '3C286', '3C295', '3C380', '3C468.1']
    
    lsout_file_name = lsin_file_name + '.sel'
    if os.path.isfile(lsout_file_name):
        os.remove(lsout_file_name)
    
    file_name = (file_name + ltab[-1:]).upper() + '.UVFITS' + extension
    if os.path.isfile(file_name):
        os.remove(file_name)
    
    with open(lsout_file_name, mode='w') as lsout_file:
        source_list = []
        scan_in_range = False
        for line in lsin_list:
            words = line.split()
            if len(words) == 0:
                lsout_file.write(line)
                continue
            if words[0] == 'FITS':
                line = line[:line.index(words[1])]
                line = line + file_name + '\n'
            elif words[0] == 'STOKES':
                line = line[:line.index(words[1])]
                for stoke in stokes_list:
                    if stoke in words[1:]:
                        line = line + ' ' + stoke
                line = line + '\n'
            elif words[0] == 'Scan':
                if len(words) == 7:
                    words = words[0:2] + [''] + words[2:]
                if int(words[1]) not in range(scan_offset, scan_offset + max_scans):
                    continue
                if int(words[1]) in flag_scans:
                    continue
                scan_in_range = True
                keep_scan = False
                if (not keep_scan) and (words[2] in bpcal_names):
                    keep_scan = True
                if (not keep_scan) and (words[2] in target_list):
                    keep_scan = True
                if (not keep_scan) and (len(target_list) == 0):
                    if len(words[2]) == 8:
                        try:
                            dummy = int(words[2][0:4])
                            dummy = int(words[2][5:8])
                        except (ValueError, IndexError):
                            keep_scan = True
                        else:
                            if words[2][4] in ['+', '-']:
                                if keep_all_cals:
                                    keep_scan = True
                            else:
                                keep_scan = True
                    else:
                        keep_scan = True
                if not keep_scan:
                    continue
                if words[2] not in source_list:
                    source_list.append(words[2])
            lsout_file.write(line)
    
    if not scan_in_range:
        os.remove(lsin_file_name)
        os.remove(lsout_file_name)
        return None
    
    # convert to UVFITS and clean up
    if len(source_list) > 0:
        if sys.stdout.name == '<stdout>':
            result = subprocess.run([gvfits, lsout_file_name]).returncode
        else:
            sys.stdout.flush()
            sys.stderr.flush()
            with open(sys.stdout.name, 'a') as logfile:
                result = subprocess.run([gvfits, lsout_file_name], stdout=logfile, stderr=subprocess.STDOUT).returncode
            sys.stdout.flush()
            sys.stderr.flush()
        if result != 0:
            raise Exception('LTA to UVFITS conversion failed for %s' % (lta_file_name))
        shutil.move(file_name, uvfits_file_name)
        if os.path.isfile('gvfits.log'):
            os.remove('gvfits.log')
    
    if os.path.isfile(lsin_file_name):
        os.remove(lsin_file_name)
    if os.path.isfile(lsout_file_name):
        os.remove(lsout_file_name)
    if os.path.isfile(plan_file_name):
        os.remove(plan_file_name)
    
    # cleanup copied libraries

    Path(f'listscan-{version}').unlink(missing_ok=True)
    Path(f'gvfits-{version}').unlink(missing_ok=True)
    shutil.rmtree('lib', ignore_errors=True)
    #for lib in libsgmrt:
    #    libname = os.path.basename(lib)
    #    if os.path.isfile(libname):
    #        os.remove(libname)

    # remove lib from LD_LIBRARY_PATH
    # export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SPAM_PATH}/lib
    if 'LD_LIBRARY_PATH' in os.environ:
        paths = os.environ['LD_LIBRARY_PATH'].split(':')
        new_paths = []
        for path in paths:
            if path != os.path.abspath('lib'):
                new_paths.append(path)
        os.environ['LD_LIBRARY_PATH'] = ':'.join(new_paths)


    return file_name, source_list
