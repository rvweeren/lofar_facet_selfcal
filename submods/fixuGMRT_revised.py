#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 - Francesco de Gasperin
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

"""
Francesco de Gasperin
based on work by Martijn Oei

Adapt the MS format of uGMRT data to one usable by LOFAR software.
Memory-optimized version to handle large datasets without overloading memory.
"""

import os, sys, logging, time
import numpy as np
from casacore import tables
import gc


class MS(object):

    def __init__(self, ms_file, chunk_size=1000):
        logging.info("Starting work on MS at '" + ms_file + "'...")

        self.ms_file = ms_file
        self.chunk_size = chunk_size
        self.t       = tables.table(ms_file,                      readonly = False, ack = False)
        self.tpol    = tables.table(ms_file + "/POLARIZATION",    readonly = False, ack = False)
        self.tspect  = tables.table(ms_file + "/SPECTRAL_WINDOW", readonly = False, ack = False)


    def close(self):
        '''
        Close tables opened in '__init__'.
        '''
        logging.info("Closing tables...")

        self.t.close()
        self.tpol.close()
        self.tspect.close()


    def columnExists(self, columnName):
        '''
        Check whether a column with name 'columnName' exists.
        '''
        columnNames = self.t.colnames()

        return (columnName in columnNames)


    def two_pol_ms(self):
        '''
        Check whether the MS has two polarizations (XX and YY or RR and LL) only.
        '''
        corr_type = self.tpol.getcol("CORR_TYPE")

        if np.size(corr_type) == 2:
            logging.info("MS has two polarizations.")
            return True
        logging.info("MS has {} polarizations.".format(np.size(corr_type)))
        return False


    def removeColumns(self):
        '''
        Remove columns that are never used by the LOFAR software, and are thus a waste of disk space (e.g. "SIGMA_SPECTRUM"),
        or that are generated (in the right shape) later in the pipeline (e.g. "MODEL_DATA").
        Note: removal of columns can give errors when executing LOFAR command 'msoverview'.
        '''
        logging.info("- Removal of unnecessary data columns -")

        for columnName in ["SIGMA_SPECTRUM", "MODEL_DATA"]: # This list could possibly be expanded.
            if (self.columnExists(columnName)):
                self.t.removecols(columnName)


    def updatePolarisation(self):
        '''
        Make sure that the MS contains 4 polarisations.
        "CORR_TYPE"    column description comment: 'The polarization type for each correlation product, as a Stokes enum.'
        "CORR_PRODUCT" column description comment: 'Indices describing receptors of feed going into correlation'
        '''
        logging.info("- Adaptation of polarisation metadata -")

        correlationTypesNew      = np.array([[5, 6, 7, 8]])
        correlationProductsNew   = np.array([[[0, 0], [0, 1], [1, 0], [1, 1]]])
        numberOfCorrelationsNew  = 4

        self.tpol.putcol("CORR_TYPE",    correlationTypesNew)
        self.tpol.putcol("CORR_PRODUCT", correlationProductsNew)
        self.tpol.putcol("NUM_CORR",     numberOfCorrelationsNew)


    def updateFreqMetadata(self):
        '''
        Flip the frequency channel order to match the LOFAR convention of ascending frequencies.
        We first check whether the flip is necessary at all, to make the code robust to running this program twice on the same MS.
        '''
        frequencies = self.tspect.getcol("CHAN_FREQ")

        if (frequencies[0, 0] > frequencies[0, -1]):
            logging.info("- Adaptation of frequency metadata -")
            self.tspect.putcol("CHAN_FREQ", np.fliplr(frequencies))
            return True
        else:
            logging.info("Frequency order already correct.")
            return False


    def updateFieldMetadata(self):
        '''
        MSs that were originally multi-field, have been split up in single-field MSs.
        Adapt the field metadata accordingly.
        '''
        logging.info("- Adaptation of field information -")

        pathMS      = self.ms_file
        pathMSField = self.ms_file + "/FIELD"

        # Remove metadata of other fields in the FIELD subtable.
        tables.taql("delete from $pathMSField where rownr() not in (select distinct FIELD_ID from $pathMS)")

        # Set 'SOURCE_ID' to 0 in the FIELD subtable.
        tables.taql("update $pathMSField set SOURCE_ID=0")

        # Set 'FIELD_ID' to 0 in the main table.
        tables.taql("update $pathMS set FIELD_ID=0")

    def fix_crosshand(self):
        '''
        set the XY/RL and YX/LR correlations to XX/RR and YY/LL values to avoid zeros in crosshand pols
        '''
    
        logging.info("- Adaptation of cross-hand polarisation visibility data -")
        logging.info("Fixing crosshand polarizations in MS %s", self.ms_file)
        if not self.columnExists("DATA"):
            logging.warning("DATA column not found, skipping.")
            return

        # Check shape using a single-row sample
        sample = self.t.getcol("DATA", startrow=0, nrow=1)
        if sample.shape[-1] != 4:
            logging.warning("Expected 4 pols in DATA, found %d. Skipping.", sample.shape[-1])
            return

        total_rows = self.t.nrows()
        for start_row in range(0, total_rows, self.chunk_size):
            nrow = min(self.chunk_size, total_rows - start_row)
            chunk = self.t.getcol("DATA", startrow=start_row, nrow=nrow)
            # chunk shape: (nrow, nchan, 4)
            chunk[:, :, 1] = (chunk[:, :, 0] - chunk[:, :, 3])  # XY/RL = XX/RR - YY/LL (Stokes Q/V)
            chunk[:, :, 2] = (chunk[:, :, 0] - chunk[:, :, 3])  # YX/LR = XX/RR - YY/LL (Stokes Q/V)
            self.t.putcol("DATA", chunk, startrow=start_row, nrow=nrow)
            del chunk
            gc.collect()


    def updateIntervals(self):
        '''
        Update the INTERVAL and TIME columns,
        so that differences between time stamps are always integer multiples of a fixed constant.
        '''
        logging.info("- Adaptation of time intervals -")

        # Calculate the new interval.
        pathMS          = self.ms_file
        times           = (tables.taql("select distinct TIME from $pathMS")).getcol("TIME")
        intervals       = times[1 : ] - times[ : -1]
        intervals       = intervals[intervals < 1.5 * np.min(intervals)] # Select only intervals that do not correspond with big jumps in time.
        intervalPrecise = np.mean(intervals)

        # Update INTERVAL column.
        intervalsOld    = self.t.getcol("INTERVAL")
        intervalsNew    = np.ones_like(intervalsOld) * intervalPrecise
        self.t.putcol("INTERVAL", intervalsNew)

        # Update TIME column.
        timesOld        = self.t.getcol("TIME")
        timesNew        = timesOld[0] + np.round((timesOld - timesOld[0]) / intervalPrecise, 0) * intervalPrecise
        self.t.putcol("TIME", timesNew)

        logging.info("Interval set to %f" % intervalPrecise)
        logging.debug("Time intervals (old; possibly unequal):")
        logging.debug(np.unique(timesOld)[1 : ] - np.unique(timesOld)[ : -1])
        logging.debug("Time intervals (new; should be equal):")
        logging.debug(np.unique(timesNew)[1 : ] - np.unique(timesNew)[ : -1])


    def _get_data_shape_and_info(self, column_name):
        '''
        Get shape information and metadata for a column without loading all data.
        '''
        # Get a small sample to determine shape
        sample_data = self.t.getcol(column_name, startrow=0, nrow=1)
        total_rows = self.t.nrows()
        
        keywordNames = self.t.colkeywordnames(column_name)
        columnDescription = self.t.getcoldesc(column_name)
        dataManagerInfo = self.t.getdminfo(column_name)
        
        return sample_data.shape, total_rows, keywordNames, columnDescription, dataManagerInfo


    def _process_data_chunk(self, data_chunk, updateFreq, expand_polarizations=True):
        '''
        Process a chunk of data (flip frequencies and expand polarizations if needed).
        '''
        if updateFreq:
            data_chunk = data_chunk[:, ::-1, :]
        
        if expand_polarizations and data_chunk.shape[2] == 2:
            # Expand from 2 to 4 polarizations
            if data_chunk.dtype == np.complex128 or data_chunk.dtype == np.complex64:
                # For visibility data
                new_chunk = np.zeros((data_chunk.shape[0], data_chunk.shape[1], 4), dtype=data_chunk.dtype)
                new_chunk[:, :, 0] = data_chunk[:, :, 0]  # XX -> XX
                new_chunk[:, :, 3] = data_chunk[:, :, 1]  # YY -> YY
                # XY and YX remain zero
            elif data_chunk.dtype == np.bool_ or data_chunk.dtype == bool:
                # For flag data
                new_chunk = np.zeros((data_chunk.shape[0], data_chunk.shape[1], 4), dtype=data_chunk.dtype)
                new_chunk[:, :, 0] = data_chunk[:, :, 0]  # XX flags
                new_chunk[:, :, 1] = data_chunk[:, :, 0]  # XY flags (copy from XX)
                new_chunk[:, :, 2] = data_chunk[:, :, 0]  # YX flags (copy from XX)
                new_chunk[:, :, 3] = data_chunk[:, :, 1]  # YY flags
            else:
                # For weight data
                new_chunk = np.zeros((data_chunk.shape[0], data_chunk.shape[1], 4), dtype=data_chunk.dtype)
                new_chunk[:, :, 0] = data_chunk[:, :, 0]  # XX weights
                new_chunk[:, :, 3] = data_chunk[:, :, 1]  # YY weights
                # set XY and YX weights (if zero DP3 will flag all visibilities when averaging)
                new_chunk[:, :, 1] = data_chunk[:, :, 0]  # XY weights
                new_chunk[:, :, 2] = data_chunk[:, :, 1]  # YX weights
            
            return new_chunk
        
        return data_chunk


    def _create_new_column(self, column_name, sample_shape, total_rows, columnDescription, dataManagerInfo, manager_suffix):
        '''
        Create a new column with updated data manager info.
        '''
        # Update data manager info
        dataManagerInfo["NAME"] = f"Tiled{column_name.title()}{manager_suffix}"
        
        # Calculate appropriate tile shapes based on data dimensions
        if len(sample_shape) == 3:  # (time, freq, pol)
            n_pol = 4 if sample_shape[2] == 2 else sample_shape[2]
            n_freq = sample_shape[1]
            
            # Optimize tile shape for chunked processing
            tile_time = min(self.chunk_size, total_rows)
            tile_freq = min(40, n_freq)
            
            dataManagerInfo["SPEC"]["DEFAULTTILESHAPE"] = np.array([n_pol, tile_freq, tile_time], dtype=np.int32)
            dataManagerInfo["SPEC"]["HYPERCUBES"]["*1"]["TileShape"] = np.array([n_pol, tile_freq, tile_time], dtype=np.int32)
            dataManagerInfo["SPEC"]["HYPERCUBES"]["*1"]["CubeShape"] = np.array([n_pol, n_freq, total_rows], dtype=np.int32)
            dataManagerInfo["SPEC"]["HYPERCUBES"]["*1"]["CellShape"] = np.array([n_pol, n_freq], dtype=np.int32)
        
        # Remove old column if it exists
        if self.columnExists(column_name):
            self.t.removecols(column_name)
        
        # Add new column
        self.t.addcols(tables.makecoldesc(column_name, columnDescription), dataManagerInfo)


    def updateColumns(self, updateFreq):
        '''
        Update DATA, FLAG and WEIGHT_SPECTRUM columns using chunked processing to avoid memory overload.
        '''
        logging.info("- Change existing (or create alternative) columns for data, flags and weights -")
        
        # Process each column separately to minimize memory usage
        for column_name in ["DATA", "FLAG", "WEIGHT_SPECTRUM"]:
            if not self.columnExists(column_name):
                logging.warning(f"Column {column_name} does not exist, skipping...")
                continue
                
            logging.info(f"Processing column '{column_name}'...")
            
            # Get metadata without loading all data
            sample_shape, total_rows, keywordNames, columnDescription, dataManagerInfo = self._get_data_shape_and_info(column_name)
            
            # Determine manager suffix
            manager_suffix = {"DATA": "Martijn", "FLAG": "Martijn", "WEIGHT_SPECTRUM": "Martijn"}.get(column_name, "Martijn")
            
            # Create temporary column to store processed data
            temp_column_name = f"{column_name}_TEMP"
            
            # Update column description for new polarization structure if needed
            if sample_shape[2] == 2:
                new_sample_shape = (sample_shape[0], sample_shape[1], 4)
            else:
                new_sample_shape = sample_shape
            
            # Create new column
            self._create_new_column(temp_column_name, new_sample_shape, total_rows, columnDescription, dataManagerInfo, manager_suffix)
            
            # Process data in chunks
            logging.info(f"Processing {total_rows} rows in chunks of {self.chunk_size}...")
            
            for start_row in range(0, total_rows, self.chunk_size):
                end_row = min(start_row + self.chunk_size, total_rows)
                current_chunk_size = end_row - start_row
                
                if start_row % (self.chunk_size * 10) == 0:  # Progress update every 10 chunks
                    logging.info(f"Processing rows {start_row} to {end_row-1} ({100*end_row/total_rows:.1f}%)")
                
                # Load chunk
                data_chunk = self.t.getcol(column_name, startrow=start_row, nrow=current_chunk_size)
                
                # Process chunk
                processed_chunk = self._process_data_chunk(data_chunk, updateFreq, expand_polarizations=True)
                
                # Write processed chunk to temporary column
                self.t.putcol(temp_column_name, processed_chunk, startrow=start_row, nrow=current_chunk_size)
                
                # Force garbage collection to free memory
                del data_chunk, processed_chunk
                gc.collect()
            
            # Remove original column and rename temporary column
            logging.info(f"Finalizing column '{column_name}'...")
            self.t.removecols(column_name)
            
            # Rename temporary column to original name
            # Note: Direct renaming is not supported in casacore, so we need to recreate
            self._create_new_column(column_name, new_sample_shape, total_rows, columnDescription, dataManagerInfo, manager_suffix)
            
            # Copy data from temp column to final column in chunks
            for start_row in range(0, total_rows, self.chunk_size):
                end_row = min(start_row + self.chunk_size, total_rows)
                current_chunk_size = end_row - start_row
                
                temp_data = self.t.getcol(temp_column_name, startrow=start_row, nrow=current_chunk_size)
                self.t.putcol(column_name, temp_data, startrow=start_row, nrow=current_chunk_size)
                
                del temp_data
                gc.collect()
            
            # Remove temporary column
            self.t.removecols(temp_column_name)
            
            logging.info(f"Column '{column_name}' processing complete.")
            
            # Force garbage collection between columns
            gc.collect()


def readArguments():
    import argparse
    parser = argparse.ArgumentParser("Adapt uGMRT MS to LOFAR MS format.")
    parser.add_argument("-v", "--verbose", help="Be verbose. Default is False", required=False, action="store_true")
    parser.add_argument("-c", "--chunk-size", help="Number of rows to process at once (default: 1000)", 
                       type=int, default=1000, required=False)
    parser.add_argument("ms_files", help="MeasurementSet name(s).", type=str, nargs="+")
    args = parser.parse_args()
    return vars(args)


if (__name__ == "__main__"):
    start_time = time.time()

    args = readArguments()
    verbose = args["verbose"]
    chunk_size = args["chunk_size"]
    ms_files = args["ms_files"]

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.info(f'Reading MSs with chunk size: {chunk_size}...')
    MSs = []
    for ms_file in ms_files:
        MSs.append(MS(ms_file, chunk_size=chunk_size))

    for MS in MSs:
        twopol = MS.two_pol_ms()
        MS.removeColumns()
        MS.updatePolarisation()
        updateFreq = MS.updateFreqMetadata()
        MS.updateFieldMetadata()
        MS.updateIntervals()
        MS.updateColumns(updateFreq)
        if twopol:
            MS.fix_crosshand()
        MS.close()

    logging.debug('Running time %.0f s' % (time.time() - start_time))
    logging.info('Done.')
