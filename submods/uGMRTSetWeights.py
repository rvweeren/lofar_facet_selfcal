"""
Martijn S.S.L. Oei and Reinout J. van Weeren, January 12026 H.E.

Update the weights of a (uGMRT) Measurement Set (MS).
This code serves as a faster alternative to the CASA task 'statwt'.
"""
import argparse, gc, logging, numpy, time
from casacore.tables import table
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import zoom


def gridMeans2D(grid, numberOfSamplesDim1, numberOfSamplesDim2):
    """
    Produce a 2D array with means shaped identically to 'grid'.
    The means are calculated using samples of size 'numberOfSamplesDim1' by 'numberOfSamplesDim2'.
    """
    # Determine the padded grid's size.
    numberOfValuesDim1, numberOfValuesDim2 = grid.shape
    numberOfValuesDim1Pad = numpy.ceil(numberOfValuesDim1 / numberOfSamplesDim1).astype(int) * numberOfSamplesDim1
    numberOfValuesDim2Pad = numpy.ceil(numberOfValuesDim2 / numberOfSamplesDim2).astype(int) * numberOfSamplesDim2

    # Create the padded grid.
    gridPadded            = numpy.full((numberOfValuesDim1Pad, numberOfValuesDim2Pad), numpy.nan)
    gridPadded[ : numberOfValuesDim1, : numberOfValuesDim2] = grid

    # Calculate means. Then create a padded mean grid, and finally a mean grid shaped as 'grid'.
    means                 = numpy.nanmean(gridPadded.reshape(numberOfValuesDim1Pad // numberOfSamplesDim1, numberOfSamplesDim1, numberOfValuesDim2Pad // numberOfSamplesDim2, numberOfSamplesDim2), axis = (1, 3))
    gridMeansPadded       = numpy.repeat(numpy.repeat(means, numberOfSamplesDim1, axis = 0), numberOfSamplesDim2, axis = 1)
    gridMeans             = gridMeansPadded[ : numberOfValuesDim1, : numberOfValuesDim2]
    gridMeans[numpy.isnan(grid)] = numpy.nan

    return gridMeans


def gridVariances2D(grid, numberOfSamplesDim1, numberOfSamplesDim2, returnNumbersOfSamples = True, interpolate = False):
    """
    Produce a 2D array with variances shaped identically to 'grid'.
    The variances are calculated using samples of size 'numberOfSamplesDim1' by 'numberOfSamplesDim2'.
    """
    # Determine the padded grid's size.
    numberOfValuesDim1, numberOfValuesDim2 = grid.shape
    numberOfValuesDim1Pad = numpy.ceil(numberOfValuesDim1 / numberOfSamplesDim1).astype(int) * numberOfSamplesDim1
    numberOfValuesDim2Pad = numpy.ceil(numberOfValuesDim2 / numberOfSamplesDim2).astype(int) * numberOfSamplesDim2

    # Create the padded grid.
    gridPadded            = numpy.full((numberOfValuesDim1Pad, numberOfValuesDim2Pad), numpy.nan)
    gridPadded[ : numberOfValuesDim1, : numberOfValuesDim2] = grid

    # Calculate variances. Then create a padded variance grid, and finally a variance grid shaped as 'grid'.
    variances             = numpy.nanvar(gridPadded.reshape(numberOfValuesDim1Pad // numberOfSamplesDim1, numberOfSamplesDim1, numberOfValuesDim2Pad // numberOfSamplesDim2, numberOfSamplesDim2), axis = (1, 3), ddof = 1)
    if (interpolate):
        gridVariancesPadded = zoom(numpy.nan_to_num(variances, nan = numpy.nanmedian(variances)), zoom = (numberOfSamplesDim1, numberOfSamplesDim2), order = 1, grid_mode = True, mode = "nearest")
    else:
        gridVariancesPadded = numpy.repeat(numpy.repeat(variances, numberOfSamplesDim1, axis = 0), numberOfSamplesDim2, axis = 1)
    gridVariances         = gridVariancesPadded[ : numberOfValuesDim1, : numberOfValuesDim2]
    gridVariances[numpy.isnan(grid)] = numpy.nan

    # Returning 'numbersOfSamples' is useful to gauge how reliable each variance is.
    if (returnNumbersOfSamples):
        # Calculate the number of non-NaN visibilities used in each sample variance.
        numbersOfSamples           = numpy.sum(~numpy.isnan(gridPadded.reshape(numberOfValuesDim1Pad // numberOfSamplesDim1, numberOfSamplesDim1, numberOfValuesDim2Pad // numberOfSamplesDim2, numberOfSamplesDim2)), axis = (1, 3))
        gridNumbersOfSamplesPadded = numpy.repeat(numpy.repeat(numbersOfSamples, numberOfSamplesDim1, axis = 0), numberOfSamplesDim2, axis = 1)
        gridNumbersOfSamples       = gridNumbersOfSamplesPadded[ : numberOfValuesDim1, : numberOfValuesDim2]

        return (gridVariances, gridNumbersOfSamples)
    else:
        return gridVariances


class MSWeightSetter:
    def __init__(self,
                 pathMS,                                 # e.g. "/data1/oei/Ardor_Telae/data/uGMRT_hypergiant/band4Sep/44_101_23sep2023_b4_gwb.ms.hypergiant.copy"
                 dataColumnName             = "RESIDUAL_DATA",
                 numberOfRowsChunk          = 87000,     # in 1; per chunk, load roughly 200 time samples (~10^3 s) for each baseline: 30 * 29 / 2 * 200 = 87000
                 numberOfSamplesTime        = 5,         # in 1
                 numberOfSamplesFreq        = 96,        # in 1
                 numberOfSamplesMinRelative = .1,        # in 1
                 weightMax                  = numpy.inf, # in 1 / Jy²
                 interpolate                = False,     # whether or not to apply linear interpolation in setting the weights
                 removeFringes              = False,
                 numberOfSamplesTimeFringes = 2,         # in 1
                 numberOfSamplesFreqFringes = 16,        # in 1
                 verbose                    = False,     # whether or not to be verbose during calculations
                 plotDirectory              = "",        # e.g. "/data1/oei/Ardor_Telae/data/uGMRT_hypergiant/band4Sep/facetselfcal_from_SPAM_UVFITS/target_DI/uGMRTSetWeightsPlots/"
                 plotFileExtension          = ".png",    # e.g. ".png" or ".pdf"
                 plotDPI                    = 1024,      # particularly useful if 'plotFileExtension' is ".png"
                 plotChunks                 = [0, 1],
                 plotPolarisations          = [0, 3],
                 plotBLs                    = [],
                 plotVisibilityMin          = -.9,       # in Jy
                 plotVisibilityMax          = +.9,       # in Jy
                 plotWeightMin              =  0.,       # in 1 / Jy²
                 plotWeightMax              = 16.,       # in 1 / Jy²; e.g. for an SD of 0.1 Jy, the variance is 0.01 Jy², so that the weight is 100 / Jy²
                 colourMapVisibilities      = "RdYlBu_r",
                 colourMapWeights           = "cividis",
                 colourMapNumbersOfSamples  = "plasma"
                 ):
        self.pathMS                     = pathMS
        self.dataColumnName             = dataColumnName
        self.numberOfRowsChunk          = numberOfRowsChunk
        self.numberOfSamplesTime        = numberOfSamplesTime
        self.numberOfSamplesFreq        = numberOfSamplesFreq
        self.weightMax                  = weightMax
        self.interpolate                = interpolate
        self.verbose                    = verbose
        self.plotDirectory              = plotDirectory
        self.plotFileExtension          = plotFileExtension
        self.plotDPI                    = plotDPI
        self.plotChunks                 = plotChunks
        self.plotPolarisations          = plotPolarisations
        self.plotBLs                    = plotBLs
        self.plotVisibilityMin          = plotVisibilityMin
        self.plotVisibilityMax          = plotVisibilityMax
        self.plotWeightMin              = plotWeightMin
        self.plotWeightMax              = plotWeightMax
        self.colourMapVisibilities      = colourMapVisibilities
        self.colourMapWeights           = colourMapWeights
        self.colourMapNumbersOfSamples  = colourMapNumbersOfSamples
        self.removeFringes              = removeFringes
        self.numberOfSamplesTimeFringes = numberOfSamplesTimeFringes
        self.numberOfSamplesFreqFringes = numberOfSamplesFreqFringes

        self.numberOfSamples            = numberOfSamplesTime * numberOfSamplesFreq
        self.numberOfSamplesMin         = numberOfSamplesMinRelative * self.numberOfSamples


    def setInverseVarianceWeights(self):
        """
        uGMRT weights are typically 1 or 0, and this doesn't make optimal use of the data.
        This function sets the weights of column 'self.dataColumnName' to
        """
        # Load MS.
        MS                        = table(self.pathMS, readonly = False)
        numberOfRows              = MS.nrows() # in 1
        numberOfChunks            = numpy.ceil(numberOfRows / self.numberOfRowsChunk).astype(int) # in 1
        print(f"Starting work on {numberOfChunks} chunks (of {self.numberOfRowsChunk} rows each, except probably for the last chunk)!")


        # Determine whether the MS is dual-polarisation or full-polarisation.
        isDualPolarisation = MS.getcolkeyword("DATA", "FAKE_RLLR")
        if (isDualPolarisation):
            logging.info("This MS is treated as a dual-polarisation data set.")
            self.polarisationNames = ["RR", "LL"]
            polarisationIndices    = [0, 3]
        else:
            logging.info("This MS is treated as a full-polarisation data set.")
            self.polarisationNames = ["RR", "RL", "LR", "LL"]
            polarisationIndices    = [0, 1, 2, 3]


        # Loop over chunks, then over baselines, to update WEIGHT_SPECTRUM.
        for self.indexChunk in range(numberOfChunks):
            print(f"Processing chunk {self.indexChunk + 1} of {numberOfChunks}...")

            # Load chunk.
            indexRowStart            = self.indexChunk * self.numberOfRowsChunk
            numberOfRowsChunkCurrent = min(self.numberOfRowsChunk, numberOfRows - indexRowStart)
            logging.debug(f"Loading column '{self.dataColumnName}'...")
            visibilities             = MS.getcol(self.dataColumnName, startrow = indexRowStart, nrow = numberOfRowsChunkCurrent) # in Jy
            visibilitiesReal         = numpy.real(visibilities)                                                                  # in Jy
            visibilitiesImag         = numpy.imag(visibilities)                                                                  # in Jy
            logging.debug(f"Loading column 'WEIGHT_SPECTRUM'...")
            weightsOld               = MS.getcol("WEIGHT_SPECTRUM",   startrow = indexRowStart, nrow = numberOfRowsChunkCurrent) # in 1 / Jy^2
            antenna1s                = MS.getcol("ANTENNA1",          startrow = indexRowStart, nrow = numberOfRowsChunkCurrent)
            antenna2s                = MS.getcol("ANTENNA2",          startrow = indexRowStart, nrow = numberOfRowsChunkCurrent)

            # Print input data shapes.
            if (self.verbose and self.indexChunk == 0):
                print(self.dataColumnName + "   shape:", visibilities.shape) # e.g. (87000, 3984, 4)
                print("WEIGHT_SPECTRUM shape:", weightsOld.shape)            # e.g. (87000, 3984, 4)
                print("ANTENNA1        shape:", antenna1s.shape)             # e.g. (87000,)
                print("ANTENNA2        shape:", antenna2s.shape)             # e.g. (87000,)
                print("")


            # Create a new weights array.
            weightsNew = numpy.full_like(weightsOld, 0.)


            # Determine 'numberOfAntennae'.
            # 'numpy.unique(antenna1s)' may give: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27]
            # 'numpy.unique(antenna2s)' may give: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28]
            numberOfAntennae = numpy.amax(antenna2s) + 1 # 'antenna2s' goes up higher than 'antenna1s'. Do '+ 1' to convert from the maximum zero-based index to a number.
            logging.debug(f"Identified {numberOfAntennae} antennae in this chunk!")


            # Loop over baselines.
            for self.antenna1Current in range(numberOfAntennae - 1):
                for self.antenna2Current in range(self.antenna1Current + 1, numberOfAntennae):
                    # Load the baseline's visibilities and weights.
                    rowsBL             = (antenna1s == self.antenna1Current) * (antenna2s == self.antenna2Current)
                    numberOfRowsBL     = numpy.sum(rowsBL) # in 1; the number of time steps within the chunk with data for this baseline
                    if (numberOfRowsBL == 0):
                        continue
                    visibilitiesRealBL = visibilitiesReal[rowsBL] # Shape: e.g. (214, 3984, 4) or (215, 3984, 4)
                    visibilitiesImagBL = visibilitiesImag[rowsBL] # Shape: e.g. (214, 3984, 4) or (215, 3984, 4)
                    weightsOldBL       = weightsOld      [rowsBL] # Shape: e.g. (214, 3984, 4) or (215, 3984, 4)

                    # Loop over polarisations specified by 'polarisationIndices'.
                    # This allows us to skip e.g. RL and LR polarisations (and thus save half of the processing time) on a dual-polarisation uGMRT data set.
                    for polarisationIndex, self.polarisationName in zip(polarisationIndices, self.polarisationNames):
                        if (self.verbose):
                            print(f"Processing baseline {self.antenna1Current}–{self.antenna2Current}, polarisation {self.polarisationName}...")

                        # Optionally remove fringes by subtracting the mean of 'self.numberOfSamplesTimeFringes' * 'self.numberOfSamplesFreqFringes' blocks.
                        if (self.removeFringes):
                            meansReal = gridMeans2D(visibilitiesRealBL[:, :, polarisationIndex], self.numberOfSamplesTimeFringes, self.numberOfSamplesFreqFringes)
                            meansImag = gridMeans2D(visibilitiesImagBL[:, :, polarisationIndex], self.numberOfSamplesTimeFringes, self.numberOfSamplesFreqFringes)
                            visibilitiesRealBL[:, :, polarisationIndex] -= meansReal
                            visibilitiesImagBL[:, :, polarisationIndex] -= meansImag

                        # Calculate weights assuming equal time and frequency intervals.
                        variancesReal, numbersOfSamplesReal = gridVariances2D(visibilitiesRealBL[:, :, polarisationIndex], self.numberOfSamplesTime, self.numberOfSamplesFreq, interpolate = self.interpolate)
                        variancesImag, numbersOfSamplesImag = gridVariances2D(visibilitiesImagBL[:, :, polarisationIndex], self.numberOfSamplesTime, self.numberOfSamplesFreq, interpolate = self.interpolate)
                        weightsNewBL               = 1 / (.5 * (variancesReal + variancesImag)) # in 1 / Jy^2

                        # Change unreliable weights to the chunk's median reliable weight.
                        areReliable                = (numbersOfSamplesReal > self.numberOfSamplesMin)
                        areFlagged                 = numpy.isnan(weightsNewBL)
                        weightsNewBLMedianReliable = numpy.nanmedian(weightsNewBL[areReliable])
                        weightsNewBL[~areReliable * ~areFlagged] = weightsNewBLMedianReliable

                        # Limit the weights to maximum value 'weightMax'.
                        weightsNewBL[weightsNewBL > self.weightMax] = self.weightMax

                        # Write baseline's new weights to new weights array.
                        # The MS convention is that weights of flagged data are 0. Thus, we change NaNs --> 0s:
                        numpy.nan_to_num(weightsNewBL, copy = False, nan = 0.)
                        weightsNew[rowsBL, : , polarisationIndex] = weightsNewBL


                        # Create plots.
                        # The following three conditions are sorted in the typical order of likelihood of violation (highest likelihood first),
                        # in the hope this triggers Python's short-circuit evaluation and increases efficiency (if only marginally).
                        if ((self.antenna1Current, self.antenna2Current) in self.plotBLs and self.indexChunk in self.plotChunks and polarisationIndex in self.plotPolarisations):
                            print(f"Creating plots for baseline {self.antenna1Current}–{self.antenna2Current}...")

                            # Plot visibilities (real part).
                            self.plotDynamicSpectrum("VisReal", visibilitiesRealBL[:, :, polarisationIndex].T, self.plotVisibilityMin, self.plotVisibilityMax, self.colourMapVisibilities, "visibility (real part) (Jy)", " | column: " + self.dataColumnName)
                            # Plot visibilities (imaginary part).
                            self.plotDynamicSpectrum("VisImag", visibilitiesImagBL[:, :, polarisationIndex].T, self.plotVisibilityMin, self.plotVisibilityMax, self.colourMapVisibilities, "visibility (imaginary part) (Jy)", " | column: " + self.dataColumnName)
                            # Plot weights (old).
                            self.plotDynamicSpectrum("WeightsOld", weightsOldBL[:, :, polarisationIndex].T, self.plotWeightMin, self.plotWeightMax, self.colourMapWeights, "weight (1 / Jy²)", "")
                            # Plot weights (new).
                            self.plotDynamicSpectrum("WeightsNew", weightsNewBL.T, self.plotWeightMin, self.plotWeightMax, self.colourMapWeights, "weight (1 / Jy²)", " | column: " + self.dataColumnName)
                            # Plot number of samples.
                            self.plotDynamicSpectrum("WeightsNum", numbersOfSamplesReal.T, 0, self.numberOfSamples, self.colourMapNumbersOfSamples, "number of visibilities per sample variance", "")


            # After processing a chunk, we write the weights back.
            print("Homogenising weights across polarisations (as required by Dysco)...")
            weightsNewPolarisationMean = numpy.mean(weightsNew[:, :, numpy.array(polarisationIndices)], axis = 2) # Take the average across RR and LL, or across RR, RL, LR, and LL.
            weightsNew                 = weightsNewPolarisationMean[ : , : , None] * numpy.ones(4)[None, None, : ]


            print("Writing new weights to MS...")
            MS.putcol("WEIGHT_SPECTRUM", weightsNew, startrow = indexRowStart, nrow = numberOfRowsChunkCurrent)
            MS.flush()
            del visibilities, visibilitiesReal, visibilitiesImag, weightsOld, weightsNew
            gc.collect()
        MS.close()
        print("Finished updating weights in '" + self.pathMS + "'!")


    def plotDynamicSpectrum(self, gridLabel, grid, vMin, vMax, colourMap, colourBarLabel, titleAddition):
        """
        Plot a dynamic spectrum with time on the horizontal axis and frequency on the vertical axis, for a specific chunk, baseline, and polarisation.
        """
        plotPath = self.plotDirectory + "BL" + str(self.antenna1Current).zfill(2) + "-" + str(self.antenna2Current).zfill(2) + "Pol" + self.polarisationName + "Chunk" + str(self.indexChunk).zfill(3) + gridLabel + self.plotFileExtension
        print(f"Plotting {plotPath}...")

        fig, ax  = pyplot.subplots(figsize = (10, 8))
        if (self.plotFileExtension == ".pdf"):
            map = ax.pcolormesh(grid, shading="nearest", vmin = vMin, vmax = vMax, antialiased = False, cmap = colourMap)
            map.set_rasterized(False) # This ensures it stays a vector image in the PDF.
        else:
            map = ax.imshow(grid, origin = "lower", interpolation = "nearest", vmin = vMin, vmax = vMax, aspect = "auto", cmap = colourMap)
        divider  = make_axes_locatable(ax)
        cax      = divider.append_axes("right", size="2%", pad=0.05)
        cb       = fig.colorbar(map, cax=cax)
        cb.set_label(colourBarLabel, fontsize = 10)
        ax.set_xlabel("time index", fontsize = 10)
        ax.set_ylabel("frequency index", fontsize = 10)
        ax.set_title(f"baseline: {self.antenna1Current}–{self.antenna2Current} | polarisation: {self.polarisationName} | chunk: {self.indexChunk}" + titleAddition + f"\nMS: {self.pathMS}", fontsize = 11)
        ax.tick_params(axis = "both", which = "major", labelsize = 10)
        pyplot.subplots_adjust(left = .07, right = .93, bottom = .06, top = .93)
        fig.savefig(plotPath, dpi = self.plotDPI)
        pyplot.close(fig)

    '''
    def weightsHistogram(self):
        """
        This function will create a weights 'bandpass' by aggregating, per frequency, weights over time.
        Still under construction.
        """
        # Load MS.
        MS             = table(self.pathMS, readonly = False)
        numberOfRows   = MS.nrows()  # in 1
        numberOfChunks = numpy.ceil(numberOfRows / self.numberOfRowsChunk).astype(int)  # in 1

        # Loop over chunks, then over baselines, to inspect WEIGHT_SPECTRUM.
        for self.indexChunk in range(numberOfChunks):
            print(f"Processing chunk {self.indexChunk + 1} of {numberOfChunks}...")

            # Load chunk.
            indexRowStart            = self.indexChunk * self.numberOfRowsChunk
            numberOfRowsChunkCurrent = min(self.numberOfRowsChunk, numberOfRows - indexRowStart)
            weights                  = MS.getcol("WEIGHT_SPECTRUM",   startrow = indexRowStart, nrow = numberOfRowsChunkCurrent) # in 1 / Jy^2
            antenna1s                = MS.getcol("ANTENNA1",          startrow = indexRowStart, nrow = numberOfRowsChunkCurrent)
            antenna2s                = MS.getcol("ANTENNA2",          startrow = indexRowStart, nrow = numberOfRowsChunkCurrent)

            # Loop over baselines.
            for self.antenna1Current in range(self.numberOfAntennae):
                for self.antenna2Current in range(self.antenna1Current + 1, self.numberOfAntennae):
                    # Load the baseline's weights.
                    rowsBL             = (antenna1s == self.antenna1Current) * (antenna2s == self.antenna2Current)
                    numberOfRowsBL     = numpy.sum(rowsBL) # in 1; the number of time steps within the chunk with data for this baseline
                    if (numberOfRowsBL == 0):
                        continue
                    weightsBL        = weights[rowsBL][0] # Shape: e.g. (214, 3984) or (215, 3984)
                    weightsBLSampled = weightsBL[::self.numberOfSamplesTime, ::self.numberOfSamplesFreq]
    '''


def parseBLs(s):
    import ast
    try:
        val = ast.literal_eval(s)
    except Exception:
        raise argparse.ArgumentTypeError('Must be a Python literal, e.g. "[(0,1),(2,3)]".')
    if not (isinstance(val, list) and all(isinstance(t, tuple) and len(t) == 2 for t in val)):
        raise argparse.ArgumentTypeError("Must be a list of 2-tuples.")
    return val


def readArguments():
    parser = argparse.ArgumentParser("Apply inverse-variance visibility weighting (e.g. to uGMRT data).")
    parser.add_argument("pathsMS",                help = "Measurement Set path(s)",                                                         type = str, nargs = "+")
    parser.add_argument("-d", "--data-column",    help = 'Data column for weight calculation (default: "RESIDUAL_DATA")', required = False, type = str, default = "RESIDUAL_DATA")
    parser.add_argument("-c", "--chunk-size",     help = "Number of rows to process at once (default: 87,000)",           required = False, type = int, default = 87000)
    parser.add_argument("-t", "--sample-size-t",  help = "Number of time steps per sample variance (default: 5)",         required = False, type = int, default = 5)
    parser.add_argument("-f", "--sample-size-f",  help = "Number of frequency steps per sample variance (default: 96)",   required = False, type = int, default = 96)
    parser.add_argument("-r", "--remove-fringes", help = "Activate removal of (RFI-induced) fringes (default: False)",    required = False, action = "store_true") # 'True' if provided.
    parser.add_argument("-v", "--verbose",        help = "Activate verbose output (default: False)",                      required = False, action = "store_true") # 'True' if provided.
    parser.add_argument("-p", "--plot-dir",       help = 'Plotting directory (default: "")',                              required = False, type = str, default = "")
    parser.add_argument("-b", "--plot-BLs",       help = 'List of baselines to plot, e.g. "[(0,1),(2,10)]"',              required = False, type = parseBLs, default = [])
    args   = parser.parse_args()
    return vars(args)


if (__name__ == "__main__"):
    startTime = time.time()
    args      = readArguments()
    pathsMS   = args["pathsMS"]

    if args["verbose"]:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)


    #'''
    #IMPORTANT: Remove these lines for MSs processed with new versions of 'fixuGMRT_revised.py'.
    #'''
    #for pathMS in pathsMS:
    #    MS = table(pathMS, readonly = False)
    #    MS.putcolkeyword("DATA", "FAKE_RLLR", True)
    #    MS.close()


    for pathMS in pathsMS:
        MSWeightSetterCurrent = MSWeightSetter(pathMS, dataColumnName = args["data_column"], numberOfRowsChunk = args["chunk_size"], numberOfSamplesTime = args["sample_size_t"], numberOfSamplesFreq = args["sample_size_f"], removeFringes = args["remove_fringes"], verbose = args["verbose"], plotDirectory = args["plot_dir"], plotBLs = args["plot_BLs"])
        MSWeightSetterCurrent.setInverseVarianceWeights()

    logging.debug("Running time %.0f s" % (time.time() - startTime))
    logging.info("Done.")