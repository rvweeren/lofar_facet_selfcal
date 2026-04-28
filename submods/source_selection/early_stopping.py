import os
import logging
import pandas as pd
import re

try:
    from .selfcal_selection import get_images_solutions, main as quality_check
    from .image_score import get_nn_model, predict_nn
except ImportError:
    from selfcal_selection import get_images_solutions, main as quality_check
    from image_score import get_nn_model, predict_nn

logger = logging.getLogger(__name__)


def _parse_source_id(inp_str: str = None):
    """Try parsing ILTJhhmmss.ss±ddmmss.s source_id string"""

    try:
        parsed_inp = re.findall(r'ILTJ\d{6}\.\d{2}[+\-]\d{6}\.\d{1}', inp_str)[0]
    except IndexError:
        print(f"WARNING: {inp_str} does not contain a valid source ID")
        parsed_inp = ''

    return parsed_inp


def _initialize_nn_model(nn_model_cache: str, skip_neural_network: bool) -> None:
    """Initialize the neural network model on the first cycle."""

    global nn_model

    if skip_neural_network:
        nn_model = None
        return

    try:
        nn_model = get_nn_model(cache=nn_model_cache)
    except (ImportError, SystemExit):
        logger.info(
            "WARNING: Issues with downloading/getting Neural Network model.. Skipping and continue without."
            "\nMost likely due to issues with accessing cortExchange or no internet access."
        )
        nn_model = None


def _get_predict_score(images: dict,
                       cycle: int) -> float:
    """Return the neural network prediction score, or 1.0 if no model is available."""
    if nn_model is None:
        return 1.0
    score = predict_nn(images[cycle], nn_model)
    logger.info(f"Neural network score: {score}")
    return score


def _has_converged(df: pd.DataFrame,
                   cycle: int,
                   predict_score: float,
                   rms_ratio: float,
                   minmax_ratio: float) -> bool:
    """Return True if selfcal has converged based on quality metrics."""
    phase = df['phase'][cycle]
    return (
        (predict_score < 0.50 and phase < 0.10 and rms_ratio < 1.00 and minmax_ratio < 0.85) or
        (predict_score < 0.50 and phase < 0.20 and rms_ratio < 0.90 and minmax_ratio < 0.50) or
        (predict_score < 0.50 and phase < 0.30 and rms_ratio < 0.95 and minmax_ratio < 0.30) or
        (predict_score < 0.40 and phase < 0.50 and rms_ratio < 1.00 and minmax_ratio < 1.00) or
        (predict_score < 0.30 and phase < 0.1) or
        (phase < 0.003) or
        (phase < 0.10 and rms_ratio < 0.50 and minmax_ratio < 0.10 and predict_score == 1.0) or
        (phase < 0.05 and minmax_ratio < 0.10 and rms_ratio < 0.50 and predict_score == 1.0)
    )


def _has_diverged(df: pd.DataFrame,
                  cycle: int,
                  rms_ratio: float,
                  minmax_ratio: float) -> bool:
    """Return True if selfcal has started to diverge."""
    phase = df['phase']
    rms = df['rms']
    minmax = df['min/max']
    c = cycle
    return (
        (rms[c-3] < rms[c] and rms[c-2] < rms[c] and rms[c-1] < rms[c] and
         minmax[c-3] < minmax[c] and minmax[c-2] < minmax[c] and minmax[c-1] < minmax[c]) or
        (minmax_ratio > 1.0 and rms_ratio > 1.0) or
        (phase[c-3] < phase[c] and phase[c-2] < phase[c] and phase[c-1] < phase[c])
    )


def _save_best(mergedh5: dict,
               images: dict,
               cycle: int,
               source_id: str) -> None:
    """Copy the current cycle's solutions and image as the best outputs."""
    src_h5 = mergedh5[cycle]
    dst_h5 = f'h5_solutions/best_{source_id}solutions.h5'
    best_image = f'best_{images[cycle].split("/")[-1]}'
    logger.info(f'{src_h5} --> {dst_h5}')
    os.system(f'cp {src_h5} {dst_h5}')
    os.system(f'cp {images[cycle]} {best_image}')


def _log_best_cycle(df: pd.DataFrame) -> None:
    """Log which cycle produced the best image and solutions."""
    logger.info(f"Best image: Cycle {max(df['min/max'].argmin(), df['rms'].argmin())}")
    logger.info(f"Best solutions: Cycle {df['phase'].argmin()}")


def early_stopping(station: str = 'international',
                   cycle: int = None,
                   start_cycle: int = None,
                   end_cycle: int = None,
                   nn_model_cache: str = None,
                   skip_neural_network: bool = None,
                   images: list = None,
                   mergedh5: list = None) -> bool:
    """
    Determine early-stopping based on Neural Network image validation and image-based metrics.

    :param station: 'international' or dutch stations
    :param cycle: cycle number
    :param start_cycle: start cycle
    :param end_cycle: end cycle
    :param nn_model_cache: neural network model cache
    :param skip_neural_network: skip use of neural network model
    :return: True to stop, False to continue
    """

    if cycle == start_cycle:
        _initialize_nn_model(nn_model_cache, skip_neural_network)

    if cycle <= 3:
        return False

    if not images:
        logger.info("WARNING: Issues with finding images for early-stopping. Skipping and continue without...")
        return False

    qualitymetrics = quality_check(mergedh5, images, station)
    df = pd.read_csv(f"./selfcal_quality_plots/selfcal_performance_{qualitymetrics[0]}.csv")

    rms_ratio = df['rms'][cycle]/df['rms'][0]
    minmax_ratio = df['min/max'][cycle]/df['min/max'][0]
    # Guard against NaN (recompute is a no-op, but preserves original behaviour)
    if minmax_ratio != minmax_ratio:
        minmax_ratio = df['min/max'][cycle]/df['min/max'][0]
        rms_ratio = df['rms'][cycle]/df['rms'][0]

    predict_score = _get_predict_score(images, cycle)
    source_id = _parse_source_id(mergedh5[cycle]) + "_"

    _log_best_cycle(df)

    if _has_converged(df, cycle, predict_score, rms_ratio, minmax_ratio):
        logger.info(f"Early-stopping at cycle {cycle}, because selfcal converged")
        _save_best(mergedh5, images, cycle, source_id)
        return True

    if _has_diverged(df, cycle, rms_ratio, minmax_ratio):
        logger.info(f"Early-stopping at cycle {cycle}, because selfcal starts to diverge...")
        _save_best(mergedh5, images, cycle, source_id)
        return True

    logger.info(f"No early-stopping at cycle {cycle}")
    if cycle == end_cycle - 1:
        _save_best(mergedh5, images, cycle, source_id)

    return False