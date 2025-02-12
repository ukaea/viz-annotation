"""
    Written by Niraj Bhujel UKRI-STFC
    Slightly adapted for Flake8 by Nicholas Whyatt UKRI-STFC,
    on behalf of CCP-EM
"""

import sys
import logging


def create_logger(log_file_path, level=logging.INFO) -> logging.Logger:
    """
    Preconfigured manual logger.
    """

    rootLogger = logging.getLogger("")
    rootLogger.handlers.clear()

    # NOTE! Settting console handler to DEBUG causes logger.info
    # to print twice in console -> bug: creater_logger is called twice,
    # and handlers were added. Clear handler before adding new.
    consoleFormat = logging.Formatter("%(message)s")
    consoleHandler = logging.StreamHandler(sys.stdout)
    # consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(consoleFormat)
    rootLogger.addHandler(consoleHandler)

    fileFormat = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fileHandler = logging.FileHandler(log_file_path)
    # fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(fileFormat)
    rootLogger.addHandler(fileHandler)

    return rootLogger
