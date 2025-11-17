#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def init_logger(log_file_name='app.log', level=logging.INFO):
    log_file_path = os.path.join(ROOT, log_file_name)

    logger = logging.getLogger()
    logger.setLevel(level)

    """Already have a handler registered"""
    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [Module: %(module)s.py, Line number: %(lineno)d] - %(message)s' ,
                    datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = init_logger()

if __name__ == "__main__":
    pass
