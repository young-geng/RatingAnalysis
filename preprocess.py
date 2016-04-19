"""
Preprocess the raw data to remove month seperation columns and filter out
unused columns.

Usage: python preprocess.py <input filename> <output filename>
"""

from __future__ import division

import numpy as np
import scipy
import pandas as pd
from sys import argv


if __name__ == '__main__':
    assert len(argv) == 3

    months = (
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    )

    data = pd.read_csv(argv[1])
    data = data[~data["PTS"].isin(months)].rename(
        columns = {
            "Visitor/Neutral": "Visitor",
            "Home/Neutral": "Home",
            "PTS": "VisitorPTS",
            "PTS.1": "HomePTS",
            "Start (ET)": "Start"
        }
    )[["Date", "Start", "Visitor", "VisitorPTS", "Home", "HomePTS"]]

    data.to_csv(argv[2])
