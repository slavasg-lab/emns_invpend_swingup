# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # RSS uses the times package; "Times" is the standard target
    "font.serif": ["Times"], 
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    # This prevents the labels from being cut off
    "figure.autolayout": False, 
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{times}"
})