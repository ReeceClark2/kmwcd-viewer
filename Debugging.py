import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, TextBox, Button
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import subprocess
from PIL import Image
import KMALL
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import warnings
import os
import time
import sys
import threading

# Define the KMALL file path
kmall_file = "data/sample.kmwcd"
# kmall_file = "data/20240521_051222_sentry.kmwcd"

# # Initialize KMALL object and index file
K = KMALL.kmall(kmall_file)
K.index_file()

# # Print report on packet types (optional, for debugging)
K.report_packet_types()

# # Extract MWC messages and their count
IMWC = K.Index["MessageType"] == "#MWC"
MWCIndex = K.Index[IMWC]
max_pings = len(MWCIndex)

last_dot_index = kmall_file.rfind('.')
name_for = kmall_file[:last_dot_index]


def pipeline(beamdata, dg):
    print(dg["rxInfo"])
    # Extract necessary data from KMALL data structure
    beamAmp = pd.DataFrame.from_dict(dg['beamData']['sampleAmplitude05dB_p'])
    numsamp = dg['beamData']['numSampleData']
    SoundSp = dg["rxInfo"]["soundVelocity_mPerSec"]
    SampleFreq = dg["rxInfo"]["sampleFreq_Hz"]
    TVGFuncApplied = dg["rxInfo"]["TVGfunctionApplied"]
    TVGOffset = dg["rxInfo"]["TVGoffset_dB"]
    txBeamWidth = dg['sectorData']['txBeamWidthAlong_deg']
    beamAngle = np.array(dg['beamData']['beamPointAngReVertical_deg'])
    # beamAngle = np.array(beamdata["beamPointAngReVertical_deg"])
    print('Beam Amplitudes:' + str(beamAmp),
            '\n\nNums Amplitudes:' + str(numsamp),
            '\n\nSound Speed:' + str(SoundSp), 
            '\n\nSample Frequency:' + str(SampleFreq),
            '\n\nTVG Function Applied:' + str(TVGFuncApplied),
            '\n\nTVG Offset:' + str(TVGOffset),
            '\n\ntx Beam Width:' + str(txBeamWidth),
            '\n\nBeam Angles:' + str(beamAngle))

    length = np.arange(1, len(beamAmp.columns)).tolist()
    rang = [x * .5 * SoundSp / SampleFreq for x in length]
    rang.insert(0, 10e-9)

    rng = np.tile(rang, ((beamAmp).shape[0], 1))
    rnge = pd.DataFrame(rng).T

    za = -(np.cos(beamAngle * np.pi / 180) * rnge)
    ya = (np.sin(beamAngle * np.pi / 180) * rnge)

    Awc = beamAmp / 2
    X = TVGFuncApplied
    C = TVGOffset
    RTval = 10 * np.log10((np.pi / 180) * (txBeamWidth[0] * np.pi / 180))

    TS = (Awc + RTval * np.ones(Awc.shape)).T - (float(X) * np.log10(np.ones((len(rnge), 1)) * rnge)) + (40 * np.log10(np.ones((len(Awc.T), 1)) * rnge)) - float(C)

    recieveAngle = 1 / np.cos(beamAngle * np.pi / 180)
    RxRad = 1 * np.pi / 180
    langth = 2 * rnge * np.sin(RxRad / 2)
    TxRad = txBeamWidth[0] * np.pi / 180 * recieveAngle
    width = ((2 * np.ones([rnge.shape[0], len(TxRad)]) * rnge) * np.sin(TxRad / 2)).T
    beamArea = langth * width.T

    Tau = 3500 / 1e5
    Sv = np.zeros(TS.size)

    Vol_log = 10 * np.log(beamArea * Tau * SoundSp / 2)
    Sv = TS - Vol_log

    return Sv, ya, za


def pingSingle(ping):
    dg = lambda ping: K.read_index_row(MWCIndex.iloc[ping])

    df = dg(ping)

    if df["header"]["dgmType"] == b"#MWC":
        beamdata = pd.DataFrame.from_dict(df["beamData"])

    Sv1, ya1, za1 = pipeline(beamdata, df)

    return Sv1, ya1, za1


sv, y, z = pingSingle(4)
# print(sv, y, z)

print(y, z)
plt.pcolor(y, z, sv, vmin=-120, vmax=-30, cmap='inferno')
plt.show()