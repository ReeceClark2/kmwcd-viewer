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


class kmwcd_viewer:
    def __init__(self, value):
        # Initialize variables and parameters necessary upon instantiation
        self.value = value
        self.MWCIndex = value
        

        # Add other initialization code as needed (e.g., file paths, data structures, etc.)


    def pipeline(self, beamdata, dg):
        # Extract necessary data from KMALL data structure
        beamAmp = pd.DataFrame.from_dict(dg['beamData']['sampleAmplitude05dB_p'])
        numsamp = dg['beamData']['numSampleData']
        SoundSp = dg["rxInfo"]["soundVelocity_mPerSec"]
        SampleFreq = dg["rxInfo"]["sampleFreq_Hz"]
        TVGFuncApplied = dg["rxInfo"]["TVGfunctionApplied"]
        TVGOffset = dg["rxInfo"]["TVGoffset_dB"]
        txBeamWidth = dg['sectorData']['txBeamWidthAlong_deg']
        beamAngle = np.array(beamdata["beamPointAngReVertical_deg"])

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


    def single_ping(self, ping):
        dg = lambda ping: K.read_index_row(self.MWCIndex.iloc[ping-1])

        df = dg(ping)

        if df["header"]["dgmType"] == b"#MWC":
            beamdata = pd.DataFrame.from_dict(df["beamData"])

        Sv, ya, za = self.pipeline(beamdata, df)

        return Sv, ya, za


    def multiple_ping(self, start_ping, end_ping):
        # Determine the number of pings
        num_pings = end_ping - start_ping + 1

        # Initialize preallocated arrays with None or zeros if sizes are known
        Sv_list = [None] * num_pings
        ya_list = [None] * num_pings
        za_list = [None] * num_pings

        for ind, i in enumerate(range(start_ping, end_ping + 1)):
            dg = lambda ping: K.read_index_row(self.MWCIndex.iloc[i-1])
            df = dg(i)

            if df["header"]["dgmType"] == b"#MWC":
                beamdata = pd.DataFrame.from_dict(df["beamData"])

            Sv, ya, za = self.pipeline(beamdata, df)

            Sv_list[ind] = Sv
            ya_list[ind] = ya
            za_list[ind] = za

        mean_ya = sum(ya_list) / len(ya_list)
        mean_za = sum(za_list) / len(za_list)
        mean_Sv = sum(Sv_list) / len(Sv_list)

        return mean_Sv, mean_ya, mean_za
    

    def compute_radial_means(self, Sv, a, b):
        radialMeans = []

        for i in range(Sv.shape[1]):
            values = Sv.iloc[a:b, i]
            radialMeans.append(values.mean())
            
        return radialMeans
    

    def setup(self):
        self.Sv, self.ya, self.za = self.multiple_ping(4, 5)
        
        self.a_init = self.Sv.shape[0]
        self.b_init = self.Sv.shape[1]
        self.c_init = 10

        self.radial_means = self.compute_radial_means(self.Sv, 0, self.b_init)

        self.fig = plt.figure(figsize=(10, 5))
        self.gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
        self.ax1 = self.fig.add_subplot(self.gs[0, 0])
        self.ax2 = self.fig.add_subplot(self.gs[0,1])

        self.axs = [self.ax1, self.ax2]

        plt.subplots_adjust(left=0.1, bottom=0.3, right=0.7, top=0.9, wspace=0.3)

        self.fig.canvas.manager.set_window_title('KMWCD Viewer')

        self.pcolor_plot = self.axs[0].pcolor(self.ya.iloc[:self.a_init, :self.b_init], self.za.iloc[:self.a_init, :self.b_init], self.Sv.iloc[:self.a_init, :self.b_init], vmin=-120, vmax=-30, cmap='inferno')
        self.cbar = self.fig.colorbar(self.pcolor_plot, ax = self.axs[0])
        self.cbar.set_label('Decibels', rotation=270)
        self.axs[0].set_title(f'Data for Pings {1}:{self.c_init}')
        self.axs[0].set_ylabel('Depth from Surface [m]')
        self.axs[0].set_xlabel('Distance [m]')

        self.prev_e = 0
        self.prev_f = 0


        def backing(x, y, w, h):
            self.inset_ax = plt.axes([x, y, w, h])
            self.inset_ax.set_facecolor('lightgrey')
            self.inset_ax.text(0.5, 0.5, '', horizontalalignment='center', verticalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

            self.inset_ax.set_xticks([])
            self.inset_ax.set_yticks([])
            self.inset_ax.set_xticklabels([])
            self.inset_ax.set_yticklabels([])

        backing(0.725,0.035,0.25,0.885)
        backing(0.055,0.035,0.37,0.17)


        self.ax_slider_a = plt.axes([0.1, 0.10, 0.23, 0.03], facecolor='lightgoldenrodyellow')
        self.slider_a = RangeSlider(self.ax_slider_a, 'Depth', 1, self.Sv.shape[0], valinit=(1, self.a_init), valstep=1)
        self.slider_a.track.set_facecolor('white')
        self.slider_a.valtext.set_visible(False)

        self.ax_textbox_a_start = plt.axes([0.34, 0.10, 0.03, 0.03])
        self.textbox_a_start = TextBox(self.ax_textbox_a_start, '', initial=str(1))

        self.ax_textbox_a_end = plt.axes([0.38, 0.10, 0.03, 0.03])
        self.textbox_a_end = TextBox(self.ax_textbox_a_end, '', initial=str(self.a_init))

        self.ax_slider_b = plt.axes([0.1, 0.05, 0.23, 0.03], facecolor='lightblue')
        self.slider_b = RangeSlider(self.ax_slider_b, 'Beams', 1, self.Sv.shape[1], valinit=(1, self.b_init), valstep=1)
        self.slider_b.track.set_facecolor('white')
        self.slider_b.valtext.set_visible(False)

        self.ax_textbox_b_start = plt.axes([0.34, 0.05, 0.03, 0.03])
        self.textbox_b_start = TextBox(self.ax_textbox_b_start, '', initial=str(1))

        self.ax_textbox_b_end = plt.axes([0.38, 0.05, 0.03, 0.03])
        self.textbox_b_end = TextBox(self.ax_textbox_b_end, '', initial=str(self.b_init))

        self.ax_slider_c = plt.axes([0.1, 0.15, 0.23, 0.03], facecolor='purple')
        self.slider_c = RangeSlider(self.ax_slider_c, 'Pings', 1, self.max_pings, valinit=(1, self.c_init), valstep=1)
        self.slider_c.track.set_facecolor('white')
        self.slider_c.valtext.set_visible(False)

        self.ax_textbox_c_start = plt.axes([0.34, 0.15, 0.03, 0.03])
        self.textbox_c_start = TextBox(self.ax_textbox_c_start, '', initial=str(1))

        self.ax_textbox_c_end = plt.axes([0.38, 0.15, 0.03, 0.03])
        self.textbox_c_end = TextBox(self.ax_textbox_c_end, '', initial=str(self.c_init))


        def label(x, y, w, h, label):
            ax_label = plt.axes([x, y, w, h])
            ax_label.axis('off')  # Hide the axes for the label
            ax_label.text(0, 0.5, f'{label}', verticalalignment='center')


        self.ax_textbox_d = plt.axes([0.73,0.64,0.03,0.03])
        self.textbox_d = TextBox(self.ax_textbox_d, '', initial=str(10))

        label(0.765,0.64,0.03,0.03,'pings per frame')

        self.ax_textbox_e = plt.axes([0.73,0.59,0.03,0.03])
        self.textbox_e = TextBox(self.ax_textbox_e, '', initial=str(6))

        label(0.765,0.59,0.03,0.03,'frames per second')

        self.ax_textbox_f = plt.axes([0.73,0.50,0.03,0.03])
        self.textbox_f = TextBox(self.ax_textbox_f, '', initial=str(10))

        label(0.765,0.50,0.03,0.03,'pings per gridpoint')

        self.ax_textbox_g = plt.axes([0.73,0.45,0.03,0.03])
        self.textbox_g = TextBox(self.ax_textbox_g, '', initial=str(10))

        label(0.765,0.45,0.03,0.03,'beams per gridpoint')

        label(0.83,0.68,0.03,0.03,'Animation')

        self.ax_button = plt.axes([0.86,0.64,0.11,0.03])
        self.animate_button = Button(self.ax_button, 'Animate')

        self.ax_button = plt.axes([0.86,0.59,0.11,0.03])
        self.animate_all_button = Button(self.ax_button, 'Animate All')

        self.ax_button = plt.axes([0.78,0.045,0.15,0.03])
        self.quit_button = Button(self.ax_button, 'Quit')

        self.ax_button = plt.axes([0.73,0.872,0.24,0.028])
        self.file_button = Button(self.ax_button, 'File Upload')

        label(0.73,0.832,0.24,0.028,f'Selected file: {os.path.basename(name_for)}')

        self.ax_button = plt.axes([0.73,0.772,0.24,0.028])
        self.folder_button = Button(self.ax_button, 'Folder Upload')

        try:
            label(0.73,0.732,0.24,0.028,f"Selected folder: {os.path.basename(folder)}")
        except:
            label(0.73,0.732,0.24,0.028,f"Selected folder: None")

        label(0.83,0.53,0.03,0.03,"Modeling")

        self.ax_button = plt.axes([0.73,0.40,0.24,0.028])
        self.three_d_model_button = Button(self.ax_button, 'Create Model')

        self.ax_button = plt.axes([0.49,0.12,0.16,0.028])
        self.save_map_button = Button(self.ax_button, 'Save Map')

        self.ax_button = plt.axes([0.49,0.08,0.16,0.028])
        self.save_averages_button = Button(self.ax_button, 'Save Plot')


    def main(self):
        # Functions defined inside the main loop

        def update_range_slider(range_slider, new_valmin, new_valmax, set_valmax):
            range_slider.valmin = new_valmin
            range_slider.valmax = new_valmax
            range_slider.ax.set_xlim(new_valmin, new_valmax)
            range_slider.set_val((new_valmin, set_valmax))  # Reset the slider value to be within the new range
            range_slider.valinit = (new_valmin, new_valmax)

        def update_file():
            # Logic to update the file being viewed
            pass

        def load_folder():
            # Logic for loading a folder
            pass

        def animate():
            # Logic for handling animation of data
            pass

        def create_animation():
            # Logic to create a single animation
            pass

        def create_animation_all():
            # Logic to create animations for all data
            pass

        def create_wire_model():
            # Logic to create a wireframe model
            pass

        def save_map():
            # Logic to save a map of the data
            pass

        def update_slider_from_textbox():
            # Logic to update slider values based on textbox input
            pass

        def update_textbox_from_slider():
            # Logic to update textbox based on slider movement
            pass

        def quit():
            # Logic to quit the application
            pass

        # Main loop or interaction logic goes here
        # For example, running animation, handling events, etc.
