import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, TextBox, Button
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
import KMALL
from tkinter import filedialog
from tqdm import tqdm
import warnings
import os
import time
import sys


# Set font family for plots
plt.rcParams["font.family"] = "Times New Roman"

# # Define the KMALL file path
kmall_file = "data/sample.kmwcd"

# # Initialize KMALL object and index file
K = KMALL.kmall(kmall_file)
K.index_file()

# # Print report on packet types (optional, for debugging)
K.report_packet_types()

# # Extract MWC messages and their count
IMWC = K.Index["MessageType"] == "b'#MWC'"
MWCIndex = K.Index[IMWC]
max_pings = len(MWCIndex)

last_dot_index = kmall_file.rfind('.')
name_for = kmall_file[:last_dot_index]

# Suppress specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=mpl.MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def pipeline(dg):
    # print(dg["rxInfo"])
    # Extract necessary data from KMALL data structure
    beamAmp = pd.DataFrame.from_dict(dg['beamData']['sampleAmplitude05dB_p'])
    numsamp = dg['beamData']['numSampleData']
    SoundSp = dg["rxInfo"]["soundVelocity_mPerSec"]
    SampleFreq = dg["rxInfo"]["sampleFreq_Hz"]
    TVGFuncApplied = dg["rxInfo"]["TVGfunctionApplied"]
    TVGOffset = dg["rxInfo"]["TVGoffset_dB"]
    txBeamWidth = dg['sectorData']['txBeamWidthAlong_deg']
    beamAngle = np.array(dg['beamData']['beamPointAngReVertical_deg'])

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
    dg = lambda ping: K.read_index_row(MWCIndex.iloc[ping-1])
    df = dg(ping)

    Sv1, ya1, za1 = pipeline(df)

    return Sv1, ya1, za1


def pingAvg(startPing, endPing):
    # Determine the number of pings
    num_pings = endPing - startPing + 1

    # Initialize preallocated arrays with None or zeros if sizes are known
    Sv_list = [None] * num_pings
    ya_list = [None] * num_pings
    za_list = [None] * num_pings

    for ind, i in enumerate(range(startPing, endPing + 1)):
        dg = lambda ping: K.read_index_row(MWCIndex.iloc[i-1])
        # print(ind, i)
        df = dg(i)

        Sv1, ya1, za1 = pipeline(df)

        Sv_list[ind] = Sv1
        ya_list[ind] = ya1
        za_list[ind] = za1

    mean_ya = sum(ya_list) / len(ya_list)
    mean_za = sum(za_list) / len(za_list)
    mean_Sv = sum(Sv_list) / len(Sv_list)

    return mean_Sv, mean_ya, mean_za


def compute_radial_means(mean_Sv, a, b):
    radialMeans = []
    for i in range(mean_Sv.shape[1]):
        values = mean_Sv.iloc[a:b, i]
        radialMeans.append(values.mean())
    return radialMeans


# Main plotting and interaction code
def main():
    global mean_Sv, mean_ya, mean_za, prev_e, prev_f, kmall_file, K, IMWC, MWCIndex, max_pings

    mean_Sv, mean_ya, mean_za = pingAvg(4, 5)

    a_initial = mean_Sv.shape[0]
    b_initial = mean_Sv.shape[1]
    c_initial = 10
    radialMeans = compute_radial_means(mean_Sv, 0, b_initial)

    # Start of new plotting
    # Create a figure
    fig = plt.figure(figsize=(10, 5))

    # Create a GridSpec object with 1 row and 4 columns
    gs = gridspec.GridSpec(1, 1, width_ratios=[1])

    # Add the first subplot that takes up half of the figure
    ax1 = fig.add_subplot(gs[0, 0])

    # Add the second and third subplots that each take up a quarter of the figure
    # ax2 = fig.add_subplot(gs[0, 1])

    axs = [ax1]
    plt.subplots_adjust(left=0.1, bottom=0.3, right=0.7, top=0.9, wspace=0.3)
    # End of new plotting

    # Set the window title
    fig.canvas.manager.set_window_title('KMWCD Analysis')

    pcolor_plot = axs[0].pcolor(mean_ya.iloc[:a_initial, :b_initial], mean_za.iloc[:a_initial, :b_initial], mean_Sv.iloc[:a_initial, :b_initial], vmin=-120, vmax=-30, cmap='inferno')
    # pcolor_plot = axs[0].pcolor(mean_ya.iloc[:a_initial, :b_initial], mean_za.iloc[:a_initial, :b_initial], mean_Sv.iloc[:a_initial, :b_initial], vmin=-120, vmax=-30, cmap='viridis')
    cbar = fig.colorbar(pcolor_plot, ax=axs[0])
    cbar.set_label('Decibels', rotation=270)
    # print(mean_ya,mean_za)
    
    ya1_min, ya1_max = mean_ya.min().min(), mean_ya.max().max()
    za1_min, za1_max = mean_za.min().min(), mean_za.max().max()
    
    axs[0].set_xlim(ya1_min, ya1_max)
    axs[0].set_ylim(za1_min, za1_max)
    axs[0].set_title(f'Data for Pings {1}:{c_initial}')
    axs[0].set_ylabel('Depth from Surface [m]')
    axs[0].set_xlabel('Distance [m]')


    def gauss(x, H, A, x0, sigma):
        return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    def gauss_fit(x, y):
        mean = sum(x * y) / sum(y)
        sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
        popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma], maxfev=5000)
        return popt

    y_fit = gauss(np.arange(1, len(radialMeans) + 1), *gauss_fit(np.arange(1, len(radialMeans) + 1), radialMeans))
    # gauss_fit_plot, = axs[1].plot(np.arange(1, len(radialMeans) + 1), y_fit, color='black')
    # gauss_scatter_plot = axs[1].scatter(np.arange(1, len(radialMeans) + 1), radialMeans, color='red', s=4, zorder=1)
    # axs[1].set_title('Reading per Beam')
    # axs[1].set_ylabel('Average Reading [m$^{-1}$]')
    # axs[1].set_xlabel('Beam #')

    prev_e = 0
    prev_f = 0

    # Function to update data when sliders or text boxes are changed
    def update_data(val):
        global mean_Sv, mean_ya, mean_za, prev_e, prev_f
        try:
            a, b = map(int, slider_a.val)
            c, d = map(int, slider_b.val)
            e, f = map(int, slider_c.val)

            if e != prev_e or f != prev_f:
                prev_e = e
                prev_f = f
                mean_Sv, mean_ya, mean_za = pingAvg(e, f)
                axs[0].relim()
                axs[0].autoscale_view()
                axs[0].set_title(f'Data for Pings {e}:{f}')

            radialMeans = compute_radial_means(mean_Sv, a, b)

            for coll in axs[0].collections:
                coll.remove()
            pcolor_plot = axs[0].pcolor(mean_ya.iloc[a:b, c:d], mean_za.iloc[a:b, c:d], mean_Sv.iloc[a:b, c:d], vmin=-120, vmax=-30, cmap='inferno')
            cbar.update_normal(pcolor_plot)

            x_data = np.arange(c, d)
            y_data = radialMeans[c:d]

            axs[0].relim()
            axs[0].autoscale_view()

            # axs[1].relim()
            # axs[1].autoscale_view()

            # plt.savefig('high_fidelity_plot.png', dpi=300, bbox_inches='tight')

            fig.canvas.draw_idle() 
        except:
            print("Check selected file!")


    # Function to update the range slider range
    def update_range_slider(range_slider, new_valmin, new_valmax, set_valmax):
        range_slider.valmin = new_valmin
        range_slider.valmax = new_valmax
        range_slider.ax.set_xlim(new_valmin, new_valmax)
        range_slider.set_val((new_valmin, set_valmax))  # Reset the slider value to be within the new range
        range_slider.valinit = (new_valmin, new_valmax)


    def update_file(event, file):
        global kmall_file, K, IMWC, MWCIndex, max_pings, name_for
        # Define the KMALL file path

        if file == None:
            kmall_file = filedialog.askopenfilename(filetypes=[("KMWCD Files", "*.kmwcd")])
            last_dot_index = kmall_file.rfind('.')
            name_for = kmall_file[:last_dot_index]
            if kmall_file == '':
                del kmall_file
            try:
                file_label.set_text(f"Selected file: {os.path.basename(kmall_file)}")
            except:
                file_label.set_text("Selected file: None")
        elif file:
            kmall_file = file
            last_dot_index = kmall_file.rfind('.')
            name_for = kmall_file[:last_dot_index]


        # Initialize KMALL object and index file
        K = KMALL.kmall(kmall_file)
        K.index_file()

        # Print report on packet types (optional, for debugging)
        K.report_packet_types()

        # Extract MWC messages and their count
        IMWC = K.Index["MessageType"] == "b'#MWC'"
        MWCIndex = K.Index[IMWC]
        max_pings = len(MWCIndex)

        mean_Sv, mean_ya, mean_za = pingAvg(4, 5)

        a_initial = mean_Sv.shape[0]
        b_initial = mean_Sv.shape[1]
        c_initial = 10
        radialMeans = compute_radial_means(mean_Sv, 0, b_initial)

        # a_max = ax_slider_a.valmax(a_initial)
        # b_max = ax_slider_b.valmax(b_initial)
        # c_max = ax_slider_c.valmax(c_initial)

        # ax_slider_a.set_xlim((1, a_initial))
        # ax_slider_b.set_xlim((1, b_initial))
        # ax_slider_c.set_xlim((1, c_initial))

        update_range_slider(slider_a, 1, a_initial, a_initial)
        update_range_slider(slider_b, 1, b_initial, b_initial)
        update_range_slider(slider_c, 1, max_pings, 11)
        slider_c.set_val((1,10))

        # update_slider_from_textbox(1)
        update_data(1)

        fig.canvas.draw_idle() 
        

    def load_folder(event):
        global folder
        folder = filedialog.askdirectory()

        if folder == '':
            del folder
        try:
            folder_label.set_text(f"Selected folder: {os.path.basename(folder)}")
            fig.canvas.draw_idle() 
        except:
            folder_label.set_text("Selected folder: None")


    # Function to update sliders from text boxes
    def update_slider_from_textbox(event, slider):
        try:
            a = int(textbox_a_start.text)
            b = int(textbox_a_end.text)
            c = int(textbox_b_start.text)
            d = int(textbox_b_end.text)
            e = int(textbox_c_start.text)
            f = int(textbox_c_end.text)
            
            if slider == 'a':
                slider_a.set_val((a, b))
            elif slider == 'b':
                slider_b.set_val((c, d))
            elif slider == 'c':
                slider_c.set_val((e, f))

        except ValueError:
            pass


    def create_animation(event):
        n = int(textbox_d.text) # pings per frame
        framerate = int(textbox_e.text)
        # This should be defined based on your dataset

        # Add debug statement to confirm the function is called
        # print("Animation button clicked")

        # Create a new figure and axis for the animation
        anim_fig, anim_ax = plt.subplots()
        plt.subplots_adjust(left=0.10, bottom=0.10, right=0.95, top=0.90, wspace=0.0)

        pbar = tqdm(total=int(max_pings / n))

        Sv1, ya1, za1 = pingAvg(n, 2 * n) 
        # Get the smallest and largest values from ya1 and za1
        ya1_min, ya1_max = ya1.min().min(), ya1.max().max()
        za1_min, za1_max = za1.min().min(), za1.max().max()

        def animate(frame):
            Sv1, ya1, za1 = pingAvg(frame * n, frame * n + n) 
            anim_ax.clear()
            anim_ax.pcolor(ya1, za1, Sv1, vmin=-120, vmax=-30, cmap='inferno')
            anim_ax.set_title(f'Pings {frame * n} to {frame * n + n}')
            anim_ax.set_ylabel('Depth from Surface [m]')
            anim_ax.set_xlabel('Distance [m]')
            anim_ax.set_xlim(ya1_min, ya1_max)
            anim_ax.set_ylim(za1_min, za1_max)

            # Add debug statement to check frame number
            # print(f"Animating frame: {frame}")

            pbar.update(1) 
  
        ani = animation.FuncAnimation(anim_fig, animate, frames=int(max_pings / n), interval=300, repeat=True)

        # print(name_for)
        ani.save(f'{name_for}_{n}ppf_{framerate}fps.gif', writer='imagemagick', fps=framerate)
        # Add debug statement to confirm animation setup
        # print("Animation Completed")f


    def create_all_animations(event):
        start = time.time()
        try:
            print(f"Selected folder: {folder}")
            n = 0
            animated = 0
            for filename in os.listdir(folder):
                if filename.endswith(".kmwcd"):  # Example file type filter, adjust as needed
                    n += 1
            for filename in os.listdir(folder):
                try:
                    if filename.endswith(".kmwcd"):
                        animated += 1
                        file_path = os.path.join(folder, filename)  # Get full file path
                        print(file_path)
                        update_file(1, file_path)

                        # fig.canvas.draw_idle() 
                        print(f"Creating animation {animated} of {n}")
                        create_animation(1) 
                except:
                    print(f"File {animated} failed to animate. Moving to next file.")
                    continue
            end = time.time()
            print(f"Animations completed in {end-start} seconds")
        except:
            print("No folder is currently selected!")


    def create_3d_model(event):
        save_path = os.getcwd()
        # Read input values
        a, b = map(int, slider_a.val)
        n = int(textbox_f.text)
        h = int(textbox_g.text)

        # Initialize progress bar
        pbar = tqdm(total=int(max_pings / n))

        # Initial computation
        Sv1, ya1, za1 = pingAvg(n, 2 * n)
        radialMeans = compute_radial_means(Sv1, a, b)
        xvalues = np.arange(1, len(radialMeans) + 1)

        allxvalues, allyvalues = [], []
        allfittedxvalues, allfittedyvalues = [], []

        for frame in range(int(max_pings / n)):
            Sv1, ya1, za1 = pingAvg(frame * n, frame * n + n)
            radialMeans = compute_radial_means(Sv1, a, b)

            degree = 10
            coefficients = np.polyfit(xvalues, radialMeans, degree)
            x_fit = list(np.linspace(xvalues.min(), xvalues.max(), int(len(radialMeans) / h)))
            y_fit = list(np.polyval(coefficients, x_fit))

            allfittedxvalues.append(x_fit)
            allfittedyvalues.append(y_fit)
            allxvalues.append(xvalues)
            allyvalues.append(radialMeans)
            pbar.update(1)

        # Generate z data
        allzvalues = np.array([[i * n] * len(row) for i, row in enumerate(allfittedxvalues)])
        allfittedxvalues = np.array(allfittedxvalues)
        allfittedyvalues = np.array(allfittedyvalues)

        # Create 3D figure
        fig2 = plt.figure(figsize=(8, 6))
        ax3 = fig2.add_subplot(111, projection='3d')
        surface = ax3.plot_surface(allfittedxvalues, allzvalues, allfittedyvalues, cmap='inferno')
        fig2.colorbar(surface, ax=ax3, label='Intensity')
        ax3.set_xlabel('Beam #')
        ax3.set_ylabel('Ping #')
        ax3.set_zlabel('Intensity')
        ax3.set_title('Plume Strength')
        ax3.set_box_aspect([1, 1, 1])
        fig2.savefig(os.path.join(save_path, f"{name_for}_{n}ppg_{h}bpg_3d_model.png"))
        plt.close(fig2)

        # Create 2D figure
        fig3 = plt.figure(figsize=(8, 6))
        ax4 = fig3.add_subplot()
        pcolor_plot = ax4.pcolor(allfittedxvalues, allzvalues, allfittedyvalues, cmap='inferno')
        fig3.colorbar(pcolor_plot, ax=ax4)
        ax4.set_title('Plume Strength')
        ax4.set_ylabel('Ping #')
        ax4.set_xlabel('Beam #')
        fig3.savefig(os.path.join(save_path, f"{name_for}_{n}ppg_{h}bpg_2d_model.png"))
        plt.close(fig3)


    def create_3d_model_all(event):
        start = time.time()
        try:
            print(f"Selected folder: {folder}")
            kmwcd_files = [f for f in os.listdir(folder) if f.endswith(".kmwcd")]
            n = len(kmwcd_files)

            for i, filename in enumerate(kmwcd_files, 1):
                try:
                    file_path = os.path.join(folder, filename)
                    print(f"[{i}/{n}] Creating model for: {file_path}")

                    # Load data for this file
                    update_file(1, file_path)

                    # Create subfolder for output
                    base_name = os.path.splitext(filename)[0]
                    output_path = os.path.join(folder)
                    os.makedirs(output_path, exist_ok=True)

                    create_3d_model(output_path)
                except:
                    print(f"File {i} failed to model. Moving to next file.")
                    continue

            print(f"Models completed in {time.time() - start:.2f} seconds")

        except Exception as e:
            print(f"Failed to create models: {e}")


    def save_map(event):
        a, b = map(int, slider_c.val)

        # Get the bounding box of the first subplot (ax1)
        bbox = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        # Save only the specific subplot with additional padding
        plt.savefig(f'{name_for}_pings{a}to{b}.png', bbox_inches=bbox.expanded(1.47, 1.3), dpi=300)


    def save_averages(event):
        # Get the bounding box of the first subplot (ax1)
        bbox = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        # Save only the specific subplot with additional padding
        plt.savefig(f'{name_for}_ping_averages.png', bbox_inches=bbox.expanded(1.47, 1.3), dpi=300)


    def quit(event):
        sys.exit()


    inset_ax = plt.axes([0.725,0.035,0.25,0.885])
    # inset_ax.axis('off')  # Hide the axes for the label
    inset_ax.set_facecolor('lightgrey')
    inset_ax.text(0.5, 0.5, '', horizontalalignment='center', verticalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_xticklabels([])
    inset_ax.set_yticklabels([])

    inset_ax = plt.axes([0.055,0.035,0.61,0.17])
    # inset_ax.axis('off')  # Hide the axes for the label
    inset_ax.set_facecolor('lightgrey')
    inset_ax.text(0.5, 0.5, '', horizontalalignment='center', verticalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_xticklabels([])
    inset_ax.set_yticklabels([])

    # Create sliders
    ax_slider_a = plt.axes([0.1, 0.10, 0.43, 0.03], facecolor='lightgoldenrodyellow')
    slider_a = RangeSlider(ax_slider_a, 'Depth', 1, mean_Sv.shape[0], valinit=(1, a_initial), valstep=1)
    slider_a.track.set_facecolor('white')
    slider_a.valtext.set_visible(False)

    ax_slider_b = plt.axes([0.1, 0.05, 0.43, 0.03], facecolor='lightblue')
    slider_b = RangeSlider(ax_slider_b, 'Beams', 1, mean_Sv.shape[1], valinit=(1, b_initial), valstep=1)
    slider_b.track.set_facecolor('white')
    slider_b.valtext.set_visible(False)

    ax_slider_c = plt.axes([0.1, 0.15, 0.43, 0.03], facecolor='purple')
    slider_c = RangeSlider(ax_slider_c, 'Pings', 1, max_pings, valinit=(1, c_initial), valstep=1)
    slider_c.track.set_facecolor('white')
    slider_c.valtext.set_visible(False)

    # Create text boxes
    ax_textbox_a_start = plt.axes([0.54, 0.10, 0.03, 0.03])
    textbox_a_start = TextBox(ax_textbox_a_start, '', initial=str(1))

    ax_textbox_a_end = plt.axes([0.58, 0.10, 0.03, 0.03])
    textbox_a_end = TextBox(ax_textbox_a_end, '', initial=str(a_initial))

    ax_textbox_b_start = plt.axes([0.54, 0.05, 0.03, 0.03])
    textbox_b_start = TextBox(ax_textbox_b_start, '', initial=str(1))

    ax_textbox_b_end = plt.axes([0.58, 0.05, 0.03, 0.03])
    textbox_b_end = TextBox(ax_textbox_b_end, '', initial=str(b_initial))

    ax_textbox_c_start = plt.axes([0.54, 0.15, 0.03, 0.03])
    textbox_c_start = TextBox(ax_textbox_c_start, '', initial=str(1))

    ax_textbox_c_end = plt.axes([0.58, 0.15, 0.03, 0.03])
    textbox_c_end = TextBox(ax_textbox_c_end, '', initial=str(c_initial))

    ax_textbox_d = plt.axes([0.73,0.64,0.03,0.03])
    textbox_d = TextBox(ax_textbox_d, '', initial=str(10))

    ax_label = plt.axes([0.765,0.64,0.03,0.03])
    ax_label.axis('off')  # Hide the axes for the label
    ax_label.text(0, 0.5, 'pings per frame', verticalalignment='center')

    ax_textbox_e = plt.axes([0.73,0.59,0.03,0.03])
    textbox_e = TextBox(ax_textbox_e, '', initial=str(6))

    ax_label = plt.axes([0.765,0.59,0.03,0.03])
    ax_label.axis('off')  # Hide the axes for the label
    ax_label.text(0, 0.5, 'frames per second', verticalalignment='center')

    ax_textbox_f = plt.axes([0.73,0.41,0.03,0.03])
    textbox_f = TextBox(ax_textbox_f, '', initial=str(10))

    ax_label = plt.axes([0.765,0.41,0.03,0.03])
    ax_label.axis('off')  # Hide the axes for the label
    ax_label.text(0, 0.5, 'pings per gridpoint', verticalalignment='center')

    ax_textbox_g = plt.axes([0.73,0.36,0.03,0.03])
    textbox_g = TextBox(ax_textbox_g, '', initial=str(10))

    ax_label = plt.axes([0.765,0.36,0.03,0.03])
    ax_label.axis('off')  # Hide the axes for the label
    ax_label.text(0, 0.5, 'beams per gridpoint', verticalalignment='center')

    # Create buttons
    ax_label = plt.axes([0.82,0.68,0.03,0.03])
    ax_label.axis('off')  # Hide the axes for the label
    ax_label.text(0, 0.5, 'Animation', verticalalignment='center', fontweight='bold')

    ax_button = plt.axes([0.73,0.54,0.24,0.028])
    button = Button(ax_button, 'Animate')

    ax_button = plt.axes([0.73,0.50,0.24,0.028])
    button_animate_all = Button(ax_button, 'Animate All')

    ax_button = plt.axes([0.62,0.05,0.04,0.13])
    button_run = Button(ax_button, 'Run')

    ax_button_quit = plt.axes([0.73,0.045,0.24,0.028])
    button_quit = Button(ax_button_quit, 'Quit')
    
    ax_button_file = plt.axes([0.73,0.872,0.24,0.028])
    button_file = Button(ax_button_file, 'File Upload')

    ax_label = plt.axes([0.73,0.832,0.24,0.028])
    ax_label.axis('off')  # Hide the axes for the label
    file_label = ax_label.text(0, 0.5, f'Selected file: {os.path.basename(name_for)}', verticalalignment='center')

    ax_button_folder = plt.axes([0.73,0.772,0.24,0.028])
    button_folder = Button(ax_button_folder, 'Folder Upload')

    ax_label = plt.axes([0.73,0.732,0.24,0.028])
    ax_label.axis('off')  # Hide the axes for the label
    try:
        folder_label = ax_label.text(0, 0.5, f"Selected folder: {os.path.basename(folder)}", verticalalignment='center')
    except:
        folder_label = ax_label.text(0, 0.5, 'Selected folder: None', verticalalignment='center')

    ax_label = plt.axes([0.82,0.45,0.03,0.03])
    ax_label.axis('off')  # Hide the axes for the label
    ax_label.text(0, 0.41, 'Modeling', verticalalignment='center', fontweight='bold')

    ax_3d_model = plt.axes([0.73,0.31,0.24,0.028])
    button_3d_model = Button(ax_3d_model, 'Create Model')

    ax_3d_model = plt.axes([0.73,0.27,0.24,0.028])
    button_3d_model_all = Button(ax_3d_model, 'Create All Models')

    ax_save_map = plt.axes([0.73,0.10,0.24,0.028])
    button_save_map = Button(ax_save_map, 'Save Map')

    # ax_save_averages = plt.axes([0.78,0.10,0.16,0.028])
    # button_save_averages = Button(ax_save_averages, 'Save Plot')

    # Update text boxes when sliders change
    def update_textbox_from_slider(val, slider):
        a, b = map(int, slider_a.val)
        c, d = map(int, slider_b.val)
        e, f = map(int, slider_c.val)
        
        if slider == 'a':
            textbox_a_start.set_val(str(a))
            textbox_a_end.set_val(str(b))
        elif slider == 'b':
            textbox_b_start.set_val(str(c))
            textbox_b_end.set_val(str(d))
        elif slider == 'c':
            textbox_c_start.set_val(str(e))
            textbox_c_end.set_val(str(f))

    # Link updates
    slider_a.on_changed(lambda event: update_textbox_from_slider(event, slider='a'))
    slider_b.on_changed(lambda event: update_textbox_from_slider(event, slider='b'))
    slider_c.on_changed(lambda event: update_textbox_from_slider(event, slider='c'))

    # slider_a.on_changed(update_data)
    # slider_b.on_changed(update_data)
    # slider_c.on_changed(update_data)

    textbox_a_start.on_submit(lambda event: update_slider_from_textbox(event, slider='a'))
    textbox_a_end.on_submit(lambda event: update_slider_from_textbox(event, slider='a'))
    textbox_b_start.on_submit(lambda event: update_slider_from_textbox(event, slider='b'))
    textbox_b_end.on_submit(lambda event: update_slider_from_textbox(event, slider='b'))
    textbox_c_start.on_submit(lambda event: update_slider_from_textbox(event, slider='c'))
    textbox_c_end.on_submit(lambda event: update_slider_from_textbox(event, slider='c'))

    button.on_clicked(create_animation)
    button_run.on_clicked(update_data)
    button_quit.on_clicked(quit)
    button_file.on_clicked(lambda event: update_file(event, file=None))
    button_folder.on_clicked(load_folder)
    button_animate_all.on_clicked(create_all_animations)
    button_3d_model.on_clicked(create_3d_model)
    button_3d_model_all.on_clicked(create_3d_model_all)
    # button_save_averages()
    button_save_map.on_clicked(save_map)
    # button_save_averages.on_clicked(save_averages)

    plt.show()


if __name__ == "__main__":
    main()

    