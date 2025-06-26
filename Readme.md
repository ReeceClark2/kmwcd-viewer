# KMWCD Data Viewer

This repository presents a lightweight, local data viewer designed to visualize water column data from EM2040 sonar devices, specifically utilizing the Kongsberg "KMALL" file format. Building upon the existing [KMALL file reader](https://github.com/valschmidt/kmall), this tool provides an interface for exploring and analyzing hydrothermal plumes and other water column features that exhibit temperature gradients.

## Features

The KMWCD Data Viewer offers a range of functionalities to guide the user through exploring data:

  * **Interactive 2D Visualization**: The main viewer window displays a 2D representation of the water column data. Users can dynamically adjust the displayed data using three sliders in the lower panel:
      * **Depth**: Restrict the data by depth range.
      * **Beams**: Filter data based on beam numbers.
      * **Pings**: Select a specific range of pings to display.
  * **Animation Generation**:
      * Create **animations** of hydrothermal plumes by stitching together a sequence of pings. You can specify the number of pings per frame and the desired frames per second for the output GIF.
      * Process multiple files at once: If a folder containing `.kmwcd` files is uploaded, the "Animate All" feature will generate individual animations for each file within that folder.
  * **2D and 3D Model Creation**:
      * Generate **2D and 3D models** that represent plume strength based on user-defined parameters for pings per gridpoint and beams per gridpoint. These models provide a comprehensive overview of the plume's characteristics across both beam and ping dimensions.
      * Automate model creation: The "Create All Models" function processes all `.kmwcd` files within a selected folder, generating 2D and 3D models for each.
  * **File Management**: Easily upload individual `.kmwcd` files or entire folders containing multiple data files directly within the viewer.
  * **Image Export**: Save the current 2D map view as a high-resolution PNG image for reports or presentations.

The viewer's primary goal is to provide users with a user-friendly, lightweight, and local tool for quick data assessment and visualization.

-----

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have the following installed:

  * **Python 3.x**: You can download Python from [python.org](https://www.python.org/downloads/).
  * **Git**: For cloning the repository. You can download Git from [git-scm.com](https://git-scm.com/downloads).

### Installation

1.  **Clone the repository:**

    Open your terminal or command prompt and run the following command inside your root directory:

    ```bash
    git clone https://github.com/ReeceClark2/kmwcd-viewer
    ```

2.  **Navigate to the repository directory:**

    ```bash
    cd kmwcd-viewer
    ```

3.  **Install the required dependencies:**

    This project uses several Python libraries. You can install them using `pip`:

    ```bash
    pip install numpy pandas matplotlib scipy tqdm Pillow
    ```

    *Note: The `KMALL` library is a core dependency and is assumed to be handled by the original repository. Ensure it's correctly set up as per the instructions in the [valschmidt/kmall](https://github.com/valschmidt/kmall) repository.*

### Running the Viewer

Once the dependencies are installed, you can run the data viewer by executing the main Python script inside your root directory:

```bash
python KMWCD_FUNC_STRUCT.py
```

The viewer will open a graphical interface, allowing you to load `.kmwcd` files and folders and interact with your data.

-----

## Usage

Upon launching the application, a window will appear displaying a 2D plot of the water column data.

  * **Loading Data**: Use the **"File Upload"** button on the right panel to select a single `.kmwcd` file, or the **"Folder Upload"** button to load all `.kmwcd` files from a directory.
  * **Adjusting Display**:
      * The **"Depth"**, **"Beams"**, and **"Pings"** sliders at the bottom of the window control the range of data displayed in the 2D plot.
      * Alternatively, you can manually enter specific ranges into the corresponding text boxes next to the sliders.
      * Click **"Run"** to apply changes from the sliders or text boxes.
  * **Creating Animations**:
      * Enter the desired **"pings per frame"** and **"frames per second"** in the "Animation" section on the right.
      * Click **"Animate"** to generate a GIF animation of the currently loaded data.
      * Click **"Animate All"** (after loading a folder) to create animations for all `.kmwcd` files in the selected folder.
  * **Generating Models**:
      * In the "Modeling" section, specify the **"pings per gridpoint"** and **"beams per gridpoint"**.
      * Click **"Create Model"** to generate 2D and 3D plume strength models for the active data.
      * Click **"Create All Models"** (after loading a folder) to generate models for all `.kmwcd` files in the selected folder.
  * **Saving Plots**: Use the **"Save Map"** button to export the current 2D water column plot as an image.
  * **Exiting**: Click the **"Quit"** button to close the application.

-----

## Contributing

This repository is open source and available for cloning and collaboration. If you'd like to contribute, please feel free to fork the repository and submit pull requests.

-----

## Next Steps

Ideally, if I had more time, I would like to refactor the code into a class structure. I started this at one point, but I have not been able to finish it. The current working viewer has many remnants from my earlier coding career. I would also like to add a 'Run All' button that runs the 'Animate All' and 'Model All'. I also would have liked to include some more graceful error handling as well.

If there are any bugs or issues then please contact me! I can see if I can fix them or investigate what is going wrong.

There is a known issue with select EM2040 data not loading properly or extremely slowly due to issues with the way the data was recorded.

-----

## Acknowledgments

  * This project builds upon the valuable work of [valschmidt](https://github.com/valschmidt) and their [KMALL file reader](https://github.com/valschmidt/kmall).
