# Low-dose-liquid-phase-STEM
This repository contains Python files related to the manuscript titled "Dynamic observations of collagen mineralization using low-dose liquid phase scanning transmission electron microscopy" by L. Rutten et al. It includes the Python interface for theoretical resolution calulations to determine approprate STEM imaging parameters, sub-sampled image reconstruction (with SenseAI), and image quality assessment (including Radar plots).
If you want to make use of the codes in this repository, use the Jupyter Notebooks.

For image reconstruction (directory: BPFA):
**** You must have the SenseAI enviornment on your computer to run this.
- Jupyter Notebook: BPFA_userFriendly_V1.3.2.ipynb
- .py files: bpfa_ui.py and bpfa_utils.py
- dir PIQ_metrics: additional .py files for metric calculations (match those in RadarPlot_ImgMetrics)

For image quality assessment (directory: RadarPlot_ImgMetrics):
- Jupyter Notebook: RadarPlot_ImgMetrics_v1.1.ipynb
- orignal .py files: utils_bone.py, utils_microscope.py, PIQ_metrics.py, PIQ_plotting.py, PIQ_utils.py, extract_metadata.py
- .py files from other sources: dom.py, niqe.py, piqe.py
- sources for external files: dom.py (https://github.com/umang-singhal/pydom), niqu.py & piqe.py (https://github.com/EadCat/NIQA)

For STEM parameter calculations (file: STEM parameters):
- Jupyter Notebooks: theoretic_calculations.ipynb, resolution_for_thick_dose_range.ipynb
- .py files: utils_bone.py, utils_microscope.py

