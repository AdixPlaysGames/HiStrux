<div style="text-align: center; margin-bottom: 20px;">
    <img src="./addit_files/logo/histrux.png" alt="HiStrux Logo" style="max-width: 100%; margin-bottom: 10px;" />
    <hr style="border: 1px solid white; width: 80%; margin: 0 auto;" />
</div>

<p style="text-align: justify;">
    <strong>HiStrux</strong> is a Python package designed for reconstructing chromatin structures from single-cell Hi-C (scHi-C) data. 
    It integrates feature extraction, machine learning, and iterative molecular dynamics simulations to facilitate both 
    data enrichment and 3D genome modeling. The entire concept is based on:
</p>

<ul style="text-align: justify;">
    <li><strong>eXtract</strong> a variety of metrics and structural features from scHi-C maps using the <code>eXtract</code> module.</li>
    <li><strong>CycleSort</strong> cells into interphase stages with a neural network-based model (<code>CycleSort</code>) and use this information for selective data pooling.</li>
    <li><strong>reConstruct</strong> spatial chromatin organization with the <code>reConstruct</code> module, implementing iterative simulations built on top of HOOMD-blue.</li>
</ul>

<p style="text-align: justify;">
    HiStrux aims to bridge gaps between single-cell heterogeneity and richer multi-cell insights, providing an easy-to-use framework for more accurate 3D chromatin reconstructions.
</p>

## Table of Contents
- [Theory Behind HiStrux](#theory-behind-histrux)
- [Modules](#modules)
  - [`eXtract`](#extract)
  - [`reConstruct`](#reconstruct)
- [Installation Guide](#installation-guide)
- [CycleSort as a Use Case Example](#cyclesort-as-a-use-case-example)


## Theory Behind HiStrux

Single-cell Hi-C (scHi-C) data offers insights into how DNA is spatially organized within individual cells, capturing the true variability often lost when many cells are pooled together. However, scHi-C maps are typically sparse and challenging to reconstruct, given the limited number of contacts detected per cell. Traditional “bulk” methods average data from many cells, potentially obscuring distinct structural features.

**HiStrux** addresses this issue by providing:
- **eXtract** – A feature extraction module inspired by existing research. It processes raw scHi-C contact maps, performs quality checks and imputation, and then generates informative metrics. These metrics help classify cells into different interphase stages, guiding further data enrichment.
- **reConstruct** – An iterative molecular dynamics approach built upon HOOMD-blue, leveraging the cell-stage context uncovered by eXtract. This “semi-bulk” enrichment method pools only relevant contacts from similar cells, striking a balance between single-cell detail and the improved coverage seen in bulk data.

By focusing on both accurate feature extraction and more informed simulation runs, HiStrux captures critical single-cell variability while mitigating the pitfalls of sparse datasets, yielding more robust 3D chromatin reconstructions.

<div style="text-align: center; margin-bottom: 20px;">
    <hr style="border: 1px solid white; width: 80%; margin: 0 auto;" />
</div>

## Modules
<img src="./addit_files/logo/extract.png" alt="Extract Logo" style="width: 25%; float: left; margin-right: 10px;" />
<p style="text-align: justify;">
</p>

### Description:
**eXtract** is a module inspired by the CIRCLET methodology for preprocessing and feature extraction from scHi-C maps.

### Key Features:
- Processing raw data (contact matrix generation, binning, filtering).
- Imputation of missing values in the contact matrix.
- Extraction of global features as percentage of trans-interactions, contact lengths.
- Calculation of metrics as insulation score, TAD boundaries, A/B compartment division.
- Visualization of processed matrices and selected statistics (facilitates parameter tuning and quality control).

### Applications:
- Quickly generates a feature vector describing each cell (example - for classification, clustering, or further structural analyses).

---

<img src="./addit_files/logo/reconstruct.png" alt="Extract Logo" style="width: 25%; float: left; margin-right: 10px;" />
<p style="text-align: justify;">
</p>
**reConstruct** was created for iterative 3D reconstruction of chromatin structures using scHi-C matrices (including enriched matrices based on classification results taken from the example of CycleSort).

### Key Features:
- Resolution scaling of scHi-C matrices (from low to high) with iterative simulation runs.
- Generation of particle systems (particles = bins from contact matrices) with force constraints (e.g., chromosomal chains, contacts, etc.).
- Dynamic simulations (e.g., using HOOMD-blue) with controlled interaction forces and parameters (e.g., temperature).
- Final 3D configuration of particles, incorporating contact map information.
- Visualization and evaluation (e.g., RMSD calculations for repeated reconstructions).

### Applications:
- Reconstructs the spatial organization of chromatin in selected cells.
- Optionally uses interphase phase information (from CycleSort) to enrich scHi-C data and reduce noise in single-cell data.

<div style="text-align: center; margin-bottom: 20px;">
    <hr style="border: 1px solid white; width: 80%; margin: 0 auto;" />
</div>

## Installation Guide

To install **HiStrux** locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/AdixPlaysGames/HiStrux.git
    ```
2. **Navigate to the HiStrux folder**:
    ```bash
    cd HiStrux
    ```
3. **Create a new environment** using the dependencies specified in `environment.yml`:
    ```bash
    micromamba create -f environment.yml
    ```
4. **Verify that the environment was created**:
    ```bash
    micromamba env list
    ```
5. **Activate the environment**:
    ```bash
    micromamba activate histrux_env
    ```
6. **Install HiStrux**:
    ```bash
    pip install .
    ```
   This will run the `setup.py` file and handle the installation of all required components. After selecting the `histrux_env` kernel, you can start using HiStrux without any issues.
