# Network Analysis and Graph Machine Learning

## Table of Contents
- [Introduction](#introduction)
- [Project Description](#project-description)
  - [Assignment 1](#assignment-1)
  - [Assignment 2](#assignment-2)
- [Setup Instructions](#setup-instructions)
  - [Step 1: Clone the repository](#step-1-clone-the-repository)
  - [Step 2: Download the Data](#step-2-download-the-data)
  - [Step 3: Setup Development Environment](#step-3-setup-development-environment)
    - [Step 3.1: Install Miniconda (If Uninstalled)](#step-31-install-miniconda-if-uninstalled)
    - [Step 3.2: Create and Activate Conda Environment](#step-32-create-and-activate-conda-environment)
- [Repo Structure](#repo-structure)

---

## Introduction

The domain of our section is graph machine learning in the field of chip design. We spent our first quarter learning about graph fundamentals and different graph machine learning models with a focus on graph neural networks. Besides the lectures, we have 2 assignments constitute our Quarter 1 project that are designed to help us learn how to construct a graph from the data, produce graph statistics, build graph machine learning model architectures, and understand the application and importance of an efficient graph machine learning algorithm when it comes to chip design.

---

## Project Description

As mentioned, there are 2 assignments that make up our quarter 1 project. For more information about each assignment, refer to the assignment-specific `README.md` and the `docs\` directory inside the assignment's directory.

### Assignment 1

In this assignment, we are tasked with finding the busiest subway stations and connections in the New York MTA subway system via graph analysis. We do this by constructing a graph representation from tabular datasets and using this graph representation to derive some graph statistics, such as highest node degrees and highest edge weights, to solve the problem. The goal of this assignment is to familiarize ourselves with utilizing network graph Python libraries like [*NetworkX*](https://networkx.org) for graph construction and analysis.

### Assignment 2

In this assignment, we are asked to reimplement the core graph neural network model of a [paper](https://arxiv.org/abs/2404.00477) (i.e. De-HNN). We compare the benchmark results of our reimplementation to the one presented in the paper to check if the results line up. The goal of this assignment is to learn how to do graph embeddings, feature extraction for graph nodes and edges, and updates of graph features (i.e. message passing) in the process of reproducing a paper on graph neural network machine learning.

---

## Setup Instructions

### Step 1: Clone the repository

Clone this repository and cd into the cloned directory:

```
git clone https://github.com/hanhoangia/DSC180A-Q1.git
cd DSC180A-Q1
```

### Step 2: Download the Data

#### For Assignment 1

Download the MTA ridership dataset [here](https://drive.google.com/drive/folders/1fV47SWGv5_AFPR_gRfvK1ra1LfSFCgOw) then place the downloaded file into the `assignment1/data` folder and renamed it to `ridership.csv`.

#### For Assignment 2

There are 2 options (raw and processed dataset):

- *Processed dataset*: Download [here](https://zenodo.org/records/10795280?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk5NjM2MzZiLTg0ZmUtNDI2My04OTQ3LTljMjA5ZjA3N2Y1OSIsImRhdGEiOnt9LCJyYW5kb20iOiJlYzFmMGJlZTU3MzE1OWMzOTU2MWZkYTE3MzY5ZjRjOCJ9.WifQFExjW1CAW0ahf3e5Qr0OV9c2cw9_RUbOXUsvRbnKlkApNZwVCL_VPRJvAve0MJDC0DDOSx_RLiTvBimr0w). Extract the files from the zip file and place the extracted files into the `assignment2/data` folder. 
- *Raw dataset*: Download [here](https://drive.google.com/file/d/1Scq35gvCQvIMrmthGs7MUhc8c1VZ8ZwN/view). Extract the zip into the `assignment2/data` folder. To process the raw data, cd into `de_hnn/data` folder and run `./run_all_python_scripts.sh` from the command line.

### Step 3: Setup Development Environment

**Note**: This project is developed and run in a Conda environment. Please follow the instructions below to setup the environment.

#### Step 3.1 Install Miniconda (If Uninstalled)

Follow the instructions [here](https://docs.anaconda.com/miniconda/install/) based on the specs of your machine.

#### Step 3.2 Create and Activate Conda Environment

```bash
conda env create -f environment.yml
conda activate DSC180A-B12
```

**Note**: When the first command is run, it automatically creates and enables a Python kernel dedicated for the new environment.

---

## Repo Structure

- `README.me`: Overview of the quarter 1 project, consists of assignment 1 and assignment 2, and reproducing instructions for the project.

- `environment.yml`: List of Conda and Pip dependencies required for the project.
- `assignment1/`
  - `data/`: Stores the relevant datasets of the assignment in CSV format.
  - `docs/`: Contains the assignment instructions document.
  - `notebooks/`: Contains the interactive code in Jupyter Notebook that produces the outputs for the assignment.
  - `outputs/`: Contains generated results as answers for the assignment.
  - `src/`: Contains the source code of the project. The source code includes functions for handling the data and graph statistics, as well as the data objects.
- `assignment2/`
  - `data/`: Stores the data of the assignment.
  - `docs/`: Contains the data description files.
  - `notebooks/`: Contains the interactive code in Jupyter Notebook that produces the outputs for the assignment.
  - `outputs/`: Contains outputs for the assignment, including data exploration plots and benchmark tables.
    - `plots/`: Contains data exploration visualization plots.
  - `src/`: Contains the source code of the project. The source code includes functions and classes for the data loader, graph machine learning model, training and evaluation.

- `.gitignore`:  Specify which files or directories Git should *ignore* when tracking changes in a repository.

