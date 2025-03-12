# Walkthrough
This tutorial walks you through how to setup and run SiMiLe-M.


## Table of Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Step by step guide](#steps)
4. [Troubleshooting](#faq)

<a name="requirements"></a>
## Requirements

### Input data

- [ ] WIP

### Expected results/output

- [ ] WIP

<a name="installlation"></a>
## Installation

### Local (without Singularity image)

This method is simpler to get started since it doesn't require `Singularity`

- Pros: Good for quick testing with small datasets on your local machine.
- Cons: For larger dataset when you need to run it on HPC with and by following steps to build [Using Singularity Image](#build-singularity-image)

- Run locally using `uv` (`uv` is `rustup`/`juliaup`/`npm` but for Python)
  1. setup `uv` by following the official instruction here: [uv: Installation](https://docs.astral.sh/uv/getting-started/installation/)
  2. then, after ran the following, `uv` will automatically create virtual environment and install dependencies.
    ```bash
    uv sync
    ```
  2. finally, you may able to trigger the main `SiMiLe-M`'s command line interface by:
    ```bash
    uv run run.py -h
    ```

<a name="build-singularity-image"></a>
### Using Singularity Image

1. Follow instruction here and build [Singularity ](https://apptainer.org/user-docs/master/quick_start.html#quick-installation-steps)

2. Then build Singularity container:
  ````
  singularity build singularity/similem.sif singularity/similem.def
  ````

3. Finally, test container. the following command should print the help dialog:
  ```
  singularity exec singularity/similem.sif python run.py -h
  ```

<a name="steps"></a>
## Step-by-step guide

### Step 1

- [ ] WIP: First, do ...
  ```bash
  command
  ```
  [ ] [optional]The output should look like
  ```bash
  some result
  ```

<a name="faq"></a>
## Troubleshooting
### Contact info
### Creating issue on project repository
