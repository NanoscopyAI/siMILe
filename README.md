# siMILe User Guide
<img width="600" height="1772" alt="toc-figure-simile-updated" src="https://github.com/user-attachments/assets/337a4682-1955-40c6-b0b5-0258107c9f58" />

**siMILe** is a Multiple Instance Learning (MIL) algorithm that extends the MILES framework with adversarial erasing and symmetric classification for weakly supervised learning tasks.

## What Does siMILe Do?

1. Takes CSV files organized into two labeled conditions (bags)
2. Uses MILES (Multiple-Instance Learning via Embedded instance Selection) to map bags into an instance-based feature space
3. Employs adversarial erasing to iteratively identify and remove confidently classified instances
4. Supports both symmetric (bidirectional) and single-sided classification modes
5. **Outputs original CSVs with instance-level classifications** in a new "classification" column

**Applications:** Biological imaging analysis, document classification, drug activity prediction, and other scenarios where labels are available at the group level but needed at the individual level.

**Paper:** [Adversarial erasing enhanced multiple instance learning (siMILe): Discriminative identification of oligomeric protein structures in single molecule localization microscopy](https://www.cs.sfu.ca/~hamarneh/ecopy/aisy2026.pdf)

```bibtex
@ARTICLE{aisy2026,
   AUTHOR       = {Christian Hallgrimson and Y. Lydia Li and Claire A. Shou and 
                   Ben Cardoen and John Lim and Timothy H. Wong and Ismail M. Khater and 
                   Ivan Robert Nabi and Ghassan Hamarneh},
   JOURNAL      = {Advanced Intelligent Systems (AISY)},
   TITLE        = {Adversarial erasing enhanced multiple instance learning 
                  (siMILe): Discriminative identification of oligomeric protein 
                  structures in single molecule localization microscopy},
   YEAR         = {2026},   
   PDF          = {https://www.cs.sfu.ca/~hamarneh/ecopy/aisy2026.pdf},
   DOI          = {10.1002/aisy.202501159},
}


> **Note:** This repository distributes siMILe as pre-compiled Python bytecode (.pyc) for Python 3.9. Source code is not included. Python 3.9 is required.

---

## Table of Contents
1. [Installation](#installation)
2. [Quick Start Guide](#quick-start)
3. [Understanding Your Results](#results)
4. [Command Reference](#commands)
5. [Troubleshooting](#troubleshooting)

---

<a name="installation"></a>
## Installation

### Option 1: Local Installation (Simple)

**Requirements:** Python 3.9 (required — bytecode is version-specific). Works on Windows, Mac, or Linux.

1. **Install `uv`** (a Python package manager):
   - Visit: https://docs.astral.sh/uv/getting-started/installation/
   - Or run: `curl -LsSf https://astral.sh/uv/install.sh | sh` (Mac/Linux)

2. **Navigate to siMILe directory** and install dependencies:
   ```bash
   cd path/to/siMILe
   uv sync
   ```

3. **Test the installation:**
   ```bash
   uv run run.pyc -h
   ```
   You should see a help message listing available options.

### Option 2: HPC with Apptainer (Advanced Users)

Especially useful if you're using a High-Performance Computing cluster:

1. **Load Apptainer module** (if available on your system):
   ```bash
   module load apptainer/1.3.5
   ```

2. **Build the container:**
   ```bash
   apptainer build apptainer/similem.sif apptainer/similem.def
   ```

3. **Test the container:**
   ```bash
   apptainer exec apptainer/similem.sif python run.pyc -h
   ```

**Note:** For all commands below, if using Apptainer, replace `uv run run.pyc` with `apptainer exec apptainer/similem.sif python /app/run.pyc`

---

<a name="quick-start"></a>
## Quick Start Guide

### Step 1: Prepare Your Data

Your CSV files should have:
- **Numeric columns:** Your measurements/features (these will be used for classification)
- **Non-numeric columns:** Any text or categorical columns will be automatically ignored

**Important:** siMILe uses **only numeric columns** for analysis. All non-numeric columns (text, categories, etc.) are excluded during data loading. Also, rows containing missing values will be dropped.

**Example CSV format:**

| feature_1 | feature_2 | feature_3 | sample_id | class |
|-----------|-----------|-----------|-----------|-------|
| 76.0      | 3.57      | 14.15     | A001      | c1    |
| 54.0      | 4.45      | 14.74     | A002      | c1    |
| 68.2      | 5.12      | 16.23     | B001      | c2    |

In this example, `feature_1`, `feature_2`, and `feature_3` will be used for classification. The `sample_id` and `class` columns will be ignored.

### Step 2: Organize Your Data

Organize your CSV files into train/valid/test directories:

```
my_data/
├── train/
│   ├── condition_A.csv    # or multiple files: sample1.csv, sample2.csv, etc.
│   └── condition_B.csv
├── valid/
│   ├── condition_A.csv
│   └── condition_B.csv
└── test/
    ├── condition_A.csv
    └── condition_B.csv
```

### Step 3: Create a Configuration File

Create a `.ini` file (or copy and modify `example/simple_case.ini`):

```ini
[DATA]
name = MY_EXPERIMENT
base_path = /path/to/my_data/

# Optional: Set hyperparameters (can be overridden via command-line)
C = 1
sigma = 1000
bagsize = 50
minacc = 0.8

[TRAIN]
0 = train/condition_A.csv
1 = train/condition_B.csv

[VALID]
0 = valid/condition_A.csv
1 = valid/condition_B.csv

[TEST]
0 = test/condition_A.csv
1 = test/condition_B.csv
```

**Configuration Sections:**

**[DATA]** - Required settings:
- `base_path` - Base directory for relative file paths
- `name` - Experiment name (used in output folder naming)
- `C`, `sigma`, `bagsize`, `minacc` - Optional hyperparameters (can override via CLI)

**[TRAIN], [VALID]** - Required data file paths:
- `0` - Path to condition A files for this split
- `1` - Path to condition B files for this split
- Paths are relative to `base_path`
- Support glob patterns: `train/*.csv` or `train/**/*.csv`

**[TEST]** - Optional test data (only needed if using `--use_test 1`):
- Same format as [TRAIN] and [VALID]
- Can be omitted entirely if you don't need a separate test set

<details>
<summary><b>Advanced: Feature Selection in Config</b></summary>

You can optionally specify which columns to use from your CSV files:

```ini
[DATA]
name = MY_EXPERIMENT
base_path = ./data

# Optional: Select specific features
all_feature_names = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
used_feats = ["feature_1", "feature_3", "feature_5"]

[TRAIN]
0 = train/condition_A.csv
1 = train/condition_B.csv
# ... etc
```

- `all_feature_names` - JSON array of all numeric column names in your CSVs (in order)
- `used_feats` - JSON array of which features to actually use for classification

This is useful if you have many features but only want to use a subset. If omitted, all numeric columns are used.

</details>

### Step 4: Run siMILe

Basic command:
```bash
uv run run.pyc --config path/to/my_config.ini
```

The hyperparameters default to values in your config file. Override them via CLI if needed:
```bash
uv run run.pyc --config path/to/my_config.ini --C 0.1 --sigma 10 --bagsize 50
```

**What these parameters mean:**
- `--config`: Path to your configuration file
- `--C 0.1`: Regularization (lower = more flexible, try 0.1 to 50)
- `--sigma 10`: Similarity threshold (lower = stricter, try 10 to 1000000)
- `--bagsize 50`: Group size for learning (larger = more stable but slower, try 25-500)

### Step 5: Test with Sample Data

Try siMILe with built-in example data to verify everything works:

```bash
# Run siMILe on example data included in the repo
uv run run.pyc --config example/simple_case.ini
```

This uses the hyperparameters defined in the config file (C=1, sigma=10, bagsize=2, minacc=0.95).

**About the example data:**
The example dataset contains 4 numeric features and two classes. Class 0 (condition A) has discriminative instances with large values in feature 4, while class 1 (condition B) has discriminative instances with large values in feature 2. Background instances in both classes have small values across all features.

The CSV files include marker columns (`*` for class 0 discriminative instances, `**` for class 1 discriminative, `-` for background) so you can compare results against ground truth.

This should complete quickly and create output in the `OUTPUT/` folder. You can verify results by comparing the `classification` column against the marker column. Note that this minimal dataset with default parameters won't produce perfect results; it's intended as a sanity check that the installation works. For your own data, you'll want to tune hyperparameters appropriately.

---

<a name="results"></a>
## Understanding Your Results

Results are saved to: `OUTPUT/siMILe_output_[experiment_name]/results/[timestamp]/`

### Primary Output: Classified CSVs

**Location:** `OUTPUT/.../results/[timestamp]/classified_csvs/`

These are **your original CSV files with two new columns added**:

| feature_1 | feature_2 | feature_3 | split | classification |
|-----------|-----------|-----------|-------|----------------|
| 76.0      | 3.57      | 14.15     | train | 0              |
| 54.0      | 4.45      | 14.74     | valid | 1              |
| 68.2      | 5.12      | 16.23     | valid | unclassified   |
| 72.0      | 2.39      | 25.41     | test  | unclassified   |

**Column descriptions:**

**`split`** - Which dataset split this row belonged to:
- `train`: Used to train the model (less reliable for evaluation)
- `valid`: Validation set - **THESE ARE YOUR MOST RELIABLE RESULTS**
- `test`: Test set (if used) - also reliable for evaluation
- `unknown`: Split information not available

**`classification`** - The predicted condition:
- `0` or `1`: The tag number of the assigned condition (discriminative instance)
- `unclassified`: Not confidently classified (likely common to both conditions)

**IMPORTANT:** Focus on rows with `split='valid'` or `split='test'` for evaluation. Train split classifications are less reliable because the model was trained on that data.

**Quick tip for filtering results:**
- In Excel/Sheets: Use the filter feature on the `split` column, select only "valid" and "test"
- In Python: `df[df['split'].isin(['valid', 'test'])]`
- In R: `df[df$split %in% c('valid', 'test'), ]`

### Split CSVs

**Location:** `OUTPUT/.../results/[timestamp]/split_csvs/`

These aggregate all instances by split into three files: `train.csv`, `valid.csv`, `test.csv`. Each includes:
- All original feature columns
- `source_file`: The original file path this row came from
- `classification`: The predicted condition (0, 1, or unclassified)

### Other Output Files

<details>
<summary>Click to expand details on additional outputs</summary>

#### Logs (`logs/`)
- Contains detailed logs of the training process
- Includes training progress, accuracy metrics, and any warnings or errors
- Useful for debugging if something goes wrong

The classified CSVs in `classified_csvs/` are the primary output and contain all the information you need for downstream analysis.

</details>

### What the Terminal Output Means

During training, you'll see logs like:
```
Training Symmetric classifier with AE | C=0.1, sigma=10, bagsize=50, min_acc=0.75
Bag accuracy: train=0.892, valid=0.845
Classified instances (train): 45 from 0, 38 from B (total: 83)
Training complete
Results saved to: OUTPUT/.../results/[timestamp]
Exported 6 classified CSV files (234 classified instances)
```

**Key metrics:**
- **Bag accuracy**: How well the model distinguishes the two conditions (higher is better, >0.75 is good)
- **Classified instances**: Number of rows confidently assigned to each condition
- **Exported files**: Number of output CSVs created

---

<a name="commands"></a>
## Command Reference

### Required Arguments

| Argument   | Description                                              | Example        |
|------------|----------------------------------------------------------|----------------|
| `--config` | Path to your configuration file                          | `path/to/my.ini` |

### Hyperparameter Arguments (Optional, Override Config)

These parameters can be set in the config file under `[DATA]`. CLI values override config values.

| Argument     | Default | Description                                           | Typical Range |
|--------------|---------|-------------------------------------------------------|---------------|
| `--C`        | 50      | Regularization strength (lower = more flexible)       | 0.1 - 50      |
| `--sigma`    | 1000    | Similarity threshold (lower = stricter matching)      | 10 - 100000   |
| `--bagsize`  | 100     | Number of instances grouped together for learning     | 50 - 500      |
| `--minacc`   | 0.75    | Minimum accuracy required to continue training        | 0.6 - 0.9     |

### Output Control

| Argument    | Default     | Description                                |
|-------------|-------------|--------------------------------------------|
| `--output`  | `./OUTPUT`  | Directory where results will be saved      |
| `--expname` | `""`        | Custom name to add to output folder        |

### Advanced Options

<details>
<summary>Click to expand advanced parameters</summary>

| Argument     | Default | Description                                              |
|--------------|---------|----------------------------------------------------------|
| `--AE`       | 1       | Enable adversarial erasing (1=enabled, 0=disabled)       |
| `--SYM_C`    | 1       | Enable symmetric classification (1=enabled, 0=disabled)  |
| `--seed`     | 42      | Random seed for reproducible results                     |
| `--max_iter` | 0       | Maximum adversarial erasing iterations (0=no limit)      |
| `--use_test` | 0       | Predict on test split instead of valid split (1=enabled) |

**Note:** Most users should keep default values for these parameters. Adversarial erasing (`--AE`) and symmetric classification (`--SYM_C`) are core features of siMILe and are enabled by default.

**Using `--use_test`:** By default, siMILe trains on the train split and evaluates/classifies on the valid split. If you want to evaluate on the test split instead, use `--use_test 1`. This is useful when you've tuned your hyperparameters on the validation set and want final results on held-out test data.

</details>

---

<a name="troubleshooting"></a>
## Troubleshooting

### "No configuration settings found"
**Problem:** Config file not found or empty
**Solution:** Check that your config file path is correct and the file contains `[DATA]`, `[TRAIN]`, and `[VALID]` sections

### "Bag size too large for the current dataset"
**Problem:** Your dataset is too small for the specified bag size
**Solution:** Reduce `--bagsize` to a smaller value (try 10-20 for small datasets)

### "FileNotFoundError" when loading data
**Problem:** CSV files specified in config can't be found
**Solution:**
- Check that paths in your `.ini` file are correct
- Make sure paths are relative to `base_path` in your config
- Verify the CSV files actually exist at those locations

### Getting lots of "unclassified" results
**Problem:** Model is too conservative in making predictions
**Solution:**
- Increase `--bagsize` for more stable predictions
- Adjust `--sigma` to be more permissive
- Check if your data truly has distinguishable patterns between conditions

### Out of memory errors
**Problem:** Dataset or bag size is too large for available RAM
**Solution:**
- Reduce `--bagsize` to use smaller groups of instances
- Reduce the size of your dataset by splitting it into smaller subsets
- Consider using a machine with more RAM or switching to HPC


---

## Tips for Best Results

1. **Start simple:** Use default parameters first, then tune if needed
2. **Check the logs:** Terminal output shows accuracy - aim for >0.75
4. **Experiment:** Try different parameter combinations if results aren't satisfactory
5. **Reproducibility:** Use the same `--seed` value to get consistent results, the default is to maintain the seed 42.
