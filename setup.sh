#!/bin/bash

# --- Configuration ---
CONDA_INSTALL_PATH="$HOME/miniconda3"
WORKSPACE_DIR="$HOME/workspace"
CONDA_INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
CONDA_INSTALLER_NAME="Miniconda3-latest-Linux-x86_64.sh"
CONDA_ENV_NAME="ling_adv"
PYTHON_VERSION="3.10.16"

echo "--- Audio Linguistic Adversarial Research Setup Script ---"
echo "This script will set up your development environment."
echo ""

# --- 1. Install Miniconda (if not already installed) ---
if [ ! -d "$CONDA_INSTALL_PATH" ]; then
    echo "Miniconda not found at $CONDA_INSTALL_PATH. Installing Miniconda..."
    mkdir -p "$WORKSPACE_DIR"

    # Download installer
    if [ ! -f "$WORKSPACE_DIR/$CONDA_INSTALLER_NAME" ]; then
        echo "Downloading Miniconda installer..."
        wget -O "$WORKSPACE_DIR/$CONDA_INSTALLER_NAME" "$CONDA_INSTALLER_URL"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to download Miniconda installer. Please check your internet connection."
            exit 1
        fi
    else
        echo "Miniconda installer already downloaded."
    fi

    # Run installer
    echo "Running Miniconda installer. Installing silently to $CONDA_INSTALL_PATH."
    bash "$WORKSPACE_DIR/$CONDA_INSTALLER_NAME" -b -p "$CONDA_INSTALL_PATH"
    if [ $? -ne 0 ]; then
        echo "Error: Miniconda installation failed."
        exit 1
    fi

    # Initialize conda
    echo "Initializing Conda for bash shell..."
    # Explicitly use the newly installed conda binary for initialization
    "$CONDA_INSTALL_PATH/bin/conda" init bash
    if [ $? -ne 0 ]; then
        echo "Warning: Conda initialization failed. You may need to manually run 'conda init bash' and 'source ~/.bashrc'."
    fi

    # Source the bashrc to make conda available in the current script session
    # We need to be careful here. If it was a fresh install, ~/.bashrc might not have the conda init lines yet
    # OR the sourcing needs to be done *after* init.
    # The most reliable way for the *current script* is to directly add to PATH and use the explicit command.
    # However, for 'conda activate' which uses shell functions, sourcing is often required.
    # We'll try sourcing, but also directly modify PATH for the script.

    # Option 1: Try sourcing (might not always work for activate in subshell)
    # echo "Attempting to source ~/.bashrc..."
    # source "$HOME/.bashrc" || source "$HOME/.profile" || echo "Warning: Could not source .bashrc or .profile."

    # Option 2: Directly add to PATH for *this script's execution* and rely on explicit 'conda run' or full path
    # For `conda activate` specifically, this can be tricky.
    # The recommended way for shell scripts to use conda environments is `conda run`.

    echo "Miniconda installed. For the current script session, we will attempt to source .bashrc."
    echo "You might still need to restart your terminal or run 'source ~/.bashrc' after this script for full Conda functionality."
    if [ -f "$HOME/.bashrc" ]; then
        source "$HOME/.bashrc"
    elif [ -f "$HOME/.profile" ]; then
        source "$HOME/.profile"
    else
        echo "Warning: Could not find .bashrc or .profile to source. Conda commands might not work directly."
    fi

    # Crucially, add conda's bin directory to the PATH for the *current script's execution*
    export PATH="$CONDA_INSTALL_PATH/bin:$PATH"
    echo "Added $CONDA_INSTALL_PATH/bin to PATH for this script session."

else
    echo "Miniconda already found at $CONDA_INSTALL_PATH. Skipping installation."
    # Ensure conda is initialized and available for the current script session
    if ! command -v conda &> /dev/null; then
        echo "Conda command not found in PATH, attempting to source .bashrc or .profile."
        if [ -f "$HOME/.bashrc" ]; then
            source "$HOME/.bashrc"
        elif [ -f "$HOME/.profile" ]; then
            source "$HOME/.profile"
        else
            echo "Warning: Could not find .bashrc or .profile to source. Conda commands might not work directly."
        fi
        # If still not found, add its bin to PATH for this script
        if ! command -v conda &> /dev/null; then
            export PATH="$CONDA_INSTALL_PATH/bin:$PATH"
            echo "Added $CONDA_INSTALL_PATH/bin to PATH for this script session."
        fi
    fi
fi

# Final check if conda is now available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda command not found after installation/initialization attempts."
    echo "Please ensure Conda is installed and initialized. You may need to manually run 'conda init bash' and 'source ~/.bashrc'."
    exit 1
fi

echo ""
echo "--- 2. Create and Activate Conda Environment ---"
# Check if environment already exists
if conda info --envs | grep -q "$CONDA_ENV_NAME"; then
    echo "Conda environment '$CONDA_ENV_NAME' already exists. Activating it."
    # Use `conda activate` if possible, but fallback to `conda run` for robustness
    # `conda activate` needs specific shell functions to be sourced.
    # `conda run` does not.
    if ! conda activate "$CONDA_ENV_NAME"; then
        echo "Warning: 'conda activate' failed. This might indicate incomplete shell initialization."
        echo "Proceeding by attempting to install packages using 'conda run' in the target environment."
        # If activate fails, we will rely on `conda run` later to execute commands within the environment.
        # This means the *current script session* won't be in the environment,
        # but the commands will execute *as if they were*.
        # We still need the env to exist.
    else
        echo "Successfully activated Conda environment: $CONDA_ENV_NAME"
    fi
else
    echo "Creating new Conda environment: $CONDA_ENV_NAME with Python $PYTHON_VERSION"
    conda create --name "$CONDA_ENV_NAME" python=="$PYTHON_VERSION" -y
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create Conda environment '$CONDA_ENV_NAME'."
        exit 1
    fi
    echo "Attempting to activate new Conda environment: $CONDA_ENV_NAME"
    if ! conda activate "$CONDA_ENV_NAME"; then
        echo "Warning: 'conda activate' failed immediately after creation. This is unusual."
        echo "Proceeding by attempting to install packages using 'conda run' in the target environment."
    else
        echo "Successfully activated Conda environment: $CONDA_ENV_NAME"
    fi
fi

echo ""
echo "--- 3. Install Python Dependencies ---"
# Ensure we are in the correct directory for pip installs
CURRENT_DIR=$(pwd)
REPO_ROOT=$(dirname "$0") # Assuming setup.sh is at the root of the repo
cd "$REPO_ROOT" || { echo "Error: Failed to change to repository root directory."; exit 1; }

# Function to run commands in the conda environment
run_in_conda_env() {
    # If the env is already activated in this shell, just run the command
    if [[ "$CONDA_DEFAULT_ENV" == "$CONDA_ENV_NAME" ]]; then
        "$@"
    else
        # Otherwise, use `conda run` which is more robust in scripts
        conda run --no-capture-output -n "$CONDA_ENV_NAME" "$@"
    fi
}

# Install fairseq editable
if [ -d "fairseq" ]; then
    echo "Installing fairseq in editable mode..."
    run_in_conda_env pip install --editable fairseq/.
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install fairseq. Ensure 'fairseq/' directory exists and is valid."
        exit 1
    fi
else
    echo "Warning: 'fairseq/' directory not found. Skipping fairseq installation."
    echo "If fairseq is required, please clone/add it to the 'fairseq/' directory."
fi

# Install other requirements
echo "Installing other Python packages from requirements.txt..."
if [ -f "requirements.txt" ]; then
    run_in_conda_env pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install packages from requirements.txt."
        exit 1
    fi
else
    echo "Warning: requirements.txt not found. Skipping general Python package installation."
fi

run_in_conda_env pip install TTS

cd "$CURRENT_DIR" # Go back to original directory

echo ""
echo "--- 4. Install FFmpeg ---"
# ... (FFmpeg section remains the same as it doesn't depend on conda env)
# Check if ffmpeg is already installed
if command -v ffmpeg &> /dev/null; then
    echo "FFmpeg is already installed. Skipping FFmpeg installation."
else
    echo "Installing FFmpeg. This may require sudo password."
    # Update package lists
    sudo apt-get update -y
    if [ $? -ne 0 ]; then echo "Warning: apt-get update failed. Continuing with FFmpeg installation."; fi

    # Add required PPA (might be deprecated or cause issues on newer OS versions, consider just `apt install ffmpeg`)
    echo "Adding ppa:mc3man/trusty-media repository..."
    sudo apt-get install software-properties-common -y
    sudo add-apt-repository ppa:mc3man/trusty-media -y
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to add ppa:mc3man/trusty-media. FFmpeg installation might fail."
        echo "On newer Ubuntu versions, FFmpeg might be directly available via 'sudo apt install ffmpeg'."
    fi

    sudo apt-get update -y # Update again after adding PPA
    if [ $? -ne 0 ]; then echo "Warning: apt-get update failed after PPA. Continuing with FFmpeg installation."; fi

    # Install ffmpeg
    echo "Installing ffmpeg..."
    sudo apt-get install ffmpeg -y
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install FFmpeg. Please try installing manually."
        echo "  sudo apt-get install ffmpeg"
        echo "Or check the PPA if you are on an older Ubuntu version."
        exit 1
    fi
fi

echo ""
echo "--- Setup Complete! ---"
echo "Your Conda environment '$CONDA_ENV_NAME' has been created and packages installed."
echo "To activate it in a new terminal, run: conda activate $CONDA_ENV_NAME"
echo "You can now navigate to your project directory and run your scripts."