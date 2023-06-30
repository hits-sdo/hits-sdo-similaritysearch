# Environment Setup

Conda is a popular open-source package management system and environment management system for installing, configuring, and managing software packages in various programming languages such as Python, R, and others. Conda provides a number of benefits, including:

1. Package management: Conda simplifies the process of installing and managing packages by automatically resolving dependencies and conflicts, ensuring that all required packages are available and up to date.

2. Environment management: Conda allows you to create isolated environments, each with its own set of dependencies, packages, and Python version. This helps you to keep your projects organized, reproducible, and easily shareable.

3. Cross-platform compatibility: Conda is designed to work across multiple platforms, including Windows, Linux, and macOS. This means that you can easily share your environments and packages across different operating systems.

4. Easy installation: Conda can be easily installed on your system, either as part of the Anaconda distribution or as a standalone package. Once installed, you can use conda to manage your packages and environments.

# Getting Started

## Install Conda
Miniconda is a minimal installer for conda. To install refer to the documentation [here](https://docs.conda.io/en/latest/miniconda.html). Depending on your operating system you can select from several of the Miniconda Installer links featured in their website.

## Setting Up the Path in .bashrc
Download miniconda3 to a new directory. Change permissions to make the .sh file executable and run. Your computer will suggest a path to install miniconda in your home directory user profile, accept and add the path to this directory (specifically to profile.d) to your .bashrc file.

PWD=<PATHTO>/miniconda3/etc/profile.d
echo ". ${PWD}/conda.sh" >> ~/.bashrc

Close and reopen your terminal to get started.
## The Conda Environment
A Conda environment is a self-contained directory that contains a specific collection of software packages and their dependencies that are installed within an isolated environment. Each Conda environment can have its own set of Python or other programming language versions, libraries, and packages that are independent of other environments on the same system. Conda environments are useful for managing different versions of software packages, avoiding conflicts between dependencies, and creating reproducible research environments.

### Creating a Conda Environment
There are two ways of creating a conda environment:
1. From a file
2. Naming an environment with the desired dependencies

#### Naming an environment
`conda create --name <ENV_NAME>`

To create the environment with specific dependencies:

`conda create --name <ENV_NAME> python=3.9`

To activate the environment and start installing packages and running code in the environment:

`conda activate <ENV_NAME>`

#### Creating an Environment from File

In order to create a conda environment from file, the file must be written as a yaml file. Traverse to the directory that the file exists in and then run the following command:

`conda env create -f <ENV_FILE>`

#### Updating an Environment
To update the conda environment upon installation of an additional package type:

`conda env update -f <ENV_FILE>`

#### Exporting an Environment
To update the corresponding yaml file to reflect this change export the conda environment:

`conda env export -f <ENV_FILE>`

### Deactivating an Environment
To deactivate the current conda environment and return to your base settings:

`conda deactivate`

#### Deleting an Environment
To delete the conda environment:

`conda remove -n <ENV_NAME> --all`
