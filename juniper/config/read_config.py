import shlex
import numpy as np

def read_config(path_to_config_file):
    """Parses config files to create a dictionary of inputs.
    Credit V.A. Boehm from ExoTiC-UVIS.

    Args:
        path_to_config_file (str): Path to the .berry file that is being read.

    Returns:
        dict: instructions for pipeline.py to follow.
    """
    # Open the dictionary.
    config = {}

    # Read out all lines.
    with open(path_to_config_file,mode='r') as f:
        lines = f.readlines()

    # Process all lines.
    for line in lines:
        line = shlex.split(line, posix=False)
        # Check if it is empty line or comment line and pass.
        if len(line) == 0:
            continue
        if line[0] == '#':
            continue
        key = line[0]
        # Param may have spaces, so we need to keep going with it.
        param = line[1]
        i = 2
        while "#" not in line[i]:
            param = ''.join([param,line[i]])
            i += 1
        try:
            param = eval(param)
        except:
            pass
        config[key] = param

    return config