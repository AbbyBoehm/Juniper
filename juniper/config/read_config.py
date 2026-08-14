import shlex
import numpy as np

def read_config(path_to_config_file):
    """Parses config files to create a dictionary of inputs.
    Credit V.A. Boehm from HUSTLE-tools.

    Args:
        path_to_config_file (str): Path to the .berry file that is being read.

    Returns:
        dict: instructions for pipeline.py to follow.
    """
    # Open the dictionary.
    config = {}

    # Define certain keys as special. These are keys for which multiple
    # similar entries are expected to appear.
    planet_keys = ["rp","fp","t_prim","t_seco","period",
                   "aor","incl","ecc","longitude"]
    planet_prior_keys = [key+"_prior" for key in planet_keys]
    for key in planet_prior_keys:
        planet_keys.append(key)
    planet_ptype_keys = [key+"_ptype" for key in planet_keys]
    for key in planet_ptype_keys:
        planet_keys.append(key)
    flare_keys = ["A","B","C","Dr","Ds","Fr","E",]
    flare_prior_keys = [key+"_prior" for key in flare_keys]
    for key in flare_prior_keys:
        flare_keys.append(key)
    flare_ptype_keys = [key+"_ptype" for key in flare_keys]
    for key in flare_ptype_keys:
        flare_keys.append(key)
    systematics_keys = ["poly","poly_order",
                        "piecewise","piecewise_n","piecewise_t","piecewise_os",
                        "dilution",
                        "mirrortilt","n_tilt_events",
                        "disp_detrend","disp_order",
                        "spatial_detrend","spatial_order",
                        "width_detrend","width_order",
                        "singleramp","doubleramp"]
    
    # Keep track of how many times we have seen this key appear.
    # Allows us to assign number IDs to each instance.
    seen_this_planet_key = {}
    for key in planet_keys:
        seen_this_planet_key[key] = 0
    seen_this_flare_key = {}
    for key in flare_keys:
        seen_this_flare_key[key] = 0

    # Read out all lines.
    with open(path_to_config_file,mode='r') as f:
        lines = f.readlines()

    # Init the event_ID variable.
    event_ID = 0

    # Process all lines.
    for line in lines:
        line = shlex.split(line, posix=False)
        # Check if it is empty line or comment line and pass.
        if len(line) == 0:
            continue
        if line[0] == '#':
            continue
        
        # It's a useful line. Take the dict key.
        key = line[0]

        # There is a chance this could be a new block of planets+flares+systematics
        if key == "event_type":
            # When the event_type key is encountered, we need to reset the planet and flare key counter.
            # Each planet and flare key is going to have its own number of recurrences within each block
            # for each parallelised spectrum.
            for reset_key in planet_keys:
                seen_this_planet_key[reset_key] = 0
            for reset_key in flare_keys:
                seen_this_flare_key[reset_key] = 0

            # Also, we need to update the event ID spectrum each time we start a new block.
            event_ID += 1

        # Count planet keys, with event ID attached.
        if key in planet_keys:
            # For planets and flares, the same key may appear multiple times.
            # e.g. if you fit two planets in transit, rp and rp_prior will both appear twice.
            # So we have to add numbers to keep them distinct.
            seen_this_planet_key[key] += 1 # keep track of how many times we've seen the special key.
            key = "{}{:.0f}_{:.0f}".format(key, seen_this_planet_key[key], event_ID) # assign tracker number to the key.

        # And count flare keys, with event ID attached.
        if key in flare_keys:
            # For flares, the same key may appear multiple times.
            # e.g. if you fit two flares, each will have its own amplitude and timing.
            # So we have to add numbers to keep them distinct.
            seen_this_flare_key[key] += 1 # keep track of how many times we've seen the special key.
            key = "{}{:.0f}_{:.0f}".format(key, seen_this_flare_key[key], event_ID) # assign tracker number to the key.

        # Param may have spaces, so we need to keep going with it.
        param = line[1]
        i = 2
        while "#" not in line[i]:
            param = ''.join([param,line[i]])
            i += 1

        if 'np.concat' in param or 'np.concatenate' in param:
            # Handling for concatenated arrays.
            repl_param = str.replace(param,'np.concatenate','')
            repl_param = str.replace(repl_param,'np.concat','')
            repl_param = eval(repl_param)
            unpacked_items = []
            for unpacked_item in repl_param:
                if isinstance(unpacked_item,tuple):
                    # Concatenate it.
                    unpacked_items.append(np.concatenate(unpacked_item))
                else:
                    # Take as is.
                    unpacked_items.append(unpacked_item)
            param = unpacked_items

        try:
            # If the parameter is an evaluatable statement (a bool, a list, a numpy object, etc.), make it so.
            param = eval(param)
        except:
            # It was just a string after all, or it has already been evaluated.
            pass

        # Note that event_type, n_planets, and n_flares are all event_ID-dependent. So...
        if key in ("event_type","n_planets","n_flares"):
            key = "{}_{:.0f}".format(key,event_ID)

        # Systematics are also event_ID-dependent variables. So...
        if key in systematics_keys:
            key = "{}_{:.0f}".format(key,event_ID)
        
        # And put it in the dictionary.
        config[key] = param
    
    return config