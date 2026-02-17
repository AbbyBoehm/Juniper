import os
import glob
import shutil
import numpy as np

from juniper.config import *
from juniper.util import *
from juniper.stage1 import *
from juniper.stage2 import *
from juniper.stage3 import *
from juniper.stage4 import *
from juniper.stage5 import *
from juniper.stage6 import *


def run_pipeline(config_folder,stages=(1,2,3,4,5,6,)):
    """Wrapper for the Juniper pipeline.

    Args:
        config_folder (str): path to the folder that stores all config files.
        stages (tuple, optional): tuple of ints which specify the stages you want to run. Defaults to (1,2,3,4,5,6,).
    """
    print("Juniper pipeline running with following stages:", stages)
    print(".berry configuration files will be sourced from:", config_folder)
    ### Run Juniper Stage 1: JWST Detector1Pipeline and 1/f Corrections
    if 1 in stages:
        # Open the config dictionary.
        s1_config_path = glob.glob(os.path.join(config_folder,'s1_*'))[0]
        s1_config = read_config.read_config(s1_config_path)

        # Set up run name and define directories.
        project_dir = s1_config["toplevel_dir"]
        input_dir = os.path.join(project_dir,s1_config["input"])
        input_run = s1_config["input_run"]
        if input_run:
            # Add an extra sub-folder to select the correct input sub-folder.
            input_dir = os.path.join(project_dir,os.path.join(s1_config["input"],input_run))
        output_dir = os.path.join(project_dir,s1_config["output"])
        output_run = s1_config["output_run"]
        if output_run:
            # Add an extra sub-folder to separate this run from other runs.
            output_dir = os.path.join(project_dir,os.path.join(s1_config["output"],output_run))
        diagnosticplots_dir = os.path.join(output_dir, s1_config["diagnostics"])

        # Open all needed directories.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(diagnosticplots_dir):
            os.makedirs(diagnosticplots_dir)

        # Find files.
        files = sorted(glob.glob(os.path.join(input_dir,"*uncal.fits")))
        outfiles = [None for i in files] # use default names and change uncal to rateints, that's all.
        if s1_config["rename"]:
            # Set up new outfile names and also change uncal to rateints.
            outfiles = ['{}_f{}_rateints'.format(s1_config["rename"],i) for i, f in enumerate(files)]

        # Set up crds cache.
        os.environ["CRDS_PATH"] = "crds_cache" # if no path to crds_cache defined, set it up in the cwdir
        if s1_config["CRDS_PATH"]:
            os.environ["CRDS_PATH"] = s1_config["CRDS_PATH"] # set path to CRDS cache if you already have on
        elif not os.path.exists("crds_cache"):
            os.makedirs("crds_cache")

        # Process Stage 1.
        do_stage1(files,outfiles,output_dir,s1_config,diagnosticplots_dir)
        
        # Write the config dictionary out as a copy.
        config_outdir = os.path.join(output_dir,"configuration")
        if not os.path.exists(config_outdir):
            os.makedirs(config_outdir)
        config_name = "s1_juniper.berry"
        if output_run:
            config_name = "s1_{}_juniper.berry".format(output_run)
        shutil.copy(s1_config_path,os.path.join(config_outdir,config_name))
    
    ### Run Juniper Stage 2: Wavelength Solution and Curvature Correction
    if 2 in stages:
        # Open the config dictionary.
        s2_config_path = glob.glob(os.path.join(config_folder,'s2_*'))[0]
        s2_config = read_config.read_config(s2_config_path)

        # Set up run name and define directories.
        project_dir = s2_config["toplevel_dir"]
        input_dir = os.path.join(project_dir,s2_config["input"])
        input_run = s2_config["input_run"]
        if input_run:
            # Add an extra sub-folder to select the correct s1 run.
            input_dir = os.path.join(project_dir,os.path.join(s2_config["input"],input_run))
        output_dir = os.path.join(project_dir,s2_config["output"])
        output_run = s2_config["output_run"]
        if output_run:
            # Add an extra sub-folder to separate this run from other runs.
            output_dir = os.path.join(project_dir,os.path.join(s2_config["output"],output_run))
        diagnosticplots_dir = os.path.join(output_dir, s2_config["diagnostics"])

        # Open all needed directories.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(diagnosticplots_dir):
            os.makedirs(diagnosticplots_dir)

        # Find files.
        files = sorted(glob.glob(os.path.join(input_dir,"*rateints.fits")))
        fnames = [str.split(f,sep='/')[-1] for f in files]
        outfiles = [str.replace(f,'_rateints.fits','_calints') for f in fnames] # use default names and change rateints to calints, that's all.
        if s2_config["rename"]:
            # Set up new outfile names and also change rateints to calints.
            outfiles = ['{}_f{}_calints'.format(s2_config["rename"],i) for i, f in enumerate(files)]

        # Set up crds cache.
        os.environ["CRDS_PATH"] = "crds_cache/jwst_ops/" # if no path to crds_cache defined, set it up in the cwdir
        if s2_config["CRDS_PATH"]:
            os.environ["CRDS_PATH"] = s2_config["CRDS_PATH"] # set path to CRDS cache if you already have on
        elif not os.path.exists("crds_cache/jwst_ops/"):
            os.makedirs("crds_cache/jwst_ops/")

        # Process Stage 2.
        do_stage2(files,outfiles,output_dir,s2_config,diagnosticplots_dir)
        
        # Write the config dictionary out as a copy.
        config_outdir = os.path.join(output_dir,"configuration")
        if not os.path.exists(config_outdir):
            os.makedirs(config_outdir)
        config_name = "s2_juniper.berry"
        if output_run:
            config_name = "s2_{}_juniper.berry".format(output_run)
        shutil.copy(s2_config_path,os.path.join(config_outdir,config_name))
    
    ### Run Juniper Stage 3: Reduction
    if 3 in stages:
        # Open the config dictionary.
        s3_config_path = glob.glob(os.path.join(config_folder,'s3_*'))[0]
        s3_config = read_config.read_config(s3_config_path)

        # Set up run name and define directories.
        project_dir = s3_config["toplevel_dir"]
        input_dir = os.path.join(project_dir,s3_config["input"])
        input_run = s3_config["input_run"]
        if input_run:
            # Add an extra sub-folder to select the correct s2 run.
            input_dir = os.path.join(project_dir,os.path.join(s3_config["input"],input_run))
        output_dir = os.path.join(project_dir,s3_config["output"])
        output_run = s3_config["output_run"]
        if output_run:
            # Add an extra sub-folder to separate this run from other runs.
            output_dir = os.path.join(project_dir,os.path.join(s3_config["output"],output_run))
        diagnosticplots_dir = os.path.join(output_dir, s3_config["diagnostics"])

        # Open all needed directories.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(diagnosticplots_dir):
            os.makedirs(diagnosticplots_dir)

        # Find files.
        files = sorted(glob.glob(os.path.join(input_dir,"*calints.fits")))
        fnames = [str.split(f,sep='/')[-1] for f in files]
        outfiles = [str.replace(f,'_calints.fits','_reduced') for f in fnames] # use default names and change rateints to calints, that's all.
        if s3_config["rename"]:
            # Set up new outfile names and also change rateints to calints.
            outfiles = ['{}_f{}_reduced'.format(s3_config["rename"],i) for i, f in enumerate(files)]

        # Process Stage 3.
        do_stage3(files,outfiles,output_dir,s3_config,diagnosticplots_dir)

        # Write the config dictionary out as a copy.
        config_outdir = os.path.join(output_dir,"configuration")
        if not os.path.exists(config_outdir):
            os.makedirs(config_outdir)
        config_name = "s3_juniper.berry"
        if output_run:
            config_name = "s3_{}_juniper.berry".format(output_run)
        shutil.copy(s3_config_path,os.path.join(config_outdir,config_name))

    ### Run Juniper Stage 4: Extraction
    if 4 in stages:
        # Open the config dictionary.
        s4_config_path = glob.glob(os.path.join(config_folder,'s4_*'))[0]
        s4_config = read_config.read_config(s4_config_path)

        # Set up run name and define directories.
        project_dir = s4_config["toplevel_dir"]
        input_dir = os.path.join(project_dir,s4_config["input"])
        input_run = s4_config["input_run"]
        if input_run:
            # Add an extra sub-folder to select the correct s3 run.
            input_dir = os.path.join(project_dir,os.path.join(s4_config["input"],input_run))
        output_dir = os.path.join(project_dir,s4_config["output"])
        output_run = s4_config["output_run"]
        if output_run:
            # Add an extra sub-folder to separate this run from other runs.
            output_dir = os.path.join(project_dir,os.path.join(s4_config["output"],output_run))
        diagnosticplots_dir = os.path.join(output_dir, s4_config["diagnostics"])

        # Open all needed directories.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(diagnosticplots_dir):
            os.makedirs(diagnosticplots_dir)

        # Find files.
        files = sorted(glob.glob(os.path.join(input_dir,"*reduced.npy")))
        fnames = [str.split(f,sep='/')[-1] for f in files]
        outfile = [str.replace(f,'_reduced.npy','_1Dspec') for f in fnames][0] # use default names and change reduced to 1Dspec, that's all.
        if s4_config["rename"]:
            # Set up new outfile names and also change reduced to 1Dspec.
            outfile = '{}_1Dspec'.format(s4_config["rename"])

        # Process Stage 4.
        do_stage4(files,outfile,output_dir,s4_config,diagnosticplots_dir)

        # Write the config dictionary out as a copy.
        config_outdir = os.path.join(output_dir,"configuration")
        if not os.path.exists(config_outdir):
            os.makedirs(config_outdir)
        config_name = "s4_juniper.berry"
        if output_run:
            config_name = "s4_{}_juniper.berry".format(output_run)
        shutil.copy(s4_config_path,os.path.join(config_outdir,config_name))

    ### Run Juniper Stage 5: Binning and Fitting
    if 5 in stages:
        # Open the config dictionary.
        s5_config_path = glob.glob(os.path.join(config_folder,'s5_*'))[0]
        s5_config = read_config.read_config(s5_config_path)

        # Set up run name and define directories.
        # In this stage, multiple Stage 4 inputs can be supplied!
        project_dir = s5_config["toplevel_dir"]
        input_dirs = [os.path.join(project_dir,s5_config["input"]),]
        input_runs = s5_config["input_runs"]
        if input_runs:
            # Add extra sub-folders to select the correct s4 runs.
            input_dirs = []
            for input_run in input_runs:
                input_dir = os.path.join(project_dir,os.path.join(s5_config["input"],input_run))
                input_dirs.append(input_dir)
        output_dir = os.path.join(project_dir,s5_config["output"])
        output_run = s5_config["output_run"]
        if output_run:
            # Add an extra sub-folder to separate this run from other runs.
            output_dir = os.path.join(project_dir,os.path.join(s5_config["output"],output_run))
        diagnosticplots_dir = os.path.join(output_dir, s5_config["diagnostics"])

        # Open all needed directories.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(diagnosticplots_dir):
            os.makedirs(diagnosticplots_dir)

        # Find files.
        files = []
        for input_dir in input_dirs:
            subfiles = sorted(glob.glob(os.path.join(input_dir,"*1Dspec.npy")))
            for subfile in subfiles:
                files.append(subfile) # collect all 1D spec together
            fnames = [str.split(f,sep='/')[-1] for f in subfiles]
            outfile = [str.replace(f,'_1Dspec.npy','_fits') for f in fnames][0] # use default name. There's only one file to be output here.
            if s5_config["rename"]:
                # Set up new outfile name.
                outfile = '{}_fits'.format(s5_config["rename"])

        # Process Stage 5.
        do_stage5(files,outfile,output_dir,s5_config,diagnosticplots_dir)

        # Write the config dictionary out as a copy.
        config_outdir = os.path.join(output_dir,"configuration")
        if not os.path.exists(config_outdir):
            os.makedirs(config_outdir)
        config_name = "s5_juniper.berry"
        if output_run:
            config_name = "s5_{}_juniper.berry".format(output_run)
        shutil.copy(s5_config_path,os.path.join(config_outdir,config_name))
    ### Run Juniper Stage 6: Results
    if 6 in stages:
        # Open the config dictionary.
        s6_config_path = glob.glob(os.path.join(config_folder,'s6_*'))[0]
        s6_config = read_config.read_config(s6_config_path)

        # Set up run name and define directories.
        project_dir = s6_config["toplevel_dir"]
        input_dir = os.path.join(project_dir,s6_config["input"])
        input_run = s6_config["input_run"]
        if input_run:
            # Add an extra sub-folder to select the correct s5 run.
            input_dir = os.path.join(project_dir,os.path.join(s6_config["input"],input_run))
        output_dir = os.path.join(project_dir,s6_config["output"])
        output_run = s6_config["output_run"]
        if output_run:
            # Add an extra sub-folder to separate this run from other runs.
            output_dir = os.path.join(project_dir,os.path.join(s6_config["output"],output_run))
        diagnosticplots_dir = os.path.join(output_dir, s6_config["diagnostics"])

        # Open all needed directories.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(diagnosticplots_dir):
            os.makedirs(diagnosticplots_dir)

        # Find files.
        files = sorted(glob.glob(os.path.join(input_dir,"*.npy")))
        fnames = [str.split(f,sep='/')[-1] for f in files]
        outfile = [str.split(f,sep='_')[0]+'_results' for f in fnames][0] # use default name. There's only one file to be output here.
        if s6_config["rename"]:
            # Set up new outfile name.
            outfile = '{}_results'.format(s6_config["rename"])

        # Process Stage 6.
        do_stage6(files,outfile,output_dir,s6_config,diagnosticplots_dir)

        # Write the config dictionary out as a copy.
        config_outdir = os.path.join(output_dir,"configuration")
        if not os.path.exists(config_outdir):
            os.makedirs(config_outdir)
        config_name = "s6_juniper.berry"
        if output_run:
            config_name = "s6_{}_juniper.berry".format(output_run)
        shutil.copy(s6_config_path,os.path.join(config_outdir,config_name))