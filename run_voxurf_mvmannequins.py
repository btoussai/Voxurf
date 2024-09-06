import subprocess
import sys

sequences = [
    "kinette-cos-hx",
    "kinette-hx",
    "kinette-jea-hx",
    "kinette-opt1-hx",
    "kinette-opt2-hx",
    "kinette-opt3-hx",
    "kinette-sho-hx",
    "kinette-tig-hx",
    "kino-cos-hx",
    "kino-hx",
    "kino-jea-hx",
    "kino-opt-hx",
    "kino-sho-hx",
    "kino-tig-hx"
]

configs_coarse = [
    "configs/mvmannequins_e2e/coarse.py",
    "configs/mvmannequins_e2e_womask/coarse.py"
]
configs_fine = [
    "configs/mvmannequins_e2e/fine.py",
    "configs/mvmannequins_e2e_womask/fine.py"
]

womask = False

for scene in sequences:
    proc = subprocess.run(
            [
                sys.executable,
                "run.py",
                "--config", 
                configs_coarse[womask], 
                "-p", 
                "workdir",
                "--no_reload",
                "--run_dvgo_init",
                "--sdf_mode",
                "voxurf_coarse",
                "--scene",
                str(scene)
            ],
            cwd = ".") 
    proc = subprocess.run(
            [
                sys.executable,
                "run.py",
                "--config", 
                configs_fine[womask], 
                "-p", 
                "workdir",
                "--no_reload",
                "--sdf_mode",
                "voxurf_fine",
                "--scene",
                str(scene),
                # "--render_only",
                "--render_train"
            ],
            cwd = ".") 