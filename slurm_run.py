import argparse
import subprocess
import sys

def slurm_submit(script):
    """
    Submit the SLURM script using sbatch and return the job ID.
    """
    try:
        # Use a list for the command and pass the script via stdin
        output = subprocess.check_output(["sbatch"], input=script, universal_newlines=True)
        job_id = output.strip().split()[-1]
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.output}", file=sys.stderr)
        sys.exit(1)

def submit(
        conda_env_name, 
        script_name,
        num_gpus, 
        partition, 
        job_name, 
        mem, 
        cpus, 
        time, 
        note, 
        mode,
        dataset, 
        paths, 
        load_jobid, 
        gpu_type,
        model,
        version,
        override
        ):
    """
    Construct and submit the SLURM script with the specified parameters.
    """
        # Define GPU configurations
    gpu_configs = {
        'all': 'g[3040-3047,3050-3057,3060-3067,3070-3077,3080-3087,3090-3097,3091-3113,3115-3132]',
        'a100': 'g[3040-3047,3050-3057,3060-3067,3070-3077,3080-3087]',
        'l40s': 'g[3091-3113,3115-3132]',
        'h200': 'g[3125-3132]',
        # Add more GPU types here if needed
    }

    gpu_resource = f"{gpu_configs[gpu_type]}"
    """Submit job to cluster."""
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}    
#SBATCH --partition={partition}
#SBATCH --account=portia
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gpus={num_gpus}
#SBATCH --mem={mem}G
#SBATCH --verbose  
#SBATCH --open-mode=append
#SBATCH -o ./OutFiles/slurm-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eabe@uw.edu
#SBATCH --nodelist={gpu_resource}
module load cuda/12.6.3
set -x
source ~/.bashrc
nvidia-smi
conda activate {conda_env_name}
unset LD_LIBRARY_PATH
echo $SLURMD_NODENAME
python -u {script_name}.py hydra.mode={mode} paths={paths} note={note} dataset={dataset} model={model} version={version} load_jobid={load_jobid} run_id=$SLURM_JOB_ID {override}
            """
    print(f"Submitting job")
    print(script)
    job_id = slurm_submit(script)
    print(job_id)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Submit a SLURM job with specified GPU type.')
    parser.add_argument('--conda_env_name', type=str, default='tidhy',
                        help='Name of the conda environment (default: tidhy)')
    parser.add_argument('--script_name', type=str, default='Run_TiDHy_NNX_vmap',
                        help='Name of the script to run (default: Run_TiDHy_NNX_vmap)')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to request (default: 1)')
    parser.add_argument('--gpu_type', type=str, default='l40s',
                        help='Type of GPU to request (default: l40s)')
    parser.add_argument('--job_name', type=str, default='Fruitfly',
                        help='Name of the SLURM job (default: rodent)')
    parser.add_argument('--mem', type=int, default=512,
                        help='Memory in GB (default: 128)')
    parser.add_argument('--cpus', type=int, default=64,
                        help='Number of CPU cores (default: 16)')
    parser.add_argument('--time', type=str, default='2-00:00:00',
                        help='Time limit for the job day-hr-min-sec (default: 2-00:00:00)')
    parser.add_argument('--partition', type=str, default='gpu-l40s',
                        help='Partition to run job (default: gpu-l40s)')
    parser.add_argument('--note', type=str, default='hyak_ckpt',
                        help='Note for job (default: hyak_ckpt)')
    parser.add_argument('--mode', type=str, default='RUN',
                        help='Name of hydra mode  (default: RUN)')
    parser.add_argument('--dataset', type=str, default='SLDS',
                        help='Name of dataset yaml  (default: SLDS)')
    parser.add_argument('--model', type=str, default='r2_sparse',
                        help='Name of model yaml  (default: r2_sparse)')
    parser.add_argument('--version', type=str, default='TiDHy',
                        help='Version of model yaml  (default: TiDHy)')
    parser.add_argument('--paths', type=str, default='hyak',
                        help='Name of paths yaml  (default: hyak)')
    parser.add_argument('--load_jobid', type=str, default='',
                        help='JobID to resume training (default: '')')
    parser.add_argument('--override', type=str, default='',
                        help='JobID to resume training (default: '')')

    args = parser.parse_args()

    submit(
        conda_env_name=args.conda_env_name,
        script_name=args.script_name,
        num_gpus=args.num_gpus,
        job_name=args.job_name,
        mem=args.mem,
        cpus=args.cpus,
        time=args.time,
        partition=args.partition,
        note=args.note,
        mode=args.mode,
        dataset=args.dataset,
        paths=args.paths,
        load_jobid=args.load_jobid,
        gpu_type=args.gpu_type,
        model=args.model,
        version=args.version,
        override=args.override,
    )

if __name__ == "__main__":
    main()
    
##### Saving commands #####
#### cancel all jobs: squeue -u $USER -h | awk '{print $1}' | xargs scancel
# python scripts/slurm_run.py --paths=hyak --dataset=CalMS21 version=TiDHy --override='seed=43' --note='hyak_calms21_seed43'


## exclude nodes g3090,g3107,g3097,g3109,g3113,g3091,g3096


#### full gpu node list: g[3040-3047,3050-3057,3060-3067,3070-3077,3080-3087,3090-3097,3091-3132]
#### a40 & a100 nodes only: g[3040-3047,3050-3057,3060-3067,3070-3077,3080-3087]
#### l40 & l40s nodes only: g[3091-3124]
#### h200 nodes only: g[3125-3132]

#### wandb regex: ^(?!.*table)(?!.*std).*$|^reward*&