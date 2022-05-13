import gin
import argparse
import subprocess
import os


parser = argparse.ArgumentParser(description='Training script.')
parser.add_argument('--cloud-config', action = 'store', type = str, help = 'Cloud configuration', required=True)
parser.add_argument('--experiment', action = 'store', type = str, help = 'Experiment configuration', required=True)

@gin.configurable
def gcloud(args, check=True, gcloud_path='gcloud', fake=False):
    print('  Running: '+ ' '.join([gcloud_path] + args))
    if not fake:
        res = subprocess.run([gcloud_path] + args, check=check)
        res = res.returncode==0
    else:
        res = False

    print()
    return res

def gcloud_start(instance_name):
    return gcloud(['compute', 'instances', 'start', instance_name], check=False)

@gin.configurable
def gcloud_create_instance(instance_name, creation_args=gin.REQUIRED):
    args = [f'--{k}={creation_args[k]}' for k in creation_args]
    return gcloud(['compute', 'instances', 'create', instance_name]+args)

REMOTE_RUNNER = 'scripts/run_remote_training.sh'

@gin.configurable
def gcloud_remote_training(args, instance_name,
        git_repo_url=gin.REQUIRED, git_branch=gin.REQUIRED, gcs_model_path=gin.REQUIRED,
        wandb_api_key=None, wandb_entity=None):
    command = [
        'bash', os.path.basename(REMOTE_RUNNER), git_repo_url, git_branch, args.experiment, gcs_model_path
    ]
    if wandb_api_key:
        command.extend([wandb_api_key, wandb_entity])
    command = ' '.join(command)
    # Comment for easier debugging
    command = f'nohup {command} 2>nohup.err.$$ >nohup.out.$$ ; tail -n 100 -f nohup.out.$$ &'
    return gcloud(['compute', 'ssh', '--command=' + command, instance_name])

if __name__=='__main__':
    args = parser.parse_args()
    gin.parse_config_file(args.cloud_config)

    instance_name = gin.query_parameter('%instance_name')

    print('\nStarting Cloud Training\n')

    print(f'Checking if instance {instance_name} already exists')
    if not gcloud_start(instance_name):
        print(f'Instance {instance_name} not yet available, creating')
        gcloud_create_instance(instance_name)
    else:
        print(f'Using instance: {instance_name}')

    print('Copying remote running script')
    gcloud(['compute', 'scp', REMOTE_RUNNER, f'{instance_name}:'])

    gcloud_remote_training(args, instance_name)