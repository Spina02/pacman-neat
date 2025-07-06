import argparse
from neat_model import trainer, run
import os

def main():
    parser = argparse.ArgumentParser(description='Train NEAT to play Pac-Man.')
    parser.add_argument('mode',                 type=str, choices=['train', 'run'], help='Mode to run: train or run')
    parser.add_argument('--render',             type=bool, default=False,           help='Render the game during training')
    parser.add_argument('--config',             type=str, default='config',         help='NEAT configuration file')
    parser.add_argument('--generations',        type=int, default=2000,              help='Number of generations to run')
    parser.add_argument('--checkpoint_dir',     type=str, default='checkpoints',    help='Directory to save checkpoints')
    parser.add_argument('--restore_checkpoint', type=str, default=None,             help='Checkpoint file to restore from')
    parser.add_argument('--observation_mode',   type=str, default='minimap',        help='Observation mode for the agent',  
                                                          choices=['simple', 'minimap'])
    parser.add_argument('--cores',              type=int, default=15,                help='Number of cores to use for training')
    parser.add_argument('--reset',              type=bool, default=False,           help='Reset the checkpoint directory')
    parser.add_argument("--best", action='store_true',
                        help="Load the best genome found overall (looks for 'best_MODE_latest.pkl'). Overrides load_file if found.")
    parser.add_argument('--max_steps', type=int, default=None,
                        help="Override the maximum steps for the run (default: use environment's MAX_EPISODE_STEPS). Only for run mode.")
    parser.add_argument('--debug', type=int, default=0,
                        help="Debug level for verbose output (0-3). Only for run mode.")
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Handle reset flag
        if args.reset:
            print(f"Resetting checkpoints for mode '{args.observation_mode}' in {args.checkpoint_dir}...")
            prefix_to_delete = f'checkpoint-{args.observation_mode}-'
            files_deleted = 0
            if not os.path.exists(args.checkpoint_dir):
                os.makedirs(args.checkpoint_dir)
            else:
                for f in os.listdir(args.checkpoint_dir):
                    if f.startswith(prefix_to_delete):
                        try:
                            os.remove(os.path.join(args.checkpoint_dir, f))
                            files_deleted += 1
                        except OSError as e:
                            print(f"Error deleting file {f}: {e}")
                            
                if "best_genomes" in os.listdir(args.checkpoint_dir):
                    for f in os.listdir(args.checkpoint_dir + "/best_genomes"):
                        try:   
                            os.remove(os.path.join(args.checkpoint_dir + "/best_genomes", f))
                            files_deleted += 1
                        except OSError as e:
                            print(f"Error deleting file {f}: {e}")
                print(f"Deleted {files_deleted} checkpoint files.")
                args.restore_checkpoint = None # Ensure we don't try to restore after reset

        # Find the latest checkpoint if not specified
        if args.restore_checkpoint is None:
            checkpoint_prefix = f'checkpoint-{args.observation_mode}-'
            checkpoints = sorted(
                [f for f in os.listdir(args.checkpoint_dir) if f.startswith(checkpoint_prefix)],
                key=lambda x: int(x.split('-')[-1]) # Sort by generation number
            )
            print(f"Checkpoints: {checkpoints}")
            if checkpoints:
                args.restore_checkpoint = os.path.join(args.checkpoint_dir, checkpoints[-1])
                print(f"Found latest checkpoint for mode '{args.observation_mode}': {args.restore_checkpoint}")
            else:
                print(f"No checkpoints found for mode '{args.observation_mode}'. Starting new training.")

        trainer.run(
            restore_checkpoint= args.restore_checkpoint,
            observation_mode=   args.observation_mode,
            config_file=        args.config,
            generations=        args.generations,
            render=             args.render,
            cores=              args.cores
        )
        
    elif args.mode == 'run':
        run.run(
            config_file=        args.config,
            checkpoint_file=    args.restore_checkpoint,
            observation_mode=   args.observation_mode,
            best=               args.best,
            max_steps=          args.max_steps,
            debug=              args.debug
        )

if __name__ == '__main__':
    main()