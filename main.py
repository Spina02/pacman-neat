import argparse
from neat_model import trainer, run

def main():
    parser = argparse.ArgumentParser(description='Train NEAT to play Pac-Man.')
    parser.add_argument('mode',                 type=str, choices=['train', 'run'], help='Mode to run: train or run')
    parser.add_argument('--render',             type=bool, default=False,           help='Render the game during training')
    parser.add_argument('--config',             type=str, default='config',         help='NEAT configuration file')
    parser.add_argument('--generations',        type=int, default=500,              help='Number of generations to run')
    parser.add_argument('--checkpoint_dir',     type=str, default='checkpoints',    help='Directory to save checkpoints')
    parser.add_argument('--restore_checkpoint', type=str, default=None,             help='Checkpoint file to restore from')
    parser.add_argument('--observation_mode',   type=str, default='minimap',        help='Observation mode for the agent',  
                                                          choices=['simple', 'minimap'])
    parser.add_argument('--cores',              type=int, default=1,                help='Number of cores to use for training')

    args = parser.parse_args()

    if args.mode == 'train':
        trainer.run(
            restore_checkpoint= args.restore_checkpoint,
            observation_mode=   args.observation_mode,
            config_file=        args.config,
            generations=        args.generations,
            render=             args.render,
            cores=              args.cores,
        )
        
    elif args.mode == 'run':
        run.run(
            config_file=        args.config,
            checkpoint_file=    args.restore_checkpoint,
            observation_mode=   args.observation_mode
        )

if __name__ == '__main__':
    main()