import os

from pytorch_lightning import Trainer
from test_tube import Experiment

from dgl.build_model import GraphLayoutVAE
from dgl.config import args

if __name__ == '__main__':
    dims = {'can_96': 96, }

    model = GraphLayoutVAE(dims[args.dataset], 2, args.dataset, args.dataset_folder)
    exp = Experiment(save_dir=os.getcwd())

    # train on cpu using only 10% of the data (for demo purposes)
    trainer = Trainer(experiment=exp, max_nb_epochs=2, train_percent_check=100, gpus=[2])

    # train on 4 gpus
    # trainer = Trainer(experiment=exp, max_nb_epochs=1, gpus=[0, 1, 2, 3])

    # train on 32 gpus across 4 nodes (make sure to submit appropriate SLURM job)
    # trainer = Trainer(experiment=exp, max_nb_epochs=1, gpus=[0, 1, 2, 3, 4, 5, 6, 7], nb_gpu_nodes=4)

    # train (1 epoch only here for demo)
    trainer.fit(model)

    # view tensorflow logs
    print(f'View tensorboard logs by running\ntensorboard --logdir {os.getcwd()}')
    print('and going to http://localhost:6006 on your browser')
