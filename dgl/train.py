import os
# import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.pt_callbacks import ModelCheckpoint
from test_tube import Experiment

from dgl.build_model import GraphLayoutVAE
from dgl.config import args

if __name__ == '__main__':

    model = GraphLayoutVAE(args)

    exp = Experiment(name=args.dataset, save_dir=args.logs, version=args.version)
    model_save_path = '{}/{}/version_{}'.format(args.logs, exp.name, exp.version)
    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        # save_best_only=True,
        verbose=True,
        monitor='loss',
        mode='min',
        prefix=args.dataset
    )

    if args.save_meta:
        exp.argparse(args)
        exp.save()

    weights = [x for x in os.listdir(model_save_path) if '.ckpt' in x]
    tags_path = exp.get_data_path(exp.name, exp.version)
    tags_path = os.path.join(tags_path, 'meta_tags.csv')
    print(weights)
    if len(weights)>0:
        print('loading: ', weights[0])
        model = model.load_from_metrics(weights_path=os.path.join(model_save_path,weights[0]), tags_csv=tags_path, on_gpu=True)

    trainer = Trainer(experiment=exp, checkpoint_callback=checkpoint, max_nb_epochs=400, gpus=[2,])
    trainer.fit(model)
    trainer.model.create_sample_grid(trainer.data_parallel_device_ids[0])

    # view tensorflow logs
    print(f'View tensorboard logs by running\ntensorboard --logdir {os.getcwd()}')
    print('and going to http://localhost:6006 on your browser')
