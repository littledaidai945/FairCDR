from config.configurator import configs
from trainer.trainer import init_seed
from models.bulid_model import build_model
from trainer.logger import Logger
from data_utils.build_data_handler import build_data_handler
from trainer.build_trainer import build_trainer
from trainer.tuner import Tuner
import os, sys
os.chdir(sys.path[0]) 

def main():
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()
    model = build_model(data_handler).to(configs['device'])
    logger = Logger()
    trainer = build_trainer(data_handler, logger)
    best_model = trainer.train(model)


def tune():

    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()
    logger = Logger()
    tuner = Tuner(logger)
    trainer = build_trainer(data_handler, logger)
    tuner.grid_search(data_handler, trainer)

def test():
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()
    model = build_model(data_handler).to(configs['device'])
    logger = Logger()
    trainer = build_trainer(data_handler, logger)
    best_model = trainer.load(model)

if not configs['tune']['enable']:
    main()
else:
    tune()


