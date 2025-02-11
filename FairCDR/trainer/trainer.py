import gc
import os
import time
import copy
import torch
import random
import datetime
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from numpy import random
from copy import deepcopy
import torch.optim as optim
from trainer.metrics import Metric
from config.configurator import configs
from models.bulid_model import build_model
from torch.utils.tensorboard import SummaryWriter
from .utils import DisabledSummaryWriter, log_exceptions
from .discriminator import GenderDiscriminator
from .DNN import DNN
from .diffusion import ModelMeanType 
from .diffusion import GaussianDiffusion 
if 'tensorboard' in configs['train'] and configs['train']['tensorboard']:
    writer = SummaryWriter(log_dir='runs')
else:
    writer = DisabledSummaryWriter()


def init_seed():
    if 'reproducible' in configs['train']:
        if configs['train']['reproducible']:
            seed = configs['train']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class PreTrainer(object):
    def __init__(self, data_handler, logger):
        self.data_handler = data_handler
        self.logger = logger
        self.metric = Metric()
        if configs["diffusion"]["mean_type"] == 'x0':
            configs["diffusion"]["mean_type"] = ModelMeanType.START_X
        elif configs["diffusion"]["mean_type"] == 'eps':
            configs["diffusion"]["mean_type"] = ModelMeanType.EPSILON
        else:
            raise ValueError("Unimplemented mean type %s" % configs["mean_type"])
        self.diffusion = GaussianDiffusion(configs,configs["diffusion"]["mean_type"], configs["diffusion"]["noise_schedule"],configs["diffusion"]["noise_scale"], configs["diffusion"]["noise_min"], configs["diffusion"]["noise_max"], configs["diffusion"]["steps"], configs['device']).to(configs['device'])
        self.emb_size=configs["diffusion"]["emb_size"]
        self.time_type=configs["diffusion"]["time_type"]
        self.norm=configs["diffusion"]["norm"]
        self.reweight=configs["diffusion"]["reweight"]
        self.hidden_size=configs["model"]["embedding_size"]
        out_dims = [1000,self.hidden_size]
        in_dims = [self.hidden_size,1000]
        optim_config = configs['optimizer']
        self.DNN= DNN(in_dims, out_dims, self.emb_size, self.time_type, self.norm).to(configs["device"])
        self.optimizer_DNN = optim.Adam(self.DNN.parameters(
            ), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])
        self.discriminator=GenderDiscriminator(configs['model']['embedding_size'],configs['model']['embedding_size'],3).to(configs['device'])
       
        if optim_config['name'] == 'adam':
            self.dis_optimizer = optim.Adam(self.discriminator.parameters(
            ), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])
        self.config=configs
    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(
            ), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])

    def get_dataloader_iter(self, dataloader):
        while True:
            for data in dataloader:
                yield data


    def train_dis_epoch(self, model, epoch_idx):     
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()
        loss_log_dict = {}
        loss_log_dict["adv_loss"]=0.0
        loss_log_dict["auc"]=0.0
        ep_loss = 0
        steps = len(train_dataloader.dataset) // configs['train']['batch_size']
        model.eval()
        self.discriminator.train()
        for _, tem in tqdm(enumerate(train_dataloader), desc='Training Discriminator', total=len(train_dataloader)):
            self.dis_optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
            loss, auc = model.dis_loss(self.discriminator,batch_data)
            ep_loss += loss.item()
            loss.backward()
            self.dis_optimizer.step()
            loss_log_dict["adv_loss"]+=loss.item()
            loss_log_dict["auc"]+=auc
   
            
        loss_log_dict["adv_loss"]/=len(train_dataloader)
        loss_log_dict["auc"]/=len(train_dataloader)
        writer.add_scalar('Loss/train', ep_loss / steps, epoch_idx)
        save_dir = "./save_model/{}_{}/".format(self.config["data"]["name"], self.config["target_data"]["name"])
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.discriminator.state_dict(), os.path.join(save_dir, "discriminator_pretrain.pth"))  
        # log loss
        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)

    @log_exceptions
    def train(self, model):
        self.create_optimizer(model)
        train_config = configs['train']
        train_config['early_stop']=False
        val_data = self.data_handler.test_dataloader.dataset
        val_data.negative_samples = []
        m = len(val_data.gender)
        for i in range(m):
            negative_sample = []
            negative_sample.append(val_data.user_pos_lists[i][0])
            while len(negative_sample)<101 :
                sample = random.randint(0, m - 1)
                if sample not in negative_sample:
                    negative_sample.append(sample)
            val_data.negative_samples.append(negative_sample)
        self.data_handler.test_dataloader.dataset.negative_samples = val_data.negative_samples

        val_data_target = self.data_handler.test_dataloader_target.dataset
        val_data_target.negative_samples = []
        m = len(val_data_target.gender)
        for i in range(m):
            negative_sample = []
            negative_sample.append(val_data_target.user_pos_lists[i][0]-configs["data"]["item_num"])
            while len(negative_sample)<101 :
                sample = random.randint(0, configs["target_data"]["item_num"])
                if sample not in negative_sample:
                    negative_sample.append(sample)
            val_data_target.negative_samples.append(negative_sample)
        self.data_handler.test_dataloader_target.dataset.negative_samples = val_data_target.negative_samples

        if not train_config['early_stop']:
            for epoch_idx in range(train_config['epoch']):
                # train
                self.train_epoch(model, epoch_idx)
                # evaluate
                self.evaluate(model, epoch_idx)
                self.evaluate_target(model, epoch_idx)
            self.test(model)
            self.save_model(model)

            for epoch_idx in range(train_config['epoch']+train_config['epoch']):
                # train
                self.train_dis_epoch(model,epoch_idx)
                # evaluate
                if epoch_idx % train_config['test_step'] == 0:
                    self.evaluate(model, epoch_idx)
                    self.evaluate_target(model, epoch_idx)
            self.test(model)
            self.save_model(model)
            return model

        
        elif train_config['early_stop']:
            now_patience = 0
            best_epoch = 0
            best_metric = -1e9
            best_state_dict = None
            for epoch_idx in range(train_config['epoch']):
                # train
                self.train_epoch(model, epoch_idx)
                # evaluate
                if epoch_idx % train_config['test_step'] == 0:
                    eval_result = self.evaluate(model, epoch_idx)

                    if eval_result[configs['test']['metrics'][0]][0] > best_metric:
                        now_patience = 0
                        best_epoch = epoch_idx
                        best_metric = eval_result[configs['test']['metrics'][0]][0]
                        best_state_dict = deepcopy(model.state_dict())
                        self.logger.log("Validation score increased.  Copying the best model ...")
                    else:
                        now_patience += 1
                        self.logger.log(f"Early stop counter: {now_patience} out of {configs['train']['patience']}")

                    # early stop
                    if now_patience == configs['train']['patience']:
                        break
            
            # re-initialize the model and load the best parameter
            self.logger.log("Best Epoch {}".format(best_epoch))
            model = build_model(self.data_handler).to(configs['device'])
            model.load_state_dict(best_state_dict)
            self.evaluate(model)
            model = build_model(self.data_handler).to(configs['device'])
            model.load_state_dict(best_state_dict)
            self.test(model)
            self.save_model(model)
            return model

    @log_exceptions
    def evaluate(self, model, epoch_idx=None):
        model.eval()
        self.discriminator.eval()
        if hasattr(self.data_handler, 'valid_dataloader'):
            eval_result,auc = self.metric.eval(model, self.discriminator,self.data_handler.valid_dataloader)
            writer.add_scalar('HR/test', eval_result[configs['test']['metrics'][0]][0], epoch_idx)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Validation set', epoch_idx=epoch_idx,auc=auc)
            
        elif hasattr(self.data_handler, 'test_dataloader'):
            eval_result,auc = self.metric.eval(model,self.discriminator, self.data_handler.test_dataloader)
            writer.add_scalar('HR/test', eval_result[configs['test']['metrics'][0]][0], epoch_idx)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set', epoch_idx=epoch_idx,auc=auc)
        else:
            raise NotImplemented
        return eval_result,auc

    def evaluate_target(self, model, epoch_idx=None):
        model.eval()
        self.discriminator.eval()
        if hasattr(self.data_handler, 'valid_dataloader'):
            eval_result,auc = self.metric.eval_target_DANN(model, self.discriminator,self.data_handler.valid_dataloader_target)
            writer.add_scalar('HR/test', eval_result[configs['test']['metrics'][0]][0], epoch_idx)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Validation set', epoch_idx=epoch_idx,auc=auc)
            
        elif hasattr(self.data_handler, 'test_dataloader'):
            eval_result,auc = self.metric.eval_target_DANN(model,self.discriminator, self.data_handler.test_dataloader_target)
            writer.add_scalar('HR/test', eval_result[configs['test']['metrics'][0]][0], epoch_idx)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set', epoch_idx=epoch_idx,auc=auc)
        else:
            raise NotImplemented
        return eval_result,auc

    @log_exceptions
    def test(self, model):
        model.eval()
        self.discriminator.eval()
        if hasattr(self.data_handler, 'test_dataloader'):
            eval_result ,auc= self.metric.eval(model,self.discriminator, self.data_handler.test_dataloader)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set',auc=auc)
        else:
            raise NotImplemented
        return eval_result

    def save_model(self, model):
        if configs['train']['save_model']:
            model_state_dict = model.state_dict()
            model_name = configs['model']['name']
            data_name = configs['data']['name']
            if not configs['tune']['enable']:
                save_dir_path = './checkpoint/{}'.format(model_name)
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                timestamp = int(time.time())
                torch.save(
                    model_state_dict, '{}/{}-{}-{}.pth'.format(save_dir_path, model_name, data_name, timestamp))
                self.logger.log("Save model parameters to {}".format(
                    '{}/{}-{}.pth'.format(save_dir_path, model_name, timestamp)))
            else:
                save_dir_path = './checkpoint/{}/tune'.format(model_name)
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                now_para_str = configs['tune']['now_para_str']
                torch.save(
                    model_state_dict, '{}/{}-{}-{}.pth'.format(save_dir_path, model_name, data_name, now_para_str))
                self.logger.log("Save model parameters to {}".format(
                    '{}/{}-{}.pth'.format(save_dir_path, model_name, now_para_str)))

    def load_model(self, model):
        if 'pretrain_path' in configs['train']:
            pretrain_path = configs['train']['pretrain_path']
            model.load_state_dict(torch.load(pretrain_path))
            self.logger.log(
                "Load model parameters from {}".format(pretrain_path))
            return model
        else:
            raise KeyError("No pretrain_path in configs['train']")


class MAERecPreTrainer(PreTrainer):
    def __init__(self, data_handler, logger):
        super(MAERecPreTrainer, self).__init__(data_handler, logger)
        self.logger = logger

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(
                [{"params": model.encoder.parameters()},
                {"params": model.decoder.parameters()},
                {"params": model.emb_layer.parameters()},
                {"params": model.transformer_layers.parameters()}],
                lr=optim_config['lr'], weight_decay=optim_config['weight_decay']
            )

    def calc_reward(self, lastLosses, eps):
        if len(lastLosses) < 3:
            return 1.0
        curDecrease = lastLosses[-2] - lastLosses[-1]
        avgDecrease = 0
        for i in range(len(lastLosses) - 2):
            avgDecrease += lastLosses[i] - lastLosses[i + 1]
        avgDecrease /= len(lastLosses) - 2
        return 1 if curDecrease > avgDecrease else eps

    def sample_pos_edges(self, masked_edges):
        return masked_edges[torch.randperm(masked_edges.shape[0])[:configs['model']['con_batch']]]

    def sample_neg_edges(self, pos, dok):
        neg = []
        for u, v in pos:
            cu_neg = []
            num_samp = configs['model']['num_reco_neg'] // 2
            for i in range(num_samp):
                while True:
                    v_neg = np.random.randint(1, configs['data']['item_num'] + 1)
                    if (u, v_neg) not in dok:
                        break
                cu_neg.append([u, v_neg])
            for i in range(num_samp):
                while True:
                    u_neg = np.random.randint(1, configs['data']['item_num'] + 1)
                    if (u_neg, v) not in dok:
                        break
                cu_neg.append([u_neg, v])
            neg.append(cu_neg)
        return torch.Tensor(neg).long()

    def train_epoch(self, model, epoch_idx):

        model.train()

        loss_his = []
        loss_log_dict = {'loss': 0, 'loss_main': 0, 'loss_reco': 0, 'loss_regu': 0, 'loss_mask': 0}
        trn_loader = self.data_handler.train_dataloader
        trn_loader.dataset.sample_negs()

        for i, batch_data in tqdm(enumerate(trn_loader), desc='Training MAERec', total=len(trn_loader)):
            if i % configs['model']['mask_steps'] == 0:
                sample_scr, candidates = model.sampler(model.ii_adj_all_one, model.encoder.get_ego_embeds())
                masked_adj, masked_edg = model.masker(model.ii_adj, candidates)
            if isinstance(masked_adj, torch.Tensor) and masked_adj.is_sparse:
                masked_adj = masked_adj.coalesce()
                edge_index = masked_adj.indices()
                edge_index=edge_index.to(configs['device'])
            elif isinstance(masked_adj, torch.sparse.csr_matrix):
                r, c = masked_adj.nonzero()
                edge_index = torch.tensor([r, c], dtype=torch.long)
            batch_data = list(map(lambda x: x.long().to(configs['device']), batch_data))

            item_emb, item_emb_his = model.encoder(edge_index)
            pos = self.sample_pos_edges(masked_edg)
            neg = self.sample_neg_edges(pos, model.ii_dok)

            loss, loss_main, loss_reco, loss_regu = model.cal_loss(batch_data, item_emb, item_emb_his, pos, neg)
            loss_his.append(loss_main)

            if i % configs['model']['mask_steps'] == 0:
                reward = self.calc_reward(loss_his, configs['model']['eps'])
                loss_mask = -sample_scr.mean() * reward
                loss_log_dict['loss_mask'] += loss_mask / (len(trn_loader) // configs['model']['mask_steps'])
                loss_his = loss_his[-1:]
                loss += loss_mask

            loss_log_dict['loss'] += loss.item() / len(trn_loader)
            loss_log_dict['loss_main'] += loss_main.item() / len(trn_loader)
            loss_log_dict['loss_reco'] += loss_reco.item() / len(trn_loader)
            loss_log_dict['loss_regu'] += loss_regu.item() / len(trn_loader)

        writer.add_scalar('Loss/train', loss_log_dict['loss'], epoch_idx)

        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)


class Trainer(object):
    def __init__(self, data_handler, logger):
        self.data_handler = data_handler
        self.logger = logger
        self.metric = Metric()
        # self.eval_type = configs['eval_type']
        if configs["diffusion"]["mean_type"] == 'x0':
            configs["diffusion"]["mean_type"] = ModelMeanType.START_X
        elif configs["diffusion"]["mean_type"] == 'eps':
            configs["diffusion"]["mean_type"] = ModelMeanType.EPSILON
        else:
            raise ValueError("Unimplemented mean type %s" % configs["mean_type"])
        self.diffusion = GaussianDiffusion(configs,configs["diffusion"]["mean_type"], configs["diffusion"]["noise_schedule"],configs["diffusion"]["noise_scale"], configs["diffusion"]["noise_min"], configs["diffusion"]["noise_max"], configs["diffusion"]["steps"], configs['device']).to(configs['device'])
        self.emb_size=configs["diffusion"]["emb_size"]
        self.time_type=configs["diffusion"]["time_type"]
        self.norm=configs["diffusion"]["norm"]
        self.reweight=configs["diffusion"]["reweight"]
        self.hidden_size=configs["model"]["embedding_size"]
        out_dims = [1000,self.hidden_size]
        in_dims = [self.hidden_size,1000]
        optim_config = configs['optimizer']
        self.DNN= DNN(in_dims, out_dims, self.emb_size, self.time_type, self.norm).to(configs["device"])
        load_dir = "./save_model/{}_{}/DNN.pth".format(configs["data"]["name"], configs["target_data"]["name"])
        if os.path.exists(load_dir):
            self.DNN.load_state_dict(torch.load(load_dir))
        self.optimizer_DNN = optim.Adam(self.DNN.parameters(
            ), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])
        self.discriminator=GenderDiscriminator(configs['model']['embedding_size'],configs['model']['embedding_size'],3).to(configs['device'])
        load_dir = "./save_model/{}_{}/discriminator_pretrain.pth".format(configs["data"]["name"], configs["target_data"]["name"])
        if os.path.exists(load_dir):
            self.discriminator.load_state_dict(torch.load(load_dir))
        if optim_config['name'] == 'adam':
            self.dis_optimizer = optim.Adam(self.discriminator.parameters(
            ), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])
        self.config=configs
        self.device=configs["device"]
    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(
            ), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])

    
    def set_color(self,log, color, highlight=True):
        color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
        try:
            index = color_set.index(color)
        except:
            index = len(color_set) - 1
        prev_log = '\033['
        if highlight:
            prev_log += '1;3'
        else:
            prev_log += '0;3'
        prev_log += str(index) + 'm'
        return prev_log + log + '\033[0m'
  
    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = 4
        train_loss_output = (self.set_color('epoch %d training', 'green') + ' [' + self.set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = (self.set_color('train_loss%d', 'blue') + ': %.' + str(des) + 'f')
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += self.set_color('train loss', 'blue') + ': ' + des % losses
        return train_loss_output + ']'

    def train_dis_epoch(self, model, epoch_idx,item_emb):
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()
        loss_log_dict = {}
        loss_log_dict["adv_loss"]=0.0
        loss_log_dict["auc"]=0.0
        ep_loss = 0
        steps = len(train_dataloader.dataset) // configs['train']['batch_size']

        model.eval()
        self.discriminator.train()
        for _, tem in tqdm(enumerate(train_dataloader), desc='Training Discriminator', total=len(train_dataloader)):
            self.dis_optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
            loss, auc = model.dis_loss_mutual(self.discriminator,batch_data,item_emb)
            ep_loss += loss.item()
            loss.backward()
            self.dis_optimizer.step()
            loss_log_dict["adv_loss"]+=loss.item()
            loss_log_dict["auc"]+=auc
        loss_log_dict["adv_loss"]/=len(train_dataloader)
        loss_log_dict["auc"]/=len(train_dataloader)
        writer.add_scalar('Loss/train', ep_loss / steps, epoch_idx)

        # log loss
        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)
        return loss_log_dict["adv_loss"],loss_log_dict["auc"]

    @log_exceptions
    def train(self, model):
        self.create_optimizer(model)
        train_config = configs['train']
        train_config['early_stop']=False
        val_data = self.data_handler.test_dataloader.dataset
        val_data.negative_samples = []
        m = len(val_data.gender)
        for i in range(m):
            negative_sample = []
            negative_sample.append(val_data.user_pos_lists[i][0])
            while len(negative_sample)<101 :
                sample = random.randint(0, configs["data"]["item_num"] - 1)
                if sample not in negative_sample:
                    negative_sample.append(sample)
            val_data.negative_samples.append(negative_sample)
        self.data_handler.test_dataloader.dataset.negative_samples = val_data.negative_samples
        

        val_data_target = self.data_handler.test_dataloader_target.dataset
        val_data_target.negative_samples = []
        m = len(val_data_target.gender)
        for i in range(m):
            negative_sample = []
            negative_sample.append(val_data_target.user_pos_lists[i][0]-configs["data"]["item_num"])
            while len(negative_sample)<101 :
                sample = random.randint(0, configs["target_data"]["item_num"])
                if sample not in negative_sample:
                    negative_sample.append(sample)
            val_data_target.negative_samples.append(negative_sample)
        self.data_handler.test_dataloader_target.dataset.negative_samples = val_data_target.negative_samples

        if not train_config['early_stop']:
            save_dir = "./save/{}_{}".format(
            self.config["data"]["name"], self.config["target_data"]["name"]
        )
        
            source_items_embeddings=torch.load(os.path.join(save_dir, "source_all_item_embeddings.pt"))
            source_all_user_embeddings=torch.load(os.path.join(save_dir, "source_all_user_embeddings.pt")).to(self.device)
            source_items_embeddings=[tensor.to(self.device) for tensor in source_items_embeddings]

            target_items_embeddings=torch.load(os.path.join(save_dir, "target_all_item_embeddings.pt"))
            source_items_embeddings=[tensor.to(self.device) for tensor in source_items_embeddings]

            source_items_embeddings_para=torch.load(os.path.join(save_dir, "source_item_embeddings_para.pt")).to(self.device)
            for epoch_idx in range(configs["train"]["epoch"]):
                save_dir = "./save_model/{}_{}/".format(self.config["data"]["name"], self.config["target_data"]["name"])
                os.makedirs(save_dir, exist_ok=True)
                torch.save(self.discriminator.state_dict(), os.path.join(save_dir, "discriminator_pretrain.pth"))  
                F_all=source_items_embeddings
                for i in range(self.config["diffusion"]["steps"]):
                    self.logger.log("--------------------------t={}-----------------------".format(i+1))
                    F_all = [tensor.detach().requires_grad_(False) for tensor in F_all] 
                    source_items_embeddings_para =  source_items_embeddings_para.detach()
                    source_items_embeddings_para.requires_grad_(False) 
                    training_start_time = time.time()
                    discriminatorloss,auc = self.train_dis_epoch(model, epoch_idx,sum(F_all))
                    training_end_time = time.time()
                    train_loss_output = \
                    self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, discriminatorloss)
                    self.logger.log("Discrinimator Model Loss:"+train_loss_output)
                    self.logger.log("auc_score:"+str(auc))
                    model.train()
                    training_start_time = time.time()
                    n_users = source_all_user_embeddings.size(0)
                    train_loss = self.train_epoch(model, epoch_idx,source_items_embeddings_para,F_all)
                    training_end_time = time.time()
                    self.evaluate(model, epoch_idx,sum(F_all),sum(target_items_embeddings))
                    self.logger.log("Recommend Model Loss:"+train_loss_output)
                    with tqdm(total=1, desc=f"Train_diffusion_model {epoch_idx:>5}", ncols=100) as pbar:
                        training_start_time=time.time()
                        self.DNN.train()
                        self.diffusion.steps = i + 1  
                        self.optimizer_DNN.zero_grad()
                        all_losses = self.diffusion.mutual_training_losses(self.DNN,self.config['device'],reweight=self.reweight)
                        all_losses = all_losses["loss"].mean()
                        n_users = source_all_user_embeddings.size(0)

                        train_loss = torch.tensor(0.0, device=all_losses.device)

                        F_all = self.diffusion.mutual_p_sample(self.DNN, i + 1, sampling_noise=False)
                        source_items_embeddings_para = self.diffusion.mutual_p_sample_para(self.DNN, i + 1, sampling_noise=False)
                        source_items_embeddings_para = source_items_embeddings_para[n_users:]
                        model.eval()
                        train_loss = self.train_epoch_no_grad(model, epoch_idx,source_items_embeddings_para,F_all)

                        p_total_loss = all_losses + 0.1*train_loss
                        p_total_loss.backward()
                        self.optimizer_DNN.step()
                        training_end_time=time.time()   
                # evaluate
                if epoch_idx % train_config['test_step'] == 0:
                    self.evaluate(model, epoch_idx,sum(F_all),sum(target_items_embeddings))
            return model

        
        elif train_config['early_stop']:
            now_patience = 0
            best_epoch = 0
            best_metric = -1e9
            best_state_dict = None
            for epoch_idx in range(train_config['epoch']):
                # train
                self.train_epoch(model, epoch_idx)
                # evaluate
                if epoch_idx % train_config['test_step'] == 0:
                    eval_result = self.evaluate(model, epoch_idx)

                    if eval_result[configs['test']['metrics'][0]][0] > best_metric:
                        now_patience = 0
                        best_epoch = epoch_idx
                        best_metric = eval_result[configs['test']['metrics'][0]][0]
                        best_state_dict = deepcopy(model.state_dict())
                        self.logger.log("Validation score increased.  Copying the best model ...")
                    else:
                        now_patience += 1
                        self.logger.log(f"Early stop counter: {now_patience} out of {configs['train']['patience']}")

                    # early stop
                    if now_patience == configs['train']['patience']:
                        break
            
            # re-initialize the model and load the best parameter
            self.logger.log("Best Epoch {}".format(best_epoch))
            model = build_model(self.data_handler).to(configs['device'])
            model.load_state_dict(best_state_dict)
            self.evaluate(model)
            model = build_model(self.data_handler).to(configs['device'])
            model.load_state_dict(best_state_dict)
            self.test(model)
            # self.save_model(model)
            return model

    @log_exceptions
    def evaluate(self, model, epoch_idx=None,source_item_emb=None,target_item_emb=None):
        model.eval()
        self.discriminator.eval()
        if hasattr(self.data_handler, 'valid_dataloader'):
            eval_result,auc = self.metric.eval(model, self.discriminator,self.data_handler.valid_dataloader,source_item_emb)
            writer.add_scalar('HR/test', eval_result[configs['test']['metrics'][0]][0], epoch_idx)
            self.logger.log("source_result")
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Validation set', epoch_idx=epoch_idx,auc=auc)

            eval_result,auc = self.metric.eval_target(model, self.discriminator,self.data_handler.valid_dataloader_target,target_item_emb)
            writer.add_scalar('HR/test', eval_result[configs['test']['metrics'][0]][0], epoch_idx)
            self.logger.log("target_result")
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Validation set', epoch_idx=epoch_idx,auc=auc)
        elif hasattr(self.data_handler, 'test_dataloader'):
            eval_result,auc = self.metric.eval(model,self.discriminator, self.data_handler.test_dataloader,source_item_emb)
            writer.add_scalar('HR/test', eval_result[configs['test']['metrics'][0]][0], epoch_idx)
            self.logger.log("source_result")
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set', epoch_idx=epoch_idx,auc=auc)

            eval_result,auc = self.metric.eval_target(model,self.discriminator, self.data_handler.test_dataloader_target,target_item_emb)
            writer.add_scalar('HR/test', eval_result[configs['test']['metrics'][0]][0], epoch_idx)
            self.logger.log("target_result")
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set', epoch_idx=epoch_idx,auc=auc)
        else:
            raise NotImplemented
        return eval_result,auc

    @log_exceptions
    def test(self, model):
        model.eval()
        self.discriminator.eval()
        if hasattr(self.data_handler, 'test_dataloader'):
            eval_result ,auc= self.metric.eval(model,self.discriminator, self.data_handler.test_dataloader)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set',auc=auc)
        else:
            raise NotImplemented
        return eval_result

    def save_model(self, model):
        if configs['train']['save_model']:
            model_state_dict = model.state_dict()
            model_name = configs['model']['name']
            data_name = configs['data']['name']
            if not configs['tune']['enable']:
                save_dir_path = './checkpoint/{}'.format(model_name)
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                timestamp = int(time.time())
                torch.save(
                    model_state_dict, '{}/{}-{}-{}.pth'.format(save_dir_path, model_name, data_name, timestamp))
                self.logger.log("Save model parameters to {}".format(
                    '{}/{}-{}.pth'.format(save_dir_path, model_name, timestamp)))
            else:
                save_dir_path = './checkpoint/{}/tune'.format(model_name)
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                now_para_str = configs['tune']['now_para_str']
                torch.save(
                    model_state_dict, '{}/{}-{}-{}.pth'.format(save_dir_path, model_name, data_name, now_para_str))
                self.logger.log("Save model parameters to {}".format(
                    '{}/{}-{}.pth'.format(save_dir_path, model_name, now_para_str)))

    def load_model(self, model):
        if 'pretrain_path' in configs['train']:
            pretrain_path = configs['train']['pretrain_path']
            model.load_state_dict(torch.load(pretrain_path))
            self.logger.log(
                "Load model parameters from {}".format(pretrain_path))
            return model
        else:
            raise KeyError("No pretrain_path in configs['train']")


class MAERecTrainer(Trainer):
    def __init__(self, data_handler, logger):
        super(MAERecTrainer, self).__init__(data_handler, logger)
        self.logger = logger

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(
                [{"params": model.encoder.parameters()},
                {"params": model.decoder.parameters()},
                {"params": model.emb_layer.parameters()},
                {"params": model.transformer_layers.parameters()}],
                lr=optim_config['lr'], weight_decay=optim_config['weight_decay']
            )

    def calc_reward(self, lastLosses, eps):
        if len(lastLosses) < 3:
            return 1.0
        curDecrease = lastLosses[-2] - lastLosses[-1]
        avgDecrease = 0
        for i in range(len(lastLosses) - 2):
            avgDecrease += lastLosses[i] - lastLosses[i + 1]
        avgDecrease /= len(lastLosses) - 2
        return 1 if curDecrease > avgDecrease else eps

    def sample_pos_edges(self, masked_edges):
        return masked_edges[torch.randperm(masked_edges.shape[0])[:configs['model']['con_batch']]]

    def sample_neg_edges(self, pos, dok):
        neg = []
        for u, v in pos:
            cu_neg = []
            num_samp = configs['model']['num_reco_neg'] // 2
            for i in range(num_samp):
                while True:
                    v_neg = np.random.randint(1, configs['data']['item_num'] + 1)
                    if (u, v_neg) not in dok:
                        break
                cu_neg.append([u, v_neg])
            for i in range(num_samp):
                while True:
                    u_neg = np.random.randint(1, configs['data']['item_num'] + 1)
                    if (u_neg, v) not in dok:
                        break
                cu_neg.append([u_neg, v])
            neg.append(cu_neg)
        return torch.Tensor(neg).long()

  
    
    def train_epoch(self, model, epoch_idx,source_item_embeddings_para,source_all_item_embeddings):
        model.train()

        loss_his = []
        loss_log_dict = {'loss': 0, 'loss_main': 0, 'loss_reco': 0, 'loss_regu': 0, 'loss_mask': 0,'loss_adv':0}
        trn_loader = self.data_handler.train_dataloader
        trn_loader.dataset.sample_negs()

        for i, batch_data in tqdm(enumerate(trn_loader), desc='Training MAERec', total=len(trn_loader)):
            if i % configs['model']['mask_steps'] == 0:
                sample_scr, candidates = model.sampler(model.ii_adj_all_one, source_item_embeddings_para)
                masked_adj, masked_edg = model.masker(model.ii_adj, candidates)

            batch_data = list(map(lambda x: x.long().to(configs['device']), batch_data))
            pos = self.sample_pos_edges(masked_edg)
            neg = self.sample_neg_edges(pos, model.ii_dok)

            loss, loss_main, loss_reco, loss_regu,adv_loss = model.cal_loss_mutual(self.discriminator,batch_data, sum(source_all_item_embeddings), source_all_item_embeddings, pos, neg)
            loss_his.append(loss_main)
            loss+=adv_loss
            if i % configs['model']['mask_steps'] == 0:
                reward = self.calc_reward(loss_his, configs['model']['eps'])
                loss_mask = -sample_scr.mean() * reward
                loss_log_dict['loss_mask'] += loss_mask / (len(trn_loader) // configs['model']['mask_steps'])
                loss_his = loss_his[-1:]
                loss += loss_mask

            loss_log_dict['loss'] += loss.item() / len(trn_loader)
            loss_log_dict['loss_main'] += loss_main.item() / len(trn_loader)
            loss_log_dict['loss_reco'] += loss_reco.item() / len(trn_loader)
            loss_log_dict['loss_regu'] += loss_regu.item() / len(trn_loader)
            loss_log_dict['loss_adv'] += adv_loss.item() / len(trn_loader)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        writer.add_scalar('Loss/train', loss_log_dict['loss'], epoch_idx)

        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)
        return loss

    def train_epoch_no_grad(self, model, epoch_idx,source_item_embeddings_para,source_all_item_embeddings):
        model.eval()
        self.discriminator.eval()
        with torch.no_grad():
            loss_his = []
            loss_log_dict = {'loss': 0, 'loss_main': 0, 'loss_reco': 0, 'loss_regu': 0, 'loss_mask': 0,'loss_adv':0}
            trn_loader = self.data_handler.train_dataloader
            trn_loader.dataset.sample_negs()

            for i, batch_data in tqdm(enumerate(trn_loader), desc='Training MAERec', total=len(trn_loader)):
                if i % configs['model']['mask_steps'] == 0:
                    sample_scr, candidates = model.sampler(model.ii_adj_all_one, source_item_embeddings_para)
                    masked_adj, masked_edg = model.masker(model.ii_adj, candidates)

                batch_data = list(map(lambda x: x.long().to(configs['device']), batch_data))

                # item_emb, item_emb_his = model.encoder(masked_adj)
                pos = self.sample_pos_edges(masked_edg)
                neg = self.sample_neg_edges(pos, model.ii_dok)

                loss, loss_main, loss_reco, loss_regu,adv_loss = model.cal_loss_mutual(self.discriminator,batch_data, sum(source_all_item_embeddings), source_all_item_embeddings, pos, neg)
                loss_his.append(loss_main)
                loss+=adv_loss
                if i % configs['model']['mask_steps'] == 0:
                    reward = self.calc_reward(loss_his, configs['model']['eps'])
                    loss_mask = -sample_scr.mean() * reward
                    loss_log_dict['loss_mask'] += loss_mask / (len(trn_loader) // configs['model']['mask_steps'])
                    loss_his = loss_his[-1:]
                    loss += loss_mask

                loss_log_dict['loss'] += loss.item() / len(trn_loader)
                loss_log_dict['loss_main'] += loss_main.item() / len(trn_loader)
                loss_log_dict['loss_reco'] += loss_reco.item() / len(trn_loader)
                loss_log_dict['loss_regu'] += loss_regu.item() / len(trn_loader)
                loss_log_dict['loss_adv'] += adv_loss.item() / len(trn_loader)

            writer.add_scalar('Loss/train', loss_log_dict['loss'], epoch_idx)

            if configs['train']['log_loss']:
                self.logger.log_loss(epoch_idx, loss_log_dict)
            else:
                self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)
            return loss

