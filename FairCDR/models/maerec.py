import copy
import torch as t
import numpy as np
from torch import nn
import scipy.sparse as sp
import os
from torch_geometric.nn import GATConv,GCNConv
from scipy.sparse import save_npz, load_npz
from config.configurator import configs
from models.base_model import BaseModel
from models.model_utils import TransformerLayer
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
init = nn.init.xavier_uniform_
uniform_init = nn.init.uniform
import torch.sparse as sparse
def sparse_dropout(x, keep_prob):
    msk = (t.rand(x._values().size()) + keep_prob).floor().type(t.bool)
    idx = x._indices()[:, msk]
    val = x._values()[msk]
    return t.sparse.FloatTensor(idx, val, x.shape).cuda()

class TransformerEmbed(nn.Module):
    def __init__(self, item_num, emb_size, max_len, dropout=0.1):
        super().__init__()
        self.position_emb = nn.Embedding(max_len, emb_size)
        self.dropout = nn.Dropout(p=dropout)
        self.emb_size = emb_size
        self.apply(self._init_weights)

    def forward(self, batch_seqs, item_emb):
        batch_size = batch_seqs.size(0)
        pos_emb = self.position_emb.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        x = item_emb[batch_seqs] + pos_emb
        return self.dropout(x)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.item_emb = nn.Parameter(init(t.empty(configs['data']['item_num']+ configs['target_data']['item_num']+ 1, configs['model']['embedding_size'])))
        self.gcn_layers = nn.Sequential(*[GCNLayer(configs['model']['embedding_size'],configs['model']['embedding_size']) for i in range(configs['model']['num_gcn_layers'])])

    def get_ego_embeds(self):
        return self.item_emb

    def forward(self, encoder_adj):
        item_nums=configs["data"]["item_num"]+1
        embeds = [self.item_emb[:item_nums]]
        for i, gcn in enumerate(self.gcn_layers):
            embeds.append(gcn(encoder_adj, embeds[-1]))
        return sum(embeds), embeds
    
class Encoder_target(nn.Module):
    def __init__(self, shared_item_emb):
        super(Encoder_target, self).__init__()
        self.item_emb = shared_item_emb
        self.gcn_layers = nn.Sequential(*[GCNLayer(configs['model']['embedding_size'],configs['model']['embedding_size']) for i in range(configs['model']['num_gcn_layers'])])

    def get_ego_embeds(self):
        return self.item_emb

    def forward(self, encoder_adj):
        with t.no_grad():
            item_nums=configs["target_data"]["item_num"]+1
            embeds = [self.item_emb[-item_nums:]]
            for i, gcn in enumerate(self.gcn_layers):
                embeds.append(gcn(encoder_adj, embeds[-1]))
            return sum(embeds), embeds

class TrivialDecoder(nn.Module):
    def __init__(self):
        super(TrivialDecoder, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(configs['model']['embedding_size'] * 3, configs['model']['embedding_size'], bias=True),
            nn.ReLU(),
            nn.Linear(configs['model']['embedding_size'], 1, bias=True),
            nn.Sigmoid()
        )
        self.apply(self.init_weights)

    def forward(self, embeds, pos, neg):
        pos_emb, neg_emb = [], []
        pos_emb.append(embeds[-1][pos[:,0]])
        pos_emb.append(embeds[-1][pos[:,1]])
        pos_emb.append(embeds[-1][pos[:,0]] * embeds[-1][pos[:,1]])
        neg_emb.append(embeds[-1][neg[:,:,0]])
        neg_emb.append(embeds[-1][neg[:,:,1]])
        neg_emb.append(embeds[-1][neg[:,:,0]] * embeds[-1][neg[:,:,1]])
        pos_emb = t.cat(pos_emb, -1) 
        neg_emb = t.cat(neg_emb, -1) 
        pos_scr = t.exp(t.squeeze(self.MLP(pos_emb)))
        neg_scr = t.exp(t.squeeze(self.MLP(neg_emb)))
        neg_scr = t.sum(neg_scr, -1) + pos_scr
        loss = -t.sum(pos_scr / (neg_scr + 1e-8) + 1e-8)
        return loss

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            init(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(configs['model']['embedding_size'] * configs['model']['num_gcn_layers'] ** 2, configs['model']['embedding_size'] * configs['model']['num_gcn_layers'], bias=True),
            nn.ReLU(),
            nn.Linear(configs['model']['embedding_size'] * configs['model']['num_gcn_layers'], configs['model']['embedding_size'], bias=True),
            nn.ReLU(),
            nn.Linear(configs['model']['embedding_size'], 1, bias=True),
            nn.Sigmoid()
        )
        self.apply(self.init_weights)

    def forward(self, embeds, pos, neg):
        pos_emb, neg_emb = [], []
        for i in range(configs['model']['num_gcn_layers']):
            for j in range(configs['model']['num_gcn_layers']):
                pos_emb.append(embeds[i][pos[:,0]] * embeds[j][pos[:,1]])
                neg_emb.append(embeds[i][neg[:,:,0]] * embeds[j][neg[:,:,1]])
        pos_emb = t.cat(pos_emb, -1)
        neg_emb = t.cat(neg_emb, -1)
        pos_scr = t.exp(t.squeeze(self.MLP(pos_emb))) 
        neg_scr = t.exp(t.squeeze(self.MLP(neg_emb)))
        neg_scr = t.sum(neg_scr, -1) + pos_scr
        loss = -t.sum(t.log(pos_scr / (neg_scr + 1e-8) + 1e-8))
        return loss

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            init(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    
class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.gat = GCNConv(input_dim, output_dim)

    def forward(self, edge_index, x):
        edge_index=edge_index.to(configs["device"])
        return self.gat(x, edge_index)
    
class GATLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=1):
        super(GATLayer, self).__init__()
        self.gat = GATConv(input_dim, output_dim, heads=num_heads)

    def forward(self, edge_index, x):
        edge_index=edge_index.to(configs["device"])
        return self.gat(x, edge_index)

class LocalGraph(nn.Module):
    def __init__(self):
        super(LocalGraph, self).__init__()

    def make_noise(self, scores):
        noise = t.rand(scores.shape).cuda()
        noise = -t.log(-t.log(noise))
        return scores + noise

    def forward(self, adj, embeds, foo=None):
        item_nums=adj.shape[0]
        embeds=embeds[:item_nums]
        order = t.sparse.sum(adj, dim=-1).to_dense().view([-1, 1])
        fstEmbeds = t.spmm(adj, embeds) - embeds
        fstNum = order

        emb = [fstEmbeds]
        num = [fstNum]

        for i in range(configs['model']['mask_depth']):
            adj = sparse_dropout(adj, configs['model']['path_prob'] ** (i + 1))
            emb.append((t.spmm(adj, emb[-1]) - emb[-1]) - order * emb[-1])
            num.append((t.spmm(adj, num[-1]) - num[-1]) - order)
            order = t.sparse.sum(adj, dim=-1).to_dense().view([-1, 1])

        subgraphEmbeds = sum(emb) / (sum(num) + 1e-8)
        subgraphEmbeds = F.normalize(subgraphEmbeds, p=2)

        embeds = F.normalize(embeds, p=2)
        scores = t.sum(subgraphEmbeds * embeds, dim=-1)
        scores = self.make_noise(scores)

        _, candidates = t.topk(scores, configs['model']['num_mask_cand'])

        return scores, candidates

class RandomMaskSubgraphs(nn.Module):
    def __init__(self):
        super(RandomMaskSubgraphs, self).__init__()

    def normalize(self, adj):
        degree = t.pow(t.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
        newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
        rowNorm, colNorm = degree[newRows], degree[newCols]
        newVals = adj._values() * rowNorm * colNorm
        return t.sparse.FloatTensor(adj._indices(), newVals, adj.shape)

    def forward(self, adj, seeds):
        rows = adj._indices()[0, :]
        cols = adj._indices()[1, :]

        masked_rows = []
        masked_cols = []
        masked_idct = []

        for i in range(configs['model']['mask_depth']):
            curSeeds = seeds if i == 0 else nxtSeeds
            nxtSeeds = list()
            idct = None
            for seed in curSeeds:
                rowIdct = (rows == seed)
                colIdct = (cols == seed)
                if idct == None:
                    idct = t.logical_or(rowIdct, colIdct)
                else:
                    idct = t.logical_or(idct, t.logical_or(rowIdct, colIdct))
            nxtRows = rows[idct]
            nxtCols = cols[idct]
            masked_rows.extend(nxtRows)
            masked_cols.extend(nxtCols)
            rows = rows[t.logical_not(idct)]
            cols = cols[t.logical_not(idct)]
            nxtSeeds = nxtRows + nxtCols
            if len(nxtSeeds) > 0 and i != configs['model']['mask_depth'] - 1:
                nxtSeeds = t.unique(nxtSeeds)
                cand = t.randperm(nxtSeeds.shape[0])
                nxtSeeds = nxtSeeds[cand[:int(nxtSeeds.shape[0] * configs['model']['path_prob'] ** (i + 1))]]

        masked_rows = t.unsqueeze(t.LongTensor(masked_rows), -1)
        masked_cols = t.unsqueeze(t.LongTensor(masked_cols), -1)
        masked_edge = t.hstack([masked_rows, masked_cols])
        encoder_adj = self.normalize(t.sparse.FloatTensor(t.stack([rows, cols], dim=0), t.ones_like(rows).cuda(), adj.shape))

        return encoder_adj, masked_edge


class MAERec(BaseModel):
    def __init__(self, data_handler):
        super(MAERec, self).__init__(data_handler)
        self.data_handler = data_handler
        self.user_num = configs['data']['user_num']
        self.item_num = configs['data']['item_num'] + 1
        self.target_item_num = configs['target_data']['item_num'] + 1
        self.con_batch = configs['model']['con_batch']
        self.max_seq_len = configs['model']['max_seq_len']
        self.num_reco_neg = configs['model']['num_reco_neg']
        self.reg = configs['model']['reg']
        self.ssl_reg = configs['model']['ssl_reg']
        self.embedding_size = configs['model']['embedding_size']
        self.mask_depth = configs['model']['mask_depth']
        self.path_prob = configs['model']['path_prob']
        self.num_attention_heads = configs['model']['num_attention_heads']
        self.num_gcn_layers = configs['model']['num_gcn_layers']
        self.num_trm_layers = configs['model']['num_trm_layers']
        self.num_mask_cand = configs['model']['num_mask_cand']
        self.mask_steps = configs['model']['mask_steps']
        self.eps = configs['model']['eps']
        self.attention_probs_dropout_prob = configs['model']['attention_probs_dropout_prob']
        self.hidden_dropout_prob = configs['model']['hidden_dropout_prob']
        self.loss_func = nn.CrossEntropyLoss()
        self.config=configs

        self.construct_graphs(data_handler)
        self.construct_graphs_target(data_handler)
        self.prepare_model()

    def normalize(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()



    def construct_graphs(self, handler, distance=3):
        print("Constructing source i-i graph...")
        graph_path = f'./graphs/{self.config["data"]["name"]}_{self.config["target_data"]["name"]}/{self.config["data"]["name"]}/ii_graph.npz'
        edge_index_path = f'./graphs/{self.config["data"]["name"]}_{self.config["target_data"]["name"]}/{self.config["data"]["name"]}/edge_index.pt'
        
        if os.path.exists(graph_path) and os.path.exists(edge_index_path):
            print(f"Graph file found at {graph_path}. Loading...")
            iigraph = load_npz(graph_path)
            edge_index = t.load(edge_index_path)
        else:
            print(f"Graph file not found at {graph_path}. Constructing...")
            seqs = handler.train_dataloader.dataset.seqs
            r, c, d = list(), list(), list()
            for i, seq in enumerate(seqs):
                for dist in range(1, distance + 1):
                    if dist >= len(seq): 
                        break
                    r += copy.deepcopy(seq[+dist:])
                    c += copy.deepcopy(seq[:-dist])
                    r += copy.deepcopy(seq[:-dist])
                    c += copy.deepcopy(seq[+dist:])
            pairs = np.unique(np.array(list(zip(r, c))), axis=0)
            r, c = pairs.T
            d = np.ones_like(r)
            iigraph = sp.csr_matrix((d, (r, c)), shape=(self.item_num, self.item_num))
            print('Constructed i-i graph, density=%.6f' % (len(d) / (self.item_num ** 2)))
            edge_index = t.tensor([r, c], dtype=t.long)
            edge_index = t.cat([edge_index, edge_index.flip(0)], dim=1)
            os.makedirs(os.path.dirname(graph_path), exist_ok=True)
            save_npz(graph_path, iigraph)
            t.save(edge_index, edge_index_path)
            print(f"Graph saved to {graph_path}")
            print(f"Edge index saved to {edge_index_path}")
        self.ii_dok = iigraph.todok()
        self.ii_adj = self.make_torch_adj(iigraph)
        self.ii_adj_all_one = self.make_all_one_adj(self.ii_adj)
        self.edge_index = edge_index

        return edge_index

    def construct_graphs_target(self, handler, distance=3):
        print("Constructing target i-i graph...")
        graph_path = f'./graphs/{self.config["data"]["name"]}_{self.config["target_data"]["name"]}/{self.config["target_data"]["name"]}/ii_graph.npz'
        edge_index_path = f'./graphs/{self.config["data"]["name"]}_{self.config["target_data"]["name"]}/{self.config["target_data"]["name"]}/edge_index_target.pt'  
        if os.path.exists(graph_path) and os.path.exists(edge_index_path):

            print(f"Graph file found at {graph_path}. Loading...")
            iigraph = load_npz(graph_path)
            edge_index = t.load(edge_index_path)
        else:
            print(f"Graph file not found at {graph_path}. Constructing...")
            seqs = handler.train_dataloader_target.dataset.seqs
            r, c, d = list(), list(), list()
            for i, seq in enumerate(seqs):
                for dist in range(1, distance + 1):
                    if dist >= len(seq): 
                        break
                    r += copy.deepcopy(seq[+dist:])
                    c += copy.deepcopy(seq[:-dist])
                    r += copy.deepcopy(seq[:-dist])
                    c += copy.deepcopy(seq[+dist:])
            pairs = np.unique(np.array(list(zip(r, c))), axis=0)
            r, c = pairs.T
            d = np.ones_like(r)
            r = r - self.item_num + 1
            c = c - self.item_num + 1
            iigraph = sp.csr_matrix((d, (r, c)), shape=(self.target_item_num, self.target_item_num))
            print('Constructed i-i graph, density=%.6f' % (len(d) / ((self.target_item_num) ** 2)))
            edge_index = t.tensor([r, c], dtype=t.long)
            edge_index = t.cat([edge_index, edge_index.flip(0)], dim=1)
            os.makedirs(os.path.dirname(graph_path), exist_ok=True)
            save_npz(graph_path, iigraph)
            t.save(edge_index, edge_index_path)
            print(f"Graph saved to {graph_path}")
            print(f"Edge index saved to {edge_index_path}")
        self.target_ii_dok = iigraph.todok()
        self.target_ii_adj = self.make_torch_adj(iigraph)
        self.target_ii_adj_all_one = self.make_all_one_adj(self.target_ii_adj)
        self.target_edge_index = edge_index

        return edge_index

    def make_torch_adj(self, mat):
        mat = (mat + sp.eye(mat.shape[0]))
        mat = (mat != 0) * 1.0
        mat = self.normalize(mat)
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()

    def make_all_one_adj(self, adj):
        idxs = adj._indices()
        vals = t.ones_like(adj._values())
        shape = adj.shape
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()
    
    def prepare_model(self):
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        
        self.emb_layer = TransformerEmbed(
            self.item_num+ self.target_item_num+ 2, self.embedding_size, self.max_seq_len)
        self.transformer_layers = nn.ModuleList([TransformerLayer(
            self.embedding_size, self.num_attention_heads, self.embedding_size * 4, self.hidden_dropout_prob) for _ in range(self.num_trm_layers)])
        self.masker = RandomMaskSubgraphs()
        self.sampler = LocalGraph()

    def forward(self, batch_seqs):
        item_emb, _ = self.encoder(self.edge_index)
        mask = (batch_seqs > 0).unsqueeze(1).repeat(
            1, batch_seqs.size(1), 1).unsqueeze(1)
        x = self.emb_layer(batch_seqs, item_emb)
        for transformer in self.transformer_layers:
            x = transformer(x, mask)
        output = x[:, -1, :]
        return output

    def forward_mutual(self, batch_seqs,item_emb):
        mask = (batch_seqs > 0).unsqueeze(1).repeat(
            1, batch_seqs.size(1), 1).unsqueeze(1)
        x = self.emb_layer(batch_seqs, item_emb)
        for transformer in self.transformer_layers:
            x = transformer(x, mask)
        output = x[:, -1, :]
        return output

    def calc_reg_loss(self, model):
        ret = 0
        for W in model.parameters():
            ret += W.norm(2).square()
        return ret
    
    def cal_loss(self, batch_data, item_emb, item_emb_his, pos, neg):
        _, batch_seqs, batch_last_items,gender = batch_data
        seq_output = self.forward(batch_seqs)

        logits = t.matmul(seq_output, item_emb.transpose(0, 1))
        loss_main = self.loss_func(logits, batch_last_items)
        loss_reco = self.decoder(item_emb_his, pos, neg) * self.ssl_reg
        loss_regu = (self.calc_reg_loss(self.encoder) + 
                     self.calc_reg_loss(self.decoder) + 
                     self.calc_reg_loss(self.emb_layer) + 
                     self.calc_reg_loss(self.transformer_layers)
                    ) * self.reg
        loss = loss_main + loss_reco + loss_regu

        return loss, loss_main, loss_reco, loss_regu
    
    def cal_loss_mutual(self,discriminator, batch_data, item_emb, item_emb_his, pos, neg):
        _, batch_seqs, batch_last_items,gender = batch_data
        seq_output = self.forward_mutual(batch_seqs,item_emb)

        logits = t.matmul(seq_output, item_emb.transpose(0, 1))
        loss_main = self.loss_func(logits, batch_last_items)
        loss_reco = self.decoder(item_emb_his, pos, neg) * self.ssl_reg
        loss_regu = (self.calc_reg_loss(self.encoder) + 
                     self.calc_reg_loss(self.decoder) + 
                     self.calc_reg_loss(self.emb_layer) + 
                     self.calc_reg_loss(self.transformer_layers)
                    ) * self.reg
        loss = loss_main + loss_reco + loss_regu
        gender_labels = gender.long()
        discriminator_output = discriminator(seq_output)
        adversarial_loss = -t.mean(t.log_softmax(discriminator_output, dim=1)[t.arange(len(gender_labels)), gender_labels])
        return loss, loss_main, loss_reco, loss_regu,3*adversarial_loss
    
    def dis_loss(self,discriminator, batch_data):
        _, batch_seqs, batch_last_items,gender_labels = batch_data
        seq_output = self.forward(batch_seqs)
        gender_labels = gender_labels.long()
        discriminator_output = discriminator(seq_output)
        adversarial_loss = -t.mean(t.log_softmax(discriminator_output, dim=1)[t.arange(len(gender_labels)), gender_labels])    
        probabilities = F.softmax(discriminator_output, dim=1).detach().cpu().numpy()
        true_labels = gender_labels.cpu().numpy()
        auc_score = roc_auc_score(true_labels, probabilities, multi_class='ovo')
        return adversarial_loss,auc_score 

    def dis_loss_mutual(self,discriminator, batch_data,item_emb):
        _, batch_seqs, batch_last_items,gender_labels = batch_data
        seq_output = self.forward_mutual(batch_seqs,item_emb)
        gender_labels = gender_labels.long()
        discriminator_output = discriminator(seq_output)
        adversarial_loss = -t.mean(t.log_softmax(discriminator_output, dim=1)[t.arange(len(gender_labels)), gender_labels])    
        probabilities = F.softmax(discriminator_output, dim=1).detach().cpu().numpy()
        true_labels = gender_labels.cpu().numpy()
        auc_score = roc_auc_score(true_labels, probabilities, multi_class='ovo')
        return adversarial_loss,auc_score 
    
    def full_predict(self,discriminator, batch_data):
        _, batch_seqs, _ ,gender_labels= batch_data
        logits = self.forward(batch_seqs)
        gender_labels = gender_labels.long()
        discriminator_output = discriminator(logits)
        probabilities = F.softmax(discriminator_output, dim=1).detach().cpu().numpy()
        true_labels = gender_labels.cpu().numpy()
        auc_score = roc_auc_score(true_labels, probabilities, multi_class='ovo')
        item_emb, _ = self.encoder(self.edge_index)
        scores = t.matmul(logits, item_emb.transpose(0, 1))
        return scores,auc_score
    
    def full_predict_mutual(self,discriminator, batch_data,item_emb):
        _, batch_seqs, _ ,gender_labels= batch_data
        logits = self.forward_mutual(batch_seqs,item_emb)
        gender_labels = gender_labels.long()
        discriminator_output = discriminator(logits)
        probabilities = F.softmax(discriminator_output, dim=1).detach().cpu().numpy()
        true_labels = gender_labels.cpu().numpy()
        auc_score = roc_auc_score(true_labels, probabilities, multi_class='ovo')
        scores = t.matmul(logits, item_emb.transpose(0, 1))
        return scores,auc_score

    def full_predict_mutual_target(self,discriminator, batch_data,item_emb):
        _, batch_seqs, _ ,gender_labels= batch_data
        batch_seqs[batch_seqs > 0] -= self.item_num-1
        logits = self.forward_mutual(batch_seqs,item_emb)
        gender_labels = gender_labels.long()
        discriminator_output = discriminator(logits)
        probabilities = F.softmax(discriminator_output, dim=1).detach().cpu().numpy()
        true_labels = gender_labels.cpu().numpy()
        auc_score = roc_auc_score(true_labels, probabilities, multi_class='ovo')
        scores = t.matmul(logits, item_emb.transpose(0, 1))
        return scores,auc_score


    
    
    def _get_user_embeddings(self, batch_data):
        user_embs={}
        user_id, batch_seqs, _ ,gender_labels= batch_data
        item_emb, _ = self.encoder(self.edge_index)
        seq_item_emb = item_emb[batch_seqs]
        mask = (batch_seqs > 0).float().unsqueeze(-1)
        user_emb = (seq_item_emb * mask).sum(dim=1) / mask.sum(dim=1) 
        for i in range(len(user_id)):
                user_embs[user_id[i].item()]=user_emb[i]

        return user_embs
    
    def _get_target_user_embeddings(self, batch_data):
        with t.no_grad():
            user_embs={}
            user_id, batch_seqs, _ ,gender_labels= batch_data
            self.encoder_target = Encoder_target(shared_item_emb=self.encoder.item_emb).cuda()
            item_emb, _ = self.encoder_target(self.target_edge_index)
            batch_seqs[batch_seqs > 0] -= self.item_num-1
            seq_item_emb = item_emb[batch_seqs]
            mask = (batch_seqs > 0).float().unsqueeze(-1)
            user_emb = (seq_item_emb * mask).sum(dim=1) / mask.sum(dim=1) 
            for i in range(len(user_id)):
                user_embs[user_id[i].item()]=user_emb[i]


            return user_embs

    def _get_all_item_embeddings(self):
        with t.no_grad():
            _, item_emb= self.encoder(self.edge_index)
            return item_emb 

    def _get_target_item_embeddings(self):
        with t.no_grad():
            _, item_emb = self.encoder_target(self.target_edge_index)
            return item_emb 
        
    def _get_item_embeddings_para(self):
        with t.no_grad():
            item_emb = self.encoder.get_ego_embeds()
            return item_emb 
    
    def _get_target_item_embeddings_para(self):
        with t.no_grad():
            item_emb = self.encoder_target.get_ego_embeds()
            return item_emb 
        
    def _get_user_embeddings_mutual(self, batch_data,item_emb):
        user_embs={}
        user_id, batch_seqs, _ ,gender_labels= batch_data
        seq_item_emb = item_emb[batch_seqs]
        mask = (batch_seqs > 0).float().unsqueeze(-1)
        user_emb = (seq_item_emb * mask).sum(dim=1) / mask.sum(dim=1) 
        for i in range(len(user_id)):
                user_embs[user_id[i].item()]=user_emb[i]
        return user_embs