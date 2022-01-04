# Code base: https://github.com/THUwangcy/ReChorus/blob/master/src/models/sequential/GRU4Rec.py

import torch
import torch.nn as nn

from model.fed_rec.general_fed_rec import FederatedRecModel
from model.components import FieldEmbeddings, init_weights, DNN

""" GRU4Rec
Reference:
    "Session-based Recommendations with Recurrent Neural Networks"
    Hidasi et al., ICLR'2016.
"""


class FedGRU(FederatedRecModel):
    
    ########################
    #       GRU4Rec        #
    ########################
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - hidden_emb_size
        - from GeneralRecModel:
            - emb_size
            - from BaseModel:
                - model_path
                - loss
                - l2_coef
                - cuda
                - optimizer
                - n_worker
        '''
        parser = FederatedRecModel.parse_model_args(parser)
        parser.add_argument('--hidden_emb_size', type=int, default=32,
                            help='Size of hidden vectors in GRU.')
        return parser
    
    @staticmethod
    def get_reader():
        return "FedSeqRecReader"
    
    def log(self):
        super().log()
        print("\thidden_emb_size = " + str(self.hidden_emb_size))

    def _define_params(self, args, reader):
        self.hidden_emb_size = args.hidden_emb_size
        self.i_embeddings = nn.Embedding(reader.n_items, args.emb_size, padding_idx=0)
        self.rnn = nn.GRU(input_size=args.emb_size, hidden_size=args.hidden_emb_size, batch_first=True)
        self.pred_embeddings = nn.Embedding(reader.n_items, args.hidden_emb_size, padding_idx=0)
        for module in [self.i_embeddings, self.pred_embeddings]:
            init_weights(module)      

    def get_forward(self, feed_dict):
        self.check_list = []
        history = feed_dict['Sequence']  # [B, max(|H|)]
        B = history.shape[0]
        i_ids = feed_dict['ItemID'].view(B,-1)  # [B, N]
#         lengths = feed_dict['SeqLen']  # [B]
        lengths = torch.LongTensor([history.shape[1]] * history.shape[0]).to(self.device)

        his_vectors = self.i_embeddings(history)

        # Sort and Pack
        sort_his_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
        sort_his_vectors = his_vectors.index_select(dim=0, index=sort_idx)
        history_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_his_vectors, sort_his_lengths.cpu(), batch_first=True) # [B, sum(|H|)]

        # RNN
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(history_packed, None)
#         output, hidden = self.rnn(his_vectors, None)

        # Unsort
        unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
        rnn_vector = hidden[-1].index_select(dim=0, index=unsort_idx).view(-1,1,self.hidden_emb_size) # (B,1,d)

        # Predicts
        pred_vectors = self.pred_embeddings(i_ids)
        # pred_vectors = self.i_embeddings(i_ids)
        prediction = (rnn_vector * pred_vectors.view(B,-1,self.hidden_emb_size)).sum(-1) # (B,N)
        prediction = prediction + self.global_bias
        
        # regularization
        reg = self.get_regularization(self.rnn)
        reg += torch.mean(his_vectors * his_vectors) + torch.mean(pred_vectors * pred_vectors)
        
        return {'preds': prediction.view(i_ids.shape[0], -1), "reg": reg}
    
    ###################################
    #       Federation Control        #
    ###################################
    
#     def keep_cloud_params(self):
#         super().keep_cloud_params()
        
#     def download_cloud_params(self, local_info):
#         super().download_cloud_params(local_info)
        
    def actions_before_epoch(self, info):
        self.param_proposal = {k: torch.zeros_like(v) for k,v in self.cloud_params.items()}
        self.param_proposal_count = {}
        for k,v in self.cloud_params.items():
            if 'embeddings' in k:
                self.param_proposal_count[k] = torch.zeros(v.shape[0]).to(torch.long).to(self.device)
            else:
                self.param_proposal_count[k] = torch.tensor(0).to(torch.long).to(self.device)
        
    def upload_edge_params(self, local_info):
        '''
        Upload edge parameters to cloud
        '''
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'rnn' in name:
                    self.param_proposal[name] += param.data.clone()
                    self.param_proposal_count[name] += 1
            for key in local_info:
                if "item" in key:
                    item_ids = local_info[key].view(-1)
                    self.param_proposal["i_embeddings.weight"][item_ids] += self.i_embeddings.weight.data[item_ids].clone()
                    self.param_proposal["pred_embeddings.weight"][item_ids] += self.pred_embeddings.weight.data[item_ids].clone()
                    self.param_proposal_count["i_embeddings.weight"][item_ids] += 1
                    self.param_proposal_count["pred_embeddings.weight"][item_ids] += 1
        self.local_info = []
        
#     def mitigate_params(self):
#         super().mitigate_params()