import torch
import torch.nn as nn
import numpy as np

from model.general import BaseModel
    
RECMODEL_LOSSTYPES = ["regression", "pointwise", "pairwisebpr", "pairwisemrl", "softmax"]

class GeneralRecModel(BaseModel):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - from BaseModel:
            - model_path
            - loss
            - l2_coef
            - cuda
            - optimizer
            - n_worker
        '''
        parser = BaseModel.parse_model_args(parser)
        parser.add_argument('--emb_size', type=int, default=16,
                            help='item/user embedding size')
        return parser
    
    @staticmethod
    def get_reader():
        return "RecDataReader"
    
    def log(self):
        super().log()
        print("\temb_size = " + str(self.emb_size))
        
    def __init__(self, args, reader, device):
        self.emb_size = args.emb_size
        self.n_ufields = len(reader.user_fields)
        self.n_ifields = len(reader.item_fields)
        assert args.loss in RECMODEL_LOSSTYPES
        super().__init__(args, reader, device)
        
        if self.loss_type == "regression":
            bias = torch.ones((1,1), requires_grad = True) * reader.get_average_response()
        else:
            bias = torch.zeros((1,1), requires_grad = True)
        self.global_bias = torch.nn.Parameter(bias)
        
        self.mse_loss = nn.MSELoss(reduction = 'none')
        self.mrl_loss = nn.MarginRankingLoss(margin = 0.2, reduction = 'none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction = 'none')
        self.ce_loss = nn.CrossEntropyLoss(reduction = 'none')
        
    def wrap_batch(self, batch):
        '''
        Build feed_dict from batch data and move data to self.device
        '''
        batch["resp"] = batch["resp"].to(torch.float)
        return super().wrap_batch(batch)
    
    def get_loss(self, feed_dict: dict, out_dict: dict):
        """
        @input:
        - feed_dict: {"resp":, ...}
        - out_dict: {"preds":, "neg_preds":, "reg":, "neg_reg":,}
        
        Loss terms implemented:
        - regression: for rating prediction
        - pairwisebpr, pairwisemrl, pointwise: ranking
        
        Loss terms not implemented:
        - cross-entropy: for classification
        """
        
        preds, reg = out_dict["preds"].view(-1), out_dict["reg"] # size (B,)

        # loss
        if self.loss_type == "regression":
            resps = feed_dict["resp"] # size (B,)
            loss = torch.mean(self.mse_loss(preds, resps.view(-1))) + self.l2_coef * reg
            return loss
        else:
            B = preds.shape[0]
            neg_preds = out_dict["neg_preds"].view(-1) # size (B*L,)
            neg_reg = out_dict["neg_reg"] # scalar
            ratio = len(neg_preds) / len(preds) # number of negative samples per positive sample
            extended_preds = preds.view(-1,1).repeat(1,int(ratio)).view(-1)
            if self.loss_type == "pairwisebpr":
                loss = torch.mean(self.sigmoid(neg_preds - extended_preds))
            elif self.loss_type == "pairwisemrl":
                y = torch.ones_like(neg_preds)
                mrl = self.mrl_loss(self.sigmoid(extended_preds), self.sigmoid(neg_preds), y)
                loss = torch.mean(mrl)
            elif self.loss_type == "pointwise":
                pos_target = torch.ones_like(extended_preds)
                neg_target = torch.zeros_like(neg_preds)
                loss = torch.mean(self.bce_loss(self.sigmoid(extended_preds), pos_target)) + \
                        torch.mean(self.bce_loss(self.sigmoid(neg_preds), neg_target))
            elif self.loss_type == "softmax":
                labels = torch.tensor([0] * B).to(self.device)
                combined_preds = torch.cat([preds.view(B,1),neg_preds.view(B,-1)], dim = 1)
                loss = torch.mean(self.ce_loss(combined_preds, labels))
            else:
                raise NotImplemented
            loss = loss + self.l2_coef * (reg * ratio + neg_reg)
            return loss
    
    def forward(self, feed_dict: dict, return_prob = True) -> dict:
        '''
        Called during evaluation or prediction
        '''
        if self.loss_type == "regression":
            out_dict = self.get_forward({k[5:]: v for k,v in feed_dict.items() if "user_" in k or "item_" in k})
            if return_prob:
                out_dict["probs"] = nn.Sigmoid()(out_dict["preds"])
        else:
            out_dict = self.get_forward({k[5:]: v for k,v in feed_dict.items() if "user_" in k or "item_" in k})
            neg_out_dict = self.get_forward({k[5:]: v for k,v in feed_dict.items() if "user_" in k or "negi_" in k})
            for k,v in neg_out_dict.items():
                out_dict["neg_" + k] = v
            if return_prob:
                out_dict["probs"] = nn.Sigmoid()(out_dict["preds"])
                out_dict["neg_probs"] = nn.Sigmoid()(out_dict["neg_preds"])
        return out_dict
    
    #############################
    #   Require Implementation  #
    #############################
    # from BaseModel
#     def _define_params(self, args, reader) -> NoReturn:
#         pass
    
#     def get_forward(self, feed_dict: dict) -> dict:
#         pass
    
    # Code base: https://github.com/THUwangcy/ReChorus/blob/master/src/models/sequential/GRU4Rec.py
