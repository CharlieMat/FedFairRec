# import numpy as np
# import torch
# import torch.nn as nn

# import model.general_model as gm
# from model.general_model import GeneralRecModel 

# class MF_MMR(GeneralRecModel):
#     @staticmethod
#     def parse_model_args(parser):
#         parser.add_argument('--emb_size', type=int, default=16,
#                             help='Size of embedding vectors.')
#         return GeneralRecModel.parse_model_args(parser)
        
#     @staticmethod
#     def make_model_path(args):
#         return "mf_" + args.loss_type +  ('_l2%.5f' % args.l2_coef) + "_dim" + str(args.emb_size)

#     def log_model(self, logger):
#         super().log_model(logger)
#         logger.log("\tembSize: " + str(self.embSize))
        
#     def __init__(self, args, logger, reader):
#         self.embSize = args.emb_size
#         super().__init__(args, logger, reader)
#         self.modelName = "mf"
        
#     def _define_params(self):
#         super()._define_params()
#         self.uEmb = nn.Embedding(self.nUser, self.embSize)
#         self.iEmb = nn.Embedding(self.nItem, self.embSize)
#         self.uBias = nn.Embedding(self.nUser, 1)
#         self.iBias = nn.Embedding(self.nItem, 1)
#         gm.init_weights(self.uEmb, {"dim": self.embSize}, m_type = "embedding")
#         gm.init_weights(self.iEmb, {"dim": self.embSize}, m_type = "embedding")
#         gm.init_weights(self.uBias, m_type = "bias")
#         gm.init_weights(self.iBias, m_type = "bias")
        
#     def get_customized_parameters(self):
#         return self.parameters()

#     def forward_recordwise(self, feed_dict):
#         '''
#         @input:
#         - feed_dict: {"user": (B,), "item": (B,L), "batch_size": B}
#         @output:
#         - result_dict: {"preds": (B,L), "reg": scalar}
#         '''
# #         self.check_list = []
#         B = feed_dict["batch_size"]
#         userIDs = feed_dict['user'].view(B)  # (B,)
#         itemIDs = feed_dict['item'].view(B,-1)  # (B,L)
#         L = itemIDs.shape[1]

#         # get embeddings
#         userEmb = self.uEmb(userIDs).view(B,1,self.embSize) # (B,1,d)
#         itemEmb = self.iEmb(itemIDs) # (B,L,d)
#         # get user and item bias
#         userBias = self.uBias(userIDs).view(B,1) # (B,1)
#         itemBias = self.iBias(itemIDs).view(B,L) # (B,L)
#         # prediction
#         prediction = torch.sum(userEmb * itemEmb, -1)  # (B,L)
#         prediction = prediction + userBias + itemBias
#         prediction = prediction + self.globalBias
#         # regularization terms
#         reg = torch.mean(torch.sum(userEmb * userEmb,-1)) + torch.mean(torch.sum(itemEmb * itemEmb,-1))
        
#         return {'preds': prediction,  "reg": reg}