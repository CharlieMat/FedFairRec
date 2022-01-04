import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



    

    
class SequenceCapsule(nn.Module):
    def __init__(self, in_dim, out_dim, max_in_cap, n_out_cap, device, n_iteration = 2):
        '''
        Sequence capsule maps a sequence of embedding capsules to a fixed number of higher level capsules
        '''
        self.in_dim = in_dim                             # input capsule size
        self.out_dim = out_dim                           # output capsule size
        self.max_in_cap = max_in_cap                     # max number of input capsules
        self.n_out_cap = n_out_cap                       # number of output capsules
        self.n_iteration = n_iteration                   # number of rounting iterations
        super(SequenceCapsule, self).__init__()
        S = torch.randn(in_dim, out_dim, requires_grad = True)
        self.bilinear_weights = torch.nn.Parameter(S)    # bilinear weights for input and output capsules 
        b = torch.randn(max_in_cap, n_out_cap, requires_grad = True)
        self.initial_logits = torch.nn.Parameter(b)      # initial logits for rounting
        self.positions = torch.arange(self.max_in_cap,0,-1)[None, :].to(device) 
        self.default_pad_value = torch.tensor([-2**32+1.]).to(device) # padding value as small as possible for rounting weights
        
    def forward(self, inputs):
        '''
        @input:
        - inputs: {"capsules": [B,N,d] where d = self.in_dim, "lengths": (B,)}
        @output:
        - output_capsules: [B,N',d'], where d' = self.out_dim and N' = self.n_out_cap
        '''
        input_capsules = inputs["capsules"]
        B = input_capsules.shape[0]
        input_capsules = input_capsules.view(B,-1,self.in_dim) # (B,N,d)
        input_lengths = inputs["lengths"].view(B,1)
        rounting_logits = torch.tile(self.initial_logits.view(1,self.max_in_cap,self.n_out_cap), [B,1,1]) # (B,N,N')
        seq_mask = self.positions > input_lengths # (B,N), True for positions outside each sequence
        
        for i in range(self.n_iteration):
            # mask b
            rounting_logits = torch.where(seq_mask[:,:,None], self.default_pad_value[:,None,None], rounting_logits)
            # routing weights w = softmax(b)
            w = F.softmax(rounting_logits, dim = -1) # (B,N,N')
            # map input capsules, SC
            mapped_input = torch.matmul(input_capsules, self.bilinear_weights) # (B,N,d')
            # raw output Z = wSC
            Z = torch.bmm(mapped_input.transpose(1,2), w) # (B,d',N')
            # output capsules C' = squash(Z)
            output_capsules = self.squash(Z.transpose(1,2)) # (B,N',d')
            # b = b + c'Sc, update rounting logits for next iteration
            rounting_logits = rounting_logits + torch.bmm(mapped_input, Z) # (B,N,N')
        return output_capsules
            
    def squash(self, capsules):
        '''
        c' = (|z|^2/(1+|z|^2)) * (z /|z|)
        '''
        squared_norm = torch.sum(capsules ** 2, dim = -1, keepdim = True) # (B,N,1)
        scalar_factor = squared_norm / (squared_norm + 1) / torch.sqrt(squared_norm + 1e-8)
        return scalar_factor * capsules
        
        
        
# class ConvLayer(nn.Module):
#     def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
#         super(ConvLayer, self).__init__()

#         self.conv = nn.Conv2d(in_channels=in_channels,
#                               out_channels=out_channels,
#                               kernel_size=kernel_size,
#                               stride=1
#                               )

#     def forward(self, x):
#         return F.relu(self.conv(x))


# class PrimaryCaps(nn.Module):
#     def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, num_routes=32 * 6 * 6):
#         super(PrimaryCaps, self).__init__()
#         self.num_routes = num_routes
#         self.capsules = nn.ModuleList([
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
#             for _ in range(num_capsules)])

#     def forward(self, x):
#         u = [capsule(x) for capsule in self.capsules]
#         u = torch.stack(u, dim=1)
#         u = u.view(x.size(0), self.num_routes, -1)
#         return self.squash(u)

#     def squash(self, input_tensor):
#         squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
#         output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
#         return output_tensor


# class DigitCaps(nn.Module):
#     def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
#         super(DigitCaps, self).__init__()

#         self.in_channels = in_channels
#         self.num_routes = num_routes
#         self.num_capsules = num_capsules

#         self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

#     def forward(self, x):
#         batch_size = x.size(0)
#         x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

#         W = torch.cat([self.W] * batch_size, dim=0)
#         u_hat = torch.matmul(W, x)

#         b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
#         if USE_CUDA:
#             b_ij = b_ij.cuda()

#         num_iterations = 3
#         for iteration in range(num_iterations):
#             c_ij = F.softmax(b_ij, dim=1)
#             c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

#             s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
#             v_j = self.squash(s_j)

#             if iteration < num_iterations - 1:
#                 a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
#                 b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

#         return v_j.squeeze(1)

#     def squash(self, input_tensor):
#         squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
#         output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
#         return output_tensor


# class Decoder(nn.Module):
#     def __init__(self, input_width=28, input_height=28, input_channel=1):
#         super(Decoder, self).__init__()
#         self.input_width = input_width
#         self.input_height = input_height
#         self.input_channel = input_channel
#         self.reconstraction_layers = nn.Sequential(
#             nn.Linear(16 * 10, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, self.input_height * self.input_width * self.input_channel),
#             nn.Sigmoid()
#         )

#     def forward(self, x, data):
#         classes = torch.sqrt((x ** 2).sum(2))
#         classes = F.softmax(classes, dim=0)

#         _, max_length_indices = classes.max(dim=1)
#         masked = Variable(torch.sparse.torch.eye(10))
#         if USE_CUDA:
#             masked = masked.cuda()
#         masked = masked.index_select(dim=0, index=Variable(max_length_indices.squeeze(1).data))
#         t = (x * masked[:, :, None, None]).view(x.size(0), -1)
#         reconstructions = self.reconstraction_layers(t)
#         reconstructions = reconstructions.view(-1, self.input_channel, self.input_width, self.input_height)
#         return reconstructions, masked


# class CapsNet(nn.Module):
#     def __init__(self, in_dim, out_dim, in_channels, out_channels, config=None):
#         super(CapsNet, self).__init__()
#         if config:
#             self.conv_layer = ConvLayer(config.cnn_in_channels, config.cnn_out_channels, config.cnn_kernel_size)
#             self.primary_capsules = PrimaryCaps(config.pc_num_capsules, config.pc_in_channels, config.pc_out_channels,
#                                                 config.pc_kernel_size, config.pc_num_routes)
#             self.digit_capsules = DigitCaps(config.dc_num_capsules, config.dc_num_routes, config.dc_in_channels,
#                                             config.dc_out_channels)
#             self.decoder = Decoder(config.input_width, config.input_height, config.cnn_in_channels)
#         else:
#             self.conv_layer = ConvLayer()
#             self.primary_capsules = PrimaryCaps()
#             self.digit_capsules = DigitCaps()
#             self.decoder = Decoder()

#         self.mse_loss = nn.MSELoss()

#     def forward(self, data):
#         output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
#         reconstructions, masked = self.decoder(output, data)
#         return output, reconstructions, masked

#     def loss(self, data, x, target, reconstructions):
#         return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)

#     def margin_loss(self, x, labels, size_average=True):
#         batch_size = x.size(0)

#         v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))

#         left = F.relu(0.9 - v_c).view(batch_size, -1)
#         right = F.relu(v_c - 0.1).view(batch_size, -1)

#         loss = labels * left + 0.5 * (1.0 - labels) * right
#         loss = loss.sum(dim=1).mean()

#         return loss

#     def reconstruction_loss(self, data, reconstructions):
#         loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
#         return loss * 0.0005