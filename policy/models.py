import torch
import torch.nn as nn
import numpy as np
import os
import json
from torch.nn.functional import softmax,relu

class PolicyNetwork(nn.Module):
    def __init__(self,
                 self_dimension,
                 static_dimension,
                 feature_dimension,
                 gcn_hidden_dimension,
                 feature_2_dimension,
                 iqn_hidden_dimension,
                 action_size,
                 dynamic_dimension=None,
                 cooperative=False,
                 device='cpu',
                 seed=0):
        super().__init__()

        self.self_dimension = self_dimension
        self.static_dimension = static_dimension
        self.feature_dimension = feature_dimension
        self.gcn_hidden_dimension = gcn_hidden_dimension
        self.feature_2_dimension = feature_2_dimension
        self.iqn_hidden_dimension = iqn_hidden_dimension
        self.action_size = action_size
        self.dynamic_dimension = dynamic_dimension,
        self.cooperative = cooperative
        self.device = device
        self.seed = seed

        self.self_encoder = encoder(self_dimension,feature_dimension)
        self.static_encoder = encoder(static_dimension,feature_dimension)
        if self.cooperative:
            self.dynamic_encoder = DynamicEncoder(dynamic_dimension,feature_dimension,device)
        else:
            self.dynamic_encoder = None

        self.gcn = GCN(feature_dimension,gcn_hidden_dimension,feature_2_dimension)
        self.iqn = IQN(feature_2_dimension,iqn_hidden_dimension,action_size,device,seed)
        
    def forward(self,x,num_tau=8,cvar=1.0):
        x_2 = None
        x_3 = None

        if self.cooperative:
            assert len(x) == 6, "The number of state types of a cooperative agent must be 6!"
            x_1, x_2, x_2_num, x_3, x_3_num, x_3_idx = x
        else:
            assert len(x) == 3, "The number of state types of a non-cooperative agent must be 3!"
            x_1, x_2, x_2_num = x

        x_1_feature_batch = self.self_encoder(x_1).unsqueeze(1)
        batch_size = x_1_feature_batch.shape[0]
        
        max_size = 0
        if x_2 is not None:
            x_2_feature = self.static_encoder(x_2)
            max_size += torch.diff(x_2_num).max()
        if x_3 is not None:
            x_3_feature = self.dynamic_encoder(x_3,x_3_idx)
            max_size += torch.diff(x_3_num).max()
        
        # reorganize dynamic and static features into the graph batch
        x_2_3_feature_batch = None
        if max_size > 0:
            x_2_3_feature_batch = torch.zeros((batch_size,max_size,self.feature_dimension)).to(self.device)

            for i in range(batch_size):
                curr_idx = 0
                if x_2 is not None:
                    # fill in static features
                    x_2_idx = x_2_num[i+1]-x_2_num[i]
                    x_2_3_feature_batch[i,:curr_idx+x_2_idx,:] = x_2_feature[x_2_num[i]:x_2_num[i+1],:]
                    curr_idx += x_2_idx
                if x_3 is not None:
                    # fill in dynamic features
                    x_3_idx = x_3_num[i+1]-x_3_num[i]
                    x_2_3_feature_batch[i,curr_idx:curr_idx+x_3_idx,:] = x_3_feature[x_3_num[i]:x_3_num[i+1],:]
                    curr_idx += x_3_idx

        X_feature = x_1_feature_batch
        if x_2_3_feature_batch is not None:
            X_feature = torch.cat((X_feature,x_2_3_feature_batch),dim=1)

        # TODO: enable batch for gcn and iqn
        R_matrix, self_interation_feature = self.gcn(X_feature)
        quantiles, taus = self.iqn(self_interation_feature,num_tau,cvar)
        return quantiles, taus, R_matrix
    
    def get_constructor_parameters(self):       
        return dict(self_dimension = self.self_dimension,
                    static_dimension = self.static_dimension,
                    feature_dimension = self.feature_dimension,
                    gcn_hidden_dimension = self.gcn_hidden_dimension,
                    feature_2_dimension = self.feature_2_dimension,
                    iqn_hidden_dimension = self.iqn_hidden_dimension,
                    action_size = self.action_size,
                    dynamic_dimension = self.dynamic_dimension,
                    cooperative = self.cooperative,
                    seed = self.seed)

    def save(self,directory):
        agent = "cooperative_model" if self.cooperative else "non_cooperative_model"
        
        # save network parameters
        torch.save(self.state_dict(),os.path.join(directory,f"{agent}_network_params.pth"))
        
        # save constructor parameters
        with open(os.path.join(directory,f"{agent}_constructor_params.json"),mode="w") as constructor_f:
            json.dump(self.get_constructor_parameters(),constructor_f)

    @classmethod
    def load(cls,directory,agent_type,device="cpu"):
        # load network parameters
        model_params = torch.load(os.path.join(directory,f"{agent_type}_model_network_params.pth"),
                                  map_location=device)

        # load constructor parameters
        with open(os.path.join(directory,f"{agent_type}_model_constructor_params.json"), mode="r") as constructor_f:
            constructor_params = json.load(constructor_f)
            constructor_params["device"] = device

        model = cls(**constructor_params)
        model.load_state_dict(model_params)
        model.to(device)

        return model


class DynamicEncoder(nn.Module):
    def __init__(self,dynamic_dimension,feature_dimension,device):
        super().__init__()
        self.feature_dimension = feature_dimension
        self.encoder = nn.RNN(dynamic_dimension,feature_dimension,batch_first=True)
        self.device = device
    
    def forward(self,x,idx_array):
        h0 = torch.zeros(1,x.size(0),self.feature_dimension).to(self.device)
        hiddens, _ = self.encoder(x,h0)
        return hiddens[idx_array[:,0],idx_array[:,1],:]


def encoder(input_dimension,output_dimension):
    l1 = nn.Linear(input_dimension,output_dimension)
    l2 = nn.ReLU()
    model = nn.Sequential(l1, l2)
    return model


class GCN(nn.Module):
    def __init__(self,feature_dimension,hidden_dimension,output_dimension) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(feature_dimension, hidden_dimension))
        self.w2 = nn.Parameter(torch.randn(hidden_dimension,output_dimension))
        self.wr = nn.Parameter(torch.randn(feature_dimension,feature_dimension))

    def forward(self,x):
        # compute the realtion matrix from the feature matrix
        R = torch.matmul(torch.matmul(x,self.wr),x.permute(0,2,1))
        R = softmax(R,dim=2)

        # compute new features that encodes interactions
        x = relu(torch.matmul(torch.matmul(R,x),self.w1))
        x = relu(torch.matmul(torch.matmul(R,x),self.w2))

        # return the self feature 
        self_x = x[:,0,:]
        return R, self_x


class IQN(nn.Module):
    def __init__(self, feature_dimension, hidden_dimension, action_size, device, seed=0):
        super().__init__()
        self.seed_id = seed
        self.seed = torch.manual_seed(seed)
        self.K = 32
        self.n = 64 # number of cosine features

        self.feature_dimension = feature_dimension
        self.action_size = action_size
        self.device = device

        # quantile encoder
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n)]).view(1,1,self.n).to(device)
        self.cos_embedding = nn.Linear(self.n,feature_dimension)

        # hidden layers
        self.hidden_layer = nn.Linear(feature_dimension, hidden_dimension)
        self.hidden_layer_2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.output_layer = nn.Linear(hidden_dimension, action_size)

    def calc_cos(self, batch_size, num_tau=8, cvar=1.0):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size,num_tau).to(self.device).unsqueeze(-1)

        # distorted quantile sampling
        taus = taus * cvar

        cos = torch.cos(taus * self.pis)
        assert cos.shape == (batch_size, num_tau, self.n), "cos shape is incorrect"
        return cos, taus


    def forward(self, x, num_tau=8, cvar=1.0):
        batch_size = x.shape[0]
        
        # encode quantiles as features
        cos, taus = self.calc_cos(batch_size, num_tau, cvar)
        cos = cos.view(batch_size*num_tau, self.n)
        cos_features = relu(self.cos_embedding(cos)).view(batch_size,num_tau,self.feature_dimension)

        # pairwise product of the input feature and cosine features
        x = (x.unsqueeze(1) * cos_features).view(batch_size*num_tau,self.feature_dimension)
        
        x = relu(self.hidden_layer(x))
        x = relu(self.hidden_layer_2(x))
        quantiles = self.output_layer(x)
        return quantiles.view(batch_size,num_tau,self.action_size), taus