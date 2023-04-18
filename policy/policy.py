import torch
import torch.nn as nn
import numpy as np

class DynamicEncoder(nn.Module):
    def __init__(self,dynamic_dimension,feature_dimension):
        super().__init__()
        self.feature_dimension = feature_dimension
        self.encoder = nn.RNN(dynamic_dimension,feature_dimension,batch_first=True)
    
    def forward(self,x,idx_array):
        assert len(idx_array) == 2, "The dimension of index_array must be 2"
        h0 = torch.zeros(1,x.size(0),self.feature_dimension)
        hiddens, _ = self.encoder(x,h0)
        return hiddens[idx_array[:,0],idx_array[:,0],:]


def encoder(input_dimension,output_dimension):
    l1 = nn.Linear(input_dimension,output_dimension)
    l2 = nn.ReLU()
    model = nn.Sequential(l1, l2)
    return model


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
                 seed=0):
        super().__init__()
        self.self_encoder = encoder(self_dimension,feature_dimension)
        self.static_encoder = encoder(static_dimension,feature_dimension)
        self.cooperative = cooperative
        if self.cooperative:
            self.dynamic_encoder = DynamicEncoder(dynamic_dimension,feature_dimension)
        else:
            self.dynamic_encoder = None

        self.gcn = GCN(feature_dimension,gcn_hidden_dimension,feature_2_dimension)
        self.iqn = IQN(feature_2_dimension,iqn_hidden_dimension,action_size,seed)
        
    def forward(self,x,num_tau=8,cvar=1.0):
        if self.cooperative:
            assert len(x) == 3, "The number of state types of a cooperative agent must be 4!"
            x_1, x_2, x_3, idx_array = x

            x_1_feature = self.self_encoder(x_1)
            x_2_feature = self.static_encoder(x_2)
            x_3_feature = self.dynamic_encoder(x_3,idx_array)

            X_feature = torch.cat((x_1_feature,x_2_feature,x_3_feature),dim=0).unsqueeze(0)
        else:
            assert len(x) == 2, "The number of state types of a non-cooperative agent must be 2!"
            x_1, x_2 = x

            x_1_feature = self.self_encoder(x_1)
            x_2_feature = self.static_encoder(x_2)

            X_feature = torch.cat((x_1_feature,x_2_feature),dim=0).unsqueeze(0)

        self_interation_feature = self.gcn(X_feature)
        quantiles, taus = self.iqn(self_interation_feature,num_tau,cvar)
        return quantiles, taus


class GCN(nn.Module):
    def __init__(self,feature_dimension,hidden_dimension,output_dimension) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(feature_dimension, hidden_dimension))
        self.w2 = nn.Parameter(torch.randn(hidden_dimension,output_dimension))
        self.wr = nn.Parameter(torch.randn(feature_dimension,feature_dimension))

    def forward(self,x):
        # compute the realtion matrix from the feature matrix
        R = torch.matmul(torch.matmul(x,self.wr),x.permute(0,2,1))
        R = nn.Softmax(R,dim=2)

        # compute new features that encodes interactions
        x = torch.matmul(torch.matmul(R,x),self.w1)+x
        x = nn.ReLU(x)
        x = torch.matmul(torch.matmul(R,x),self.w2)+x 
        x = nn.ReLU(x)

        # return the self feature 
        self_x = x[:,0,:]
        return self_x


class IQN(nn.Module):
    def __init__(self, feature_dimension, hidden_dimension, action_size, seed=0):
        super().__init__()
        self.seed_id = seed
        self.seed = torch.manual_seed(seed)
        self.K = 32
        self.n = 64

        self.feature_dimension = feature_dimension
        self.action_size = action_size

        # quantile encoder
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n)]).view(1, 1, self.n)
        self.cos_embedding = nn.Linear(self.n,feature_dimension)

        # hidden layers
        self.hidden_layer = nn.Linear(feature_dimension, hidden_dimension)
        self.hidden_layer_2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.output_layer = nn.Linear(hidden_dimension, action_size)

    def calc_cos(self, batch_size, n_tau=8, cvar=1.0):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1) # (batch_size, n_tau, 1) for broadcast

        # distorted quantile sampling
        taus = taus * cvar

        cos = torch.cos(taus * self.pis)
        assert cos.shape == (batch_size, n_tau, self.n), "cos shape is incorrect"
        return cos, taus


    def forward(self, x, num_tau=8, cvar=1.0):
        batch_size = x.shape[0]

        # encode quantiles as features
        cos, taus = self.calc_cos(batch_size, num_tau, cvar) # cos shape (batch_size, num_tau, 64)
        cos = cos.view(batch_size * num_tau, self.n)
        cos_features = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.feature_dimension)

        # x has shape (batch, feature_dimension) for multiplication –> reshape to (batch, 1, feature_dimension)
        x = (x.unsqueeze(1) * cos_features).view(batch_size * num_tau, self.feature_dimension)
        
        x = torch.relu(self.hidden_layer(x))
        x = torch.relu(self.hidden_layer_2(x))
        out = self.output_layer(x)
        return out.view(batch_size, num_tau, self.action_size), taus

    def get_qvals(self, inputs, cvar):
        quantiles, _ = self.forward(inputs=inputs, num_tau=self.K, cvar=cvar)
        qvals = quantiles.mean(dim=1)
        return qvals

    def get_constructor_parameters(self):       
        return dict(state_size=self.state_size, \
                    action_size=self.action_size, \
                    seed=self.seed_id)

    def save(self,directory):
        # torch.save({"state_dict": self.state_dict(), "constructor_params": self.get_constructor_parameters()}, path)
        # torch.save(self.state_dict(), path)

        # save network parameters
        torch.save(self.state_dict(),os.path.join(directory,"network_params.pth"))
        
        # save constructor parameters
        with open(os.path.join(directory,"constructor_params.json"),mode="w") as constructor_f:
            json.dump(self.get_constructor_parameters(),constructor_f)

    @classmethod
    def load(cls,directory,device="cpu"):
        
        # load network parameters
        model_params = torch.load(os.path.join(directory,"network_params.pth"),
                                  map_location=device)

        # load constructor parameters
        with open(os.path.join(directory,"constructor_params.json"), mode="r") as constructor_f:
            constructor_params = json.load(constructor_f)
            constructor_params["device"] = device

        model = cls(**constructor_params)
        model.load_state_dict(model_params)
        model.to(device)

        return model