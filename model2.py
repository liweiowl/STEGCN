import torch
import torchsnooper
from config import cf


# define a basic graph convolution operation
## use the super can succedd the init function in parent class
@torchsnooper.snoop()
class GraphConvolution(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphConvolution, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, adj, features):
        out = torch.mm(adj, features)  # graph = f(A_hat*W)
        out = self.linear(out)
        return out


# define a GCN network
@torchsnooper.snoop()
class GCN(torch.nn.Module):
    def __init__(self, input_size=207, hidden_size=100, output_size=207):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_size, hidden_size)
        self.gcn2 = GraphConvolution(hidden_size, output_size)

    def forward(self, adj, features):
        print(f"adj shape {adj.shape}, features shape {features.shape}")
        out = torch.nn.functional.relu(self.gcn1(adj, features))
        # out = self.gcn2(adj, out)
        return out

# class GCN(torch.nn.Module):
#     def __init__(self,
#                  input_size=207,
#                  hidden_size=150):
#         super(GCN, self).__init__()
#         self.output_dim = hidden_size
#         self.gcn1 = GraphConvolution(input_size, hidden_size)
#         self.gcn2 = GraphConvolution(hidden_size, hidden_size)
#         self.gcn3 = GraphConvolution(hidden_size, self.output_dim)
#
#     def forward(self, adj, features):
#         print(f"feature shape is {features.shape}")
#         out = torch.nn.ReLU(self.gcn1(adj, features))
#         out = torch.nn.ReLU(self.gcn2(adj, out))
#         out = torch.nn.ReLU(self.gcn3(adj, out))
#         return out


class TGCN(torch.nn.Module):
    def __init__(self, input_size,
                 output_size,
                 hidden_dim,
                 num_nodes=207,
                 # seq_len=50,
                 dim=1
                 ):
        super(TGCN, self).__init__()
        self.gcn = GCN(input_size, output_size)
        self.lstm = torch.nn.LSTM(output_size, hidden_dim)
        self.linear = torch.nn.Linear(dim * num_nodes * num_nodes, dim * num_nodes * num_nodes)

    def forward(self, adj, features):
        out = self.gcn(adj, features)
        out = self.lstm(out)
        out = torch.nn.Flatteern()
        out = self.linear(out)
        return out

@torchsnooper.snoop()
class NormalLSTM(torch.nn.Module):
    def __init__(self,
                 input_size=207,
                 hidden_dim=200,
                 num_layers=4):
        super(NormalLSTM, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # self.lstm = torch.nn.LSTM(input_size, hidden_dim, num_layers,batch_first=True)
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers,
                                  batch_first=True)
        self.fc = torch.nn.Linear(in_features=self.hidden_dim * cf.seq_len, out_features=input_size * cf.pred_len)

    def forward(self, x):
        h0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim)).to(cf.device)
        c0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim)).to(cf.device)
        out, _ = self.lstm(x, (h0, c0))
        # print(f"out shape is {out.shape}")

        out = self.fc(out.reshape(-1, self.hidden_dim * cf.seq_len)).reshape(-1, cf.pred_len, self.input_dim)
        # out, _ = self.lstm(x)
        # out shape is (batch_size, pred_len, num_node)
        return out

@torchsnooper.snoop()
class MYTGCN(torch.nn.Module):
    def __init__(self, num_node=cf.num_nodes):
        super(MYTGCN, self).__init__()
        self.gcn_layers = 2
        self.num_node = num_node
        self.gcn = GCN(input_size=207, hidden_size=150)
        self.lstm = NormalLSTM(input_size=207, hidden_dim=200, num_layers=4)

    def forward(self, adj, features):
        lstm_input = []
        for i in range(cf.seq_len):
            print(features.shape)
            print(features[:,i,:].shape)
            print(adj.shape)
            # adj = adj.repeat(cf.batch_size, 1,1)
            # print(f"adj repeat shape:{adj.shape}")
            # feature = torch.unsqueeze(features[:,i,:], dim=-1)
            # print(f"feature shape :{feature.shape}")
            # out1 = self.gcn(adj, features)
            out1 = self.gcn(adj, features[:,i,:].transpose(1,0))
            # concate = torch.cat(out1)
            print(f"out1 shape is {out1.shape}")
            lstm_input.append(out1)
        lstm_input = torch.tensor(lstm_input).to(cf.device)
        print(lstm_input.shape)
        lstm_output = self.lstm(lstm_input)
        # encoder model and decoder part
        out = lstm_output.view(cf.pred_len, self.num_node)
        return out

@torchsnooper.snoop()
class STEGCN(torch.nn.Module):
    def __init__(self,
                 embedding_dim=cf.embedding_dim,
                 num_node=cf.num_nodes,
                 encoder_input=cf.seq_len * cf.lstm_outdim,
                 encoder_output=cf.encoder_output_dim,
                 decoder_output=cf.num_nodes * cf.pred_len,
                 adj=cf.adjacent_matrix):
        super(STEGCN, self).__init__()
        self.embedding_dim = embedding_dim
        self.gcn_layers = 2
        self.num_node = num_node
        self.encoder_output = encoder_output
        self.decoder_output = decoder_output
        self.encoder = torch.nn.Linear(encoder_input, self.encoder_output)
        self.decoder = torch.nn.Linear(self.encoder_output + cf.embedding_dim, self.decoder_output)
        self.gcn = GCN()
        self.lstm = NormalLSTM_2layer()

    def forward(self, adj, features, embedding_inputs):
        # get x representation
        lstm_input = []
        for i in range(cf.seq_len):
            out1 = self.gcn(adj, features[i])
            concate = torch.cat(out1, embedding_inputs[i])
            lstm_input.append(concate)
        lstm_input = torch.tensor(lstm_input)
        lstm_output = self.lstm(lstm_input)
        mask_lstm_input = []
        # get mask y representation
        for j in range(cf.pred_len):
            mask = torch.tensor([]).byte()
            out2 = torch.masked_select(torch.cat(self.gcn(adj, features[i]), embedding_inputs[j]), mask=mask)
            mask_lstm_input.append(out2)
        mask_lstm_input = torch.tensor(mask_lstm_input)
        mask_lstm_output = self.lstm(mask_lstm_input)
        # encoder model and decoder part
        encoder_out = self.encoder(torch.cat(lstm_output, mask_lstm_output), self.encoder_output)
        decode_out = self.decoder(torch.cat(encoder_out, embedding_inputs))
        out = decode_out.view(cf.pred_len, self.num_node)
        return out



if __name__ == "__main__":
    print('begin to test')
    # testmodel = STEGCN()
    testmodel1 = MYTGCN()

    print('over')








######################################### history versions backup ########################################

class NormalLSTM_2layer(torch.nn.Module):
    def __init__(self,
                 input_size=207,
                 hidden_dim=200,
                 num_layers=4):
        super(NormalLSTM_2layer, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # self.lstm = torch.nn.LSTM(input_size, hidden_dim, num_layers,batch_first=True)
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers,
                                  batch_first=True)
        self.lstm2 = torch.nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers,
                                   batch_first=True)
        self.fc = torch.nn.Linear(in_features=self.hidden_dim * cf.seq_len, out_features=input_size * cf.pred_len)

    def forward(self, x):
        h0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim)).to(cf.device)
        c0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim)).to(cf.device)
        h0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim)).to(cf.device)
        c0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim)).to(cf.device)
        out1, (h1, c1) = self.lstm(x, (h0, c0))
        # print(f"out shape is {out.shape}")
        out2, _ = self.lstm2(out1, (h1, c1))
        out = self.fc(out2.reshape(-1, self.hidden_dim * cf.seq_len)).reshape(-1, cf.pred_len, self.input_dim)

        # out, _ = self.lstm(x)
        # out shape is (batch_size, pred_len, num_node)
        return out

class GCN_backup(torch.nn.Module):
    def __init__(self, input_size=207, hidden_size=100, output_size=207):
        super(GCN_backup, self).__init__()
        self.gcn1 = GraphConvolution(input_size, hidden_size)
        self.gcn2 = GraphConvolution(hidden_size, output_size)

    def forward(self, adj, features):
        out = torch.nn.functional.relu(self.gcn1(adj, features))
        out = self.gcn2(adj, out)
        return out




