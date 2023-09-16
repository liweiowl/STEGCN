import torch
from config import cf
from model import GCN, NormalLSTM_2layer

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
