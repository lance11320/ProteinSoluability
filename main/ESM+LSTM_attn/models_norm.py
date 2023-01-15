from transformers import  EsmForMaskedLM
import torch
import torch.nn as nn
from transformers.models import esm
class ESMclassficationModel(nn.Module):
    def __init__(self):
        super(ESMclassficationModel,self).__init__()
        config = esm.configuration_esm.EsmConfig.from_pretrained("facebook/esm2_t33_650M_UR50D", output_hidden_states=True)
        self.esm = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D",config=config)
        for param in self.esm.parameters():
            param.requires_grad = False
        
        hid_dim = 1280
        self.rnn = nn.GRU(input_size=hid_dim, hidden_size=hid_dim, num_layers=1, batch_first=True)
        self.attn = Attention()
        self.dense = nn.Linear(hid_dim, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids,attention_mask):
        output = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output.hidden_states[-1]
        out, gru_hidden = self.rnn(hidden_state)
        hidden_state, atten1 = self.attn(hidden_state,out,hidden_state,scale=1)
        self.attn_weight = atten1
        hidden_state = self.dropout(hidden_state)
        linear_output = self.dense(hidden_state)
        linear_output = torch.squeeze(torch.mean(linear_output,dim=1),dim=1)
        return  linear_output

class Attention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, scale, attn_mask=None):
        attention = torch.matmul(q, k.transpose(-1, -2))

        attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, 1e-9)
        attention = self.softmax(attention)
        attention = self.dropout(attention)

        context = torch.matmul(attention, v)
        return context, attention

