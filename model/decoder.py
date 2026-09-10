import torch
import torch.nn as nn
from attention import BahdanauAttention
class Decoder(nn.Module):
    def __init__(self,vocabulary_size,embedding_dimension,hidden_dimensions,encoder_hidden_dimension,numOfLayers,dropout,attention_dimension):
        super().__init__()
        self.hidden_dimensions=hidden_dimensions
        self.encoder_hidden_dimension=encoder_hidden_dimension
        self.numOfLayers=numOfLayers
        self.embedding=nn.Embedding(vocabulary_size,embedding_dimension)
        self.attention=BahdanauAttention(hidden_dimensions,encoder_hidden_dimension,attention_dimension)
        self.lstm=nn.LSTM(embedding_dimension+encoder_hidden_dimension,hidden_dimensions,numOfLayers,batch_first=True,dropout=dropout if numOfLayers>1 else 0)
        self.dropout=nn.Dropout(dropout)
        self.out_projection=nn.Linear(hidden_dimensions+encoder_hidden_dimension,vocabulary_size)
    def forward_step(self,input_token,hidden_state,cell_state,encoder_outputs):
        input_token=input_token.unsqueeze(1)
        embedded=self.embedding(input_token)
        embedded=self.dropout(embedded)
        query=hidden_state[-1]
        context,attn_weights=self.attention(query,encoder_outputs)
        context_unsq=context.unsqueeze(1)
        lstm_input=torch.cat([embedded,context_unsq],dim=2)
        lstm_out,(hidden_state,cell_state)=self.lstm(lstm_input,(hidden_state,cell_state))
        lstm_out=lstm_out.squeeze(1)
        combined=torch.cat([lstm_out,context],dim=1)
        output=self.out_projection(combined)
        return output,hidden_state,cell_state,attn_weights
    def forward(self,target,encoder_outputs,hidden_state,cell_state,teacher_forcing_ratio=0.5):
        batch_size,target_len=target.shape
        vocabulary_size=self.out_projection.out_features
        outputs=torch.zeros(batch_size,target_len-1,vocabulary_size,device=target.device)
        input_token=target[:,0]
        for t in range(1,target_len):
            output,hidden_state,cell_state,attn_weights=self.forward_step(input_token,hidden_state,cell_state,encoder_outputs)
            outputs[:,t-1,:]=output
            teacher_force=torch.rand(1).item()<teacher_forcing_ratio
            input_token=target[:,t] if teacher_force else output.argmax(dim=-1)
        return outputs,hidden_state,cell_state