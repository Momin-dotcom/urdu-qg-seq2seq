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
        self.out=nn.Linear(hidden_dimensions+encoder_hidden_dimension,vocabulary_size)
    def forward_step(self,input_token,hidden_state,cell_state,encoder_outputs,mask):
        input_token=input_token.unsqueeze(1)
        emb_result=self.embedding(input_token)
        emb_result=self.dropout(emb_result)
        query=hidden_state[-1]
        context,weights=self.attention(query,encoder_outputs,mask)
        u_context=context.unsqueeze(1)
        in_lstm=torch.cat([emb_result,u_context],dim=2)
        output_lstm,(hidden_state,cell_state)=self.lstm(in_lstm,(hidden_state,cell_state))
        output_lstm=output_lstm.squeeze(1)
        combined_out=torch.cat([output_lstm,context],dim=1)
        logits=self.out(combined_out)
        return logits,hidden_state,cell_state,weights


    def forward(self,target,encoder_outputs,hidden_state,cell_state,mask,tf_ratio=0.5):
        batch_size,target_len=target.shape
        vocabulary_size=self.out.out_features
        outputs=torch.zeros(batch_size,target_len-1,vocabulary_size,device=target.device)
        input_token=target[:,0]
        for t in range(1,target_len):
            output,hidden_state,cell_state,weights=self.forward_step(input_token,hidden_state,cell_state,encoder_outputs,mask)
            outputs[:,t-1,:]=output
            teacher_force=torch.rand(1).item()<tf_ratio
            input_token=target[:,t] if teacher_force else output.argmax(dim=-1)
        return outputs,hidden_state,cell_state
    