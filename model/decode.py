import torch
import torch.nn.functional as F
import sentencepiece as spm
from encoder import Encoder
from decoder import Decoder

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer=spm.SentencePieceProcessor(model_file="tokenizer/ur_sp.model")

sos_id=tokenizer.bos_id()
eos_id=tokenizer.eos_id()
pad_id=tokenizer.pad_id()
max_len=30
vocab_size=8000

def combine_bidirectional(h_or_c,num_layers):
    h_or_c=h_or_c.view(num_layers,2,h_or_c.size(1),h_or_c.size(2))
    combined=h_or_c[:,0,:,:]+h_or_c[:,1,:,:]
    return combined

encoder=Encoder().to(device)
decoder=Decoder(vocabulary_size=vocab_size,embedding_dimension=256,hidden_dimensions=512,encoder_hidden_dimension=1024,numOfLayers=2,dropout=0.3,attention_dimension=512).to(device)

checkpoint=torch.load("best_checkpoint.pt",map_location=device)
encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

encoder.eval()
decoder.eval()

def prepare_input(sentence,answer):
    tagged=sentence.replace(answer,"<ans> "+answer+" </ans>")
    ids=tokenizer.encode(tagged,out_type=int)
    input_ids=torch.tensor([ids],device=device)
    mask=(input_ids!=pad_id)
    return input_ids,mask

def greedy_decode(sentence,answer):
    input_ids,mask=prepare_input(sentence,answer)
    with torch.no_grad():
        encoder_outputs,(h_n,c_n)=encoder(input_ids)
        hidden=combine_bidirectional(h_n,2)
        cell=combine_bidirectional(c_n,2)
        input_token=torch.tensor([sos_id],device=device)
        generated_ids=[]
        for step in range(max_len):
            logits,hidden,cell,weights=decoder.forward_step(input_token,hidden,cell,encoder_outputs,mask)
            next_id=logits.argmax(dim=-1)
            if next_id.item()==eos_id:
                break
            generated_ids.append(next_id.item())
            input_token=next_id
    return tokenizer.decode(generated_ids)

def beam_search_decode(sentence,answer,beam_width=3,length_alpha=0.7):
    input_ids,mask=prepare_input(sentence,answer)
    with torch.no_grad():
        encoder_outputs,(h_n,c_n)=encoder(input_ids)
        hidden=combine_bidirectional(h_n,2)
        cell=combine_bidirectional(c_n,2)
        beams=[([sos_id],0.0,hidden,cell,False)]
        for step in range(max_len):
            all_candidates=[]
            for seq,score,h,c,finished in beams:
                if finished:
                    all_candidates.append((seq,score,h,c,finished))
                    continue
                input_token=torch.tensor([seq[-1]],device=device)
                logits,new_h,new_c,weights=decoder.forward_step(input_token,h,c,encoder_outputs,mask)
                log_probs=F.log_softmax(logits,dim=-1)
                top_log_probs,top_ids=log_probs.topk(beam_width,dim=-1)
                for i in range(beam_width):
                    token_id=top_ids[0,i].item()
                    token_log_prob=top_log_probs[0,i].item()
                    new_seq=seq+[token_id]
                    new_score=score+token_log_prob
                    new_finished=(token_id==eos_id)
                    all_candidates.append((new_seq,new_score,new_h,new_c,new_finished))
            def normalized_score(candidate):
                seq,score,h,c,finished=candidate
                return score/(len(seq)**length_alpha)
            all_candidates.sort(key=normalized_score,reverse=True)
            beams=all_candidates[:beam_width]
            if all(beam[4] for beam in beams):
                break
    best_seq=beams[0][0]
    output_ids=[tok_id for tok_id in best_seq[1:] if tok_id!=eos_id]
    return tokenizer.decode(output_ids)

def generate(sentence,answer,method="greedy"):
    if method=="greedy":
        return greedy_decode(sentence,answer)
    elif method=="beam":
        return beam_search_decode(sentence,answer)
    else:
        raise ValueError("method must be greedy or beam")

