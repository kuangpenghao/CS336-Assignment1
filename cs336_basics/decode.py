from cs336_basics.BPE_Tokenizer import BPE_Tokenizer
from cs336_basics.Transformer_LM import Transformer_LM
from cs336_basics.train_utils import *
from cs336_basics.Transformer_utils import *

class Config:
    def __init__(self):
        self.corpus_path="/home/kuangph/CS336-Assignment1/data/validation/decode_1.txt"
        self.checkpoint_path="/home/kuangph/CS336-Assignment1/outputs/276M_checkpoints"

        self.vocab_path="/home/kuangph/CS336-Assignment1/data/vocab_32000.txt"
        self.merge_path="/home/kuangph/CS336-Assignment1/data/merges_32000.txt"
        self.special_tokens=["<|endoftext|>"]

        self.d_model=512
        self.num_heads=8
        self.d_ff=1344
        self.vocab_size=32000
        self.num_layers=6
        self.max_seq_length=512
        self.batch_size=32
        self.theta=100000
        self.device="cuda"
        self.dtype=torch.float32

if __name__=="__main__":
    config=Config()

    tokenizer=BPE_Tokenizer.from_files(config.vocab_path,config.merge_path,config.special_tokens)
    with open(config.corpus_path) as f:
        encoded_list=tokenizer.encode(f.read())
    
    decoded_list=tokenizer.decode(encoded_list)
    print(f"decoded text:\n{decoded_list}\n\ngenerated text:")

    token_positions=torch.arange(len(encoded_list),device=config.device)
        
    transformer_lm=Transformer_LM(d_model=config.d_model,
                                  num_heads=config.num_heads,
                                  d_ff=config.d_ff,
                                  vocab_size=config.vocab_size,
                                  num_layers=config.num_layers,
                                  max_seq_length=config.max_seq_length,
                                  theta=config.theta,
                                  dtype=config.dtype,
                                  device=config.device)
                                  #token_positions=token_positions)
    
    checkpoint_manager=Checkpoint_Manager()
    checkpoint_manager.load(config.checkpoint_path,transformer_lm)

    ite=0
    softmax_activation=Softmax_Activation(-1)
    while ite<config.max_seq_length:
        if len(encoded_list)<config.max_seq_length:
            input_ids=encoded_list
        else:
            input_ids=encoded_list[-config.max_seq_length:]
        input_tensor=torch.tensor([input_ids],dtype=torch.long,device=config.device)

        #print(f"shape of input tensor:{input_tensor.shape}")

        with torch.no_grad():
            # create token positions,which is the list of indexes of input tensors in encoded list
            token_positions=torch.arange(ite-len(input_ids)+1,ite+1,device=config.device)

            output_scores=transformer_lm(input_tensor,token_positions)
            last_token_scores=output_scores[0,-1,:]
            last_token_weights=softmax_activation(last_token_scores)

            # let all weights below 0.3 to be zero
            #last_token_weights[last_token_weights<0.01]=0
            #last_token_weights=last_token_weights/last_token_weights.sum()

            # sample from the distribution
            #sampled_id=torch.multinomial(last_token_weights,num_samples=1).item()
            sampled_id=torch.argmax(last_token_weights).item()
            encoded_list.append(sampled_id)
            ite+=1

            print(f"{tokenizer.decode([sampled_id])}",end="")

            if sampled_id in tokenizer.special_tokens:
                break
    print("\n")