from cs336_basics.train_utils import *
from cs336_basics.BPE_Tokenizer import BPE_Tokenizer
from cs336_basics.Transformer_LM import Transformer_LM
import torch
import torch.nn as nn
import numpy as np
import tracemalloc
import argparse

class Memmap_Manager:
    def __init__(self,
                 chunk_size,
                 vocab_path,
                 merge_path,
                 special_tokens,
                 corpus_path):
        self.chunk_size=chunk_size
        self.vocab_path=vocab_path
        self.merge_path=merge_path
        self.special_tokens=special_tokens
        self.corpus_path=corpus_path
        
    def save_by_chunks(self,token_ids,buffer_len,chunk_num=0):
        fname=f"/home/kuangph/CS336-Assignment1/data/276M_chunks/encoded_tokens_chunk_{chunk_num}.dat"
        dtype=np.int32
        shape=(buffer_len,)
        memmap_arr = np.memmap(fname, dtype=dtype, mode="w+", shape=shape)
        memmap_arr[:] = token_ids[:]
        memmap_arr.flush()

    def save_as_memmap(self):
        tokenizer=BPE_Tokenizer.from_files(self.vocab_path,self.merge_path,self.special_tokens)
        buffer=[]
        chunk_num=0
        with open(self.corpus_path) as f:
            encoder=tokenizer.encode_iterable(f)
            for id in encoder:
                buffer.append(id)
                if len(buffer)>=self.chunk_size:
                    self.save_by_chunks(buffer,self.chunk_size,chunk_num)
                    chunk_num+=1
                    buffer=[]
            if len(buffer)>0:
                self.save_by_chunks(buffer,len(buffer),chunk_num)
                buffer=[]

    def load_by_range(self,start_idx,end_idx):
        chunk_size=self.chunk_size
        start_chunk=start_idx//chunk_size
        end_chunk=end_idx//chunk_size
        idx_in_start=start_idx%chunk_size
        idx_in_end=end_idx%chunk_size

        token_ids=[]
        for chunk in range(start_chunk,end_chunk+1):
            fname=f"/home/kuangph/CS336-Assignment1/data/276M_chunks/encoded_tokens_chunk_{chunk}.dat"
            dtype=np.int32
            memmap_arr=np.memmap(fname,dtype=dtype,mode="r")
            if start_chunk==end_chunk:
                token_ids.extend(memmap_arr[idx_in_start:idx_in_end])
            else:
                if chunk==start_chunk:
                    token_ids.extend(memmap_arr[idx_in_start:])
                elif chunk>start_chunk and chunk<end_chunk:
                    token_ids.extend(memmap_arr[:])
                else:
                    token_ids.extend(memmap_arr[:idx_in_end])
        return token_ids
                

def parse_bash_args():
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--d_model",type=int,default=512)
    parser.add_argument("--num_heads",type=int,default=8)
    parser.add_argument("--d_ff",type=int,default=2048)
    parser.add_argument("--vocab_size",type=int,default=32000)
    parser.add_argument("--num_layers",type=int,default=6)
    parser.add_argument("--max_seq_length",type=int,default=512)
    parser.add_argument("--seq_length",type=int,default=256)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--theta",type=int,default=100000)
    parser.add_argument("--device",type=str,default="cuda")

    parser.add_argument("--num_epochs",type=int,default=10)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--lr_min",type=float,default=1e-5)
    parser.add_argument("--warmup_ratio",type=float,default=0.1)
    parser.add_argument("--warmfix_ratio",type=float,default=0.9)
    
    parser.add_argument("--chunk_size",type=int,default=500000)
    parser.add_argument("--vocab_path",type=str,default="data/vocab_32000.txt")
    parser.add_argument("--merges_path",type=str,default="data/merges_32000.txt")
    parser.add_argument("--special_tokens", type=str, nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--corpus_path",type=str,default="data/21M.txt")

    parser.add_argument("--save_path",type=str)
    parser.add_argument("--log_interval",type=int)    
    parser.add_argument("--save_interval",type=int)
    
    args=parser.parse_args()
    return args


def training_loop(batch_getter,batch_size,seq_length,dataset_length,device,
                  transformer_lm,
                  loss_fn,optimizer,grad_clipper):
    
    tensor_input=batch_getter.get_batch(batch_size,seq_length,dataset_length,device=device)
    token_ids,labels=tensor_input

    output=transformer_lm(token_ids)

    loss=loss_fn.forward(output,labels)
    optimizer.zero_grad()
    loss.backward()
    grad_clipper.clip(transformer_lm.parameters())
    optimizer.step()

    return loss.item()


def train_manage():
    
    args=parse_bash_args()

    d_model=args.d_model
    num_heads=args.num_heads
    d_ff=args.d_ff
    vocab_size=args.vocab_size
    num_layers=args.num_layers
    max_seq_length=args.max_seq_length
    seq_length=args.seq_length
    batch_size=args.batch_size
    theta=args.theta
    dtype=torch.float32
    device=args.device

    num_epochs=args.num_epochs
    lr_max=args.lr
    lr_min=args.lr_min
    warmup_ratio=args.warmup_ratio
    warmfix_ratio=args.warmfix_ratio

    chunk_size=args.chunk_size
    vocab_path=args.vocab_path
    merge_path=args.merges_path
    special_tokens=args.special_tokens
    corpus_path=args.corpus_path

    if corpus_path=="/home/kuangph/CS336-Assignment1/data/5M.txt":
        dataset_length=1310528
    if corpus_path=="/home/kuangph/CS336-Assignment1/data/21M.txt":
        dataset_length=5621617
    if corpus_path=="/home/kuangph/CS336-Assignment1/data/276M.txt":
        dataset_length=66296750
    if corpus_path=="/home/kuangph/CS336-Assignment1/data/2G.txt":
        dataset_length=556539005

    save_path=args.save_path
    log_interval=args.log_interval
    save_interval=args.save_interval
    # weight decay,betas,eps for AdamW_Optimizer can be added later
    # max_norm for Gradient_Clipper can be added later

    token_positions=torch.arange(seq_length,device=device)

    transformer_lm=Transformer_LM(d_model=d_model,
                                  num_heads=num_heads,
                                  d_ff=d_ff,
                                  vocab_size=vocab_size,
                                  num_layers=num_layers,
                                  max_seq_length=max_seq_length,
                                  theta=theta,
                                  dtype=dtype,
                                  device=device,
                                  token_positions=token_positions)

    loss_fn=Cross_Entropy_Calculator()
    optimizer=AdamW_Optimizer(transformer_lm.parameters(),lr=0,weight_decay=0.01,betas=(0.9,0.95),eps=1e-8)
    lr_scheduler=Learning_Rate_Scheduler()
    grad_clipper=Gradient_Clipper(max_norm=1.0)
    memmap_manager=Memmap_Manager(chunk_size,vocab_path,merge_path,special_tokens,corpus_path)
    batch_getter=Batch_By_Memmap(memmap_manager)
    checkpoint_manager=Checkpoint_Manager()
    tokenizer=BPE_Tokenizer.from_files(vocab_path,merge_path,special_tokens)

    token_per_tensor=batch_size*seq_length
    total_iterations=dataset_length//token_per_tensor*num_epochs
    warmup_iterations=int(total_iterations*warmup_ratio)
    warmfix_iterations=int(total_iterations*warmfix_ratio)

    print(f"total_iterations:{total_iterations},warmup_iterations:{warmup_iterations},warmfix_iterations:{warmfix_iterations}")

    for ite in range(total_iterations):
        lr=lr_scheduler.get_lr(ite,lr_max,lr_min,warmup_iterations,warmfix_iterations)
        optimizer.lr=lr
        loss=training_loop(batch_getter,batch_size,seq_length,dataset_length,device,
                      transformer_lm,
                      loss_fn,optimizer,grad_clipper)
        if (ite+1)%20==0:
            print(f"iteration/total:{ite+1}/{total_iterations}, loss:{loss}, lr:{lr}")


if __name__=="__main__":
    train_manage()
'''
    chunk_size=500000
    vocab_path="data/vocab_32000.txt"
    merge_path="data/merges_32000.txt"
    special_tokens=["<|endoftext|>"]
    corpus_path="data/5M.txt"
    memmap_manager=Memmap_Manager(chunk_size,vocab_path,merge_path,special_tokens,corpus_path)

    memmap_manager.save_as_memmap()

    token_ids=memmap_manager.load_by_range(97500,132500)
    print(f"loaded {len(token_ids)} tokens")
    input("press enter to continue")
    tokenizer=BPE_Tokenizer.from_files(vocab_path,merge_path,special_tokens)
    decoded_text=tokenizer.decode(token_ids)
    print(f"decoded text:{decoded_text}")
'''