import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device=None):
        super(RoPE,self).__init__()
        self.theta=theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len
        self.device=device

        d_half=d_k//2
        #create a tensor:cos_values(1,max_seq_len,d_half)
        positions=torch.arange(max_seq_len, device=device).unsqueeze(1)
        dims=torch.arange(d_half,device=device).unsqueeze(0)
        angles=positions/(theta**(2*dims/d_k))

        cos_values=torch.cos(angles).unsqueeze(0)
        self.register_buffer("cos_values",cos_values)#(1,max_seq_len,d_half)
        sin_values=torch.sin(angles).unsqueeze(0)
        self.register_buffer("sin_values",sin_values)#(1,max_seq_len,d_half)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        #x:(bsz,seq_len,d_k)
        bsz=x.shape[0]
        seq_len=x.shape[-2]
        #print(f"bsz={bsz},seq_len={seq_len},d_k={self.d_k}")
        d_k=x.shape[-1]
        x_splited=x.reshape(*x.shape[:-1],d_k//2,2)
        #print(f"shape of x_splited:{x_splited.shape}")

        #odd transform:(1,seq_len,d_k/2,2),(cos,-sin)
        cos_chunk=self.cos_values[:,token_positions,:]
        sin_chunk=self.sin_values[:,token_positions,:]
        #print(f"shape of cos_chunk:{cos_chunk.shape},shape of sin_chunk:{sin_chunk.shape}")
        odd_transform=torch.stack([cos_chunk,-sin_chunk],dim=-1)
        even_transform=torch.stack([sin_chunk,cos_chunk],dim=-1)
        #print(f"shape of odd_transform:{odd_transform.shape}")

        x_rotated_odd=torch.sum(x_splited*odd_transform,dim=-1)#(bsz,seq_len,d_k//2)
        x_rotated_even=torch.sum(x_splited*even_transform,dim=-1)#(bsz,seq_len,d_k//2)
        stacked_x=torch.stack([x_rotated_odd,x_rotated_even],dim=-1)
        x_rotated=stacked_x.reshape(*stacked_x.shape[:-2],d_k) 
        
        return x_rotated