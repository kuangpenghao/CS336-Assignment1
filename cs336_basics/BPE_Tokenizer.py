from collections import defaultdict
import regex

class BPE_Tokenizer:
    def __init__(self,vocab:dict[int,bytes],merge_list:list[tuple[bytes,bytes]],special_tokens=None):
        self.vocab=vocab
        self.vocab_reverse={v:k for k,v in vocab.items()}
        self.special_tokens=special_tokens
        if self.special_tokens is not None:
            if len(self.special_tokens)==2:
                if self.special_tokens[0]*2==self.special_tokens[1]:
                    temp=self.special_tokens[0]
                    self.special_tokens[0]=self.special_tokens[1]
                    self.special_tokens[1]=temp
        
        merge_dict={pair:i for i,pair in enumerate(merge_list)}
        self.merge_dict=merge_dict
    
    def from_files(vocab_path:str,merge_path:str,special_tokens=None):
        vocab={}
        #read vocab file,this file is a json file,the file format is like {"!": 0, "\"": 1, "#": 2, "$": 3, ……
        with open(vocab_path,"r",encoding="utf-8") as f:
            import json
            print("vocab_path:",vocab_path)
            vocab_json=json.load(f)
            for token_str,token_id in vocab_json.items():
                token_bytes=token_str.encode("utf-8")
                vocab[token_id]=token_bytes

        #read merge file,this file is like:a b\n c d\n……,one of this two parts can be ' '
        merge_list=[]
        with open(merge_path,"r",encoding="utf-8") as f:
            for line in f:
                parts=line.strip().split(" ")
                if len(parts)!=2:
                    continue
                first,second=parts
                merge_list.append((first.encode("utf-8"),second.encode("utf-8")))
        return BPE_Tokenizer(vocab,merge_list,special_tokens)

    def _chunk_text(self,text:str):
        if self.special_tokens is None or len(self.special_tokens)==0:
            yield text
            return
        pattern = "(" + "|".join([regex.escape(token) for token in self.special_tokens]) + ")"
        regex_chunk = regex.compile(pattern)
        chunks = regex_chunk.split(text)
        for chunk in chunks:
            if chunk:
                yield chunk

    def _encode_merge(self,token_list:list[bytes]):
        dict_idx=defaultdict(int)
        for i in range(len(token_list)-1):
            pair=(token_list[i],token_list[i+1])
            if pair in self.merge_dict:
                dict_idx[pair]=self.merge_dict[pair]
        if len(dict_idx)==0:
            return token_list,False
        
        min_number=998244353
        min_pair=None
        for pair,number in dict_idx.items():
            if number<min_number:
                min_number=number
                min_pair=pair

        new_token=min_pair[0]+min_pair[1]
        for i in range(len(token_list)-1):
            pair=(token_list[i],token_list[i+1])
            if pair==min_pair:
                token_list=token_list[:i]+[new_token]+token_list[i+2:]
                break
        return token_list,True

    def _process_char(self,char):
        return bytes([char])
    
    def _process_encode(self,ori_text:str):
        texts=self._chunk_text(ori_text)

        GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens is None:
            self.special_tokens=[]
        special_tokens_pattern = "|".join([regex.escape(token) for token in self.special_tokens])
        pattern_str = f"({special_tokens_pattern})|{GPT2_SPLIT_PATTERN}"
        pattern = regex.compile(pattern_str, flags=regex.UNICODE)

        encoded_text_list=[]

        for text in texts:
            text_split=pattern.finditer(text)
            for token in text_split:
                token_str=token.group()
                if token_str in self.special_tokens:
                    token_bytes=token_str.encode("utf-8")
                    encoded_text_list.append(self.vocab_reverse[token_bytes])
                    continue

                token_list=[self._process_char(b) for b in token_str.encode("utf-8")]

                can_merge=True
                while can_merge:
                    token_list,can_merge=self._encode_merge(token_list)

                encoded_list=[self.vocab_reverse[token] for token in token_list if token in self.vocab_reverse]
                encoded_list=[self.vocab[token] for token in encoded_list]

                encoded_text_list.extend([self.vocab_reverse[token] for token in encoded_list])

        return encoded_text_list
    '''
    def chunk_text_by_space(self,text:str,max_bytes:int):
        start=0
        while start<len(text):
            end=start+max_bytes
            if end>=len(text):
                yield text[start:]
                break
            else:
                space_pos=end
                while space_pos<len(text) and text[space_pos]!=" ":
                    space_pos+=1
                if space_pos==len(text):
                    yield text[start:]
                    break
                else:
                    yield text[start:space_pos]
                    start=space_pos
    '''       
    
    def encode(self,ori_text:str)->list[int]:
        encoded_text_list=[]
        encoded_text_list=self._process_encode(ori_text)
        return encoded_text_list


    def encode_iterable(self,iterable):
        for text in iterable:
            encoded_line=self.encode(text)
            for id in encoded_line:
                yield id

    def decode(self,ids:list[int])->str:
        bytes_list=[self.vocab[id] for id in ids]
        bytes_string=b''.join(bytes_list)
        decoded_string=bytes_string.decode("utf-8",errors="ignore")
        return decoded_string