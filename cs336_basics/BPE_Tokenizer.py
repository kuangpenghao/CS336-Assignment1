from collections import defaultdict
import regex
#from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path

VOCAB_PATH="/home/kuangph/CS336-Assignment1/tests/fixtures/gpt2_vocab.json"
MERGES_PATH="/home/kuangph/CS336-Assignment1/tests/fixtures/gpt2_merges.txt"

class BPE_Tokenizer:
    # 此处self.merge_dict后续需要修改，因为不需要考虑merge_list为none
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

        #print(f"length of merge_dict:{len(self.merge_dict)}")
        #print(f"merge_dict's 30 items:{list(self.merge_dict)[:30]}")
    
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

        #print(f"vocab's 30 items:{list(vocab.items())[:30]}")
        #print(f"merge_list:{merge_list[:30]}")
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
        #input("press enter to continue...")
        #print(f"token_list:{token_list}")
        dict_idx=defaultdict(int)
        for i in range(len(token_list)-1):
            pair=(token_list[i],token_list[i+1])
            #print(f"    pair:{pair}")
            if pair in self.merge_dict:
                dict_idx[pair]=self.merge_dict[pair]
            #else:
                #print(f"    pair:{pair} not in merge_dict")
        #print(f"dict_idx:{dict_idx}")
        if len(dict_idx)==0:
            return token_list,False
        
        min_number=998244353
        min_pair=None
        for pair,number in dict_idx.items():
            #print(f"pair:{pair},number:{number}")
            if number<min_number:
                min_number=number
                min_pair=pair

        #print(f"min_pair:{min_pair},min_number:{min_number}")

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
            #print(f"text:{text}")
            text_split=pattern.finditer(text)
            for token in text_split:
                token_str=token.group()
                if token_str in self.special_tokens:
                    #print(f"special token:{token_str}")
                    token_bytes=token_str.encode("utf-8")
                    encoded_text_list.append(self.vocab_reverse[token_bytes])
                    continue

                token_list=[self._process_char(b) for b in token_str.encode("utf-8")]
                #print(f"TOKEN LIST BEFORE MERGE1:{token_list}")

                can_merge=True
                while can_merge:
                    token_list,can_merge=self._encode_merge(token_list)

                #print(f"   TOKEN LIST AFTER MERGE:{token_list}")

                encoded_list=[self.vocab_reverse[token] for token in token_list if token in self.vocab_reverse]
                #print(f"    ENCODED LIST:{encoded_list}")
                encoded_list=[self.vocab[token] for token in encoded_list]
                #print(f"    ENCODED LIST (bytes):{encoded_list}")
                bytes_string_to_decode=b"".join(encoded_list)
                #print(f"    bytes_string_to_decode:{bytes_string_to_decode.decode('utf-8')}\n")

                encoded_text_list.extend([self.vocab_reverse[token] for token in encoded_list])

        return encoded_text_list
    
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
                
    
    def encode(self,ori_text:str)->list[int]:
        ''''''
        encoded_text_list=[]
        for chunk in self.chunk_text_by_space(ori_text,100):
            #input("press enter to continue...")
            #print(f"chunk:\n{chunk}")
            encoded_text_list.extend(self._process_encode(chunk))
        return encoded_text_list


    def encode_iterable(self,iterable):#iterable is:Iterable[str] ->Iterable[int]
        for text in iterable:
            encoded_line=self.encode(text)
            for id in encoded_line:
                yield id

    def decode(self,ids:list[int])->str:
        bytes_list=[self.vocab[id] for id in ids]
        bytes_string=b''.join(bytes_list)
        #print(f"bytes_string:{bytes_string}")
        decoded_string=bytes_string.decode("utf-8",errors="ignore")
        return decoded_string

'''
if __name__=="__main__":
    special_tokens = ["<|endoftext|>"]
    vocab_path=VOCAB_PATH
    merges_path=MERGES_PATH
    vocab,merge=get_tokenizer_from_vocab_merges_path(vocab_path,merges_path,special_tokens)
    tokenizer=BPE_Tokenizer(vocab,merge,special_tokens)
    #text="Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    text="Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    #text="Hello, how are you? <|endoftext|>I'm fine, thank you! <|endoftext|>"
    encoded=tokenizer.encode(text)
    print(f"encoded:{encoded}")
    decoded=tokenizer.decode(encoded)
    print(f"decoded:{decoded}")
'''