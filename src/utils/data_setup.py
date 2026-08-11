import torch
from torch.utils.data import Dataset
import numpy as np



class KmerTokenizer:
    def __init__(self, k=6, stride=6, vocab=None):
        """
        Args:
            k: k-mer 的长度
            stride: 步长
            vocab: 预训练模型的词表 (dict: token -> id)
        """
        self.k = k
        self.stride = stride
        
        if vocab is None:
            # 默认兜底 (仅用于测试，实际运行时 main.py 会传入 vocab)
            self.vocab = {'[PAD]': 1, '[UNK]': 0, '[CLS]': 3, '[MASK]': 2, '[SEP]': None}
            for kmer_tuple in product('ACGT', repeat=k):
                kmer = "".join(kmer_tuple)
                self.vocab[kmer] = len(self.vocab)
        else:
            self.vocab = vocab
            
        self.vocab_size = len(self.vocab)

        self.cls_token_id = self.vocab.get('[CLS]', self.vocab.get('<s>', self.vocab.get('<cls>', 2)))
        

        self.unk_token_id = self.vocab.get('[UNK]', self.vocab.get('<unk>', 3))
        
    
        self.pad_token_id = self.vocab.get('[PAD]', self.vocab.get('<pad>', 1))


        print(f"[Tokenizer Info] CLS ID: {self.cls_token_id}, PAD ID: {self.pad_token_id}, UNK ID: {self.unk_token_id}")

    def __call__(self, text, max_length=512, padding='max_length', truncation=True, return_tensors=None):
    
        text = str(text).upper()
        text = text.replace('N', 'A')
  
        kmers = [text[i:i+self.k] for i in range(0, len(text) - self.k + 1, self.stride)]
        

        input_ids = [self.cls_token_id]
        
        for kmer in kmers:
    
            input_ids.append(self.vocab.get(kmer, self.unk_token_id))
            
 
        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            
    
        attention_mask = [1] * len(input_ids)
        if padding == 'max_length':
            pad_len = max_length - len(input_ids)
            if pad_len > 0:
  
                input_ids += [self.pad_token_id] * pad_len
                attention_mask += [0] * pad_len
                
  
        if return_tensors == 'pt':
            return {
                'input_ids': torch.tensor([input_ids], dtype=torch.long),
                'attention_mask': torch.tensor([attention_mask], dtype=torch.long)
            }
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


class PairSeqData(Dataset):
    def __init__(self, train_pairs, raw_sequences, tokenizer, max_len=128, transform=None):

        self.train_pairs = train_pairs 
        self.raw_sequences = raw_sequences
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

        if len(self.train_pairs) != len(self.raw_sequences):
            print(f"Warning: Images ({len(self.train_pairs)}) and Sequences ({len(self.raw_sequences)}) length mismatch!")

    def __len__(self) -> int:
        return len(self.train_pairs)

    def __getitem__(self, index: int):
 
        original_chaos = torch.Tensor(self.train_pairs[index][0]).unsqueeze(dim=0)
        mimic_chaos = torch.Tensor(self.train_pairs[index][1]).unsqueeze(dim=0)

        if self.transform:
            img1 = self.transform(original_chaos.repeat(3, 1, 1))
            img2 = self.transform(mimic_chaos.repeat(3, 1, 1))
        else:
            img1 = original_chaos
            img2 = mimic_chaos

  
        id1 = torch.zeros(self.max_len, dtype=torch.long)
        mask1 = torch.zeros(self.max_len, dtype=torch.long)
        id2 = torch.zeros(self.max_len, dtype=torch.long)
        mask2 = torch.zeros(self.max_len, dtype=torch.long)

        if self.tokenizer is not None:
     
            seq1_text = str(self.raw_sequences[index][0]) # Weak View Text
            seq2_text = str(self.raw_sequences[index][1]) # Strong View Text

            # === Tokenize Text 1 (对应 img1) ===
            enc1 = self.tokenizer(
                seq1_text,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            id1 = enc1['input_ids'].squeeze(0)
            mask1 = enc1['attention_mask'].squeeze(0)

            # === Tokenize Text 2 (对应 img2) ===
            enc2 = self.tokenizer(
                seq2_text,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            id2 = enc2['input_ids'].squeeze(0)
            mask2 = enc2['attention_mask'].squeeze(0)

      
        return img1, img2, id1, mask1, id2, mask2


class SeqData(Dataset):
    def __init__(self, fcgr_images, raw_sequences, labels, classes, class_to_idx, tokenizer, max_len=128, transform=None):

        self.fcgr_images = fcgr_images
        self.raw_sequences = raw_sequences 
        self.labels = labels
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

    def __len__(self) -> int:
        return len(self.fcgr_images)

    def __getitem__(self, index: int):

        chaos = torch.Tensor(self.fcgr_images[index]).unsqueeze(dim=0)
        
        class_idx = self.labels[index]
        if not isinstance(class_idx, torch.Tensor):
            class_idx = torch.tensor(class_idx, dtype=torch.long)

        if self.transform:
            img = self.transform(chaos.repeat(3, 1, 1))
        else:
            img = chaos

    
        input_ids = torch.zeros(self.max_len, dtype=torch.long)
        attention_mask = torch.zeros(self.max_len, dtype=torch.long)

        if self.tokenizer is not None:
  
            seq = str(self.raw_sequences[index]) 
            encoding = self.tokenizer(
                seq,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)

     
        return img, input_ids, attention_mask, class_idx