from help import *
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from ruletaker_utils import score_pn_proof, load_jsonl, make_ruletaker_slots, make_input_output_string, decompose_slots


class MyDataset(Dataset):
    def __init__(self,train_data,tokenizer):
        #print("dataset",train_data)
        self.data=train_data


        self.tokenizer=tokenizer


    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):

        input,output=make_input_output_string(self.data[idx],[['question','context'],['answer','proof']])       #input은 $answer$ ; $proof$ ; $question$ = Harry is big? ; $context$ = sent1: If Harry is rough and Harry is cold then Harry is furry.
                                                                                                            #output은 output $answer$ = True ; $proof$ = # sent5@int1 & # sent21@int2 sent11 #  형태

        #print("input[idx]",input)
        #print("output[idx]", output)
        input=self.tokenizer([input],padding="longest",max_length=1024,truncation=True, return_tensors="pt")
        output=self.tokenizer([output],padding="longest", max_length=1024, truncation=True)
        item={
            'input' :torch.LongTensor(input.input_ids).squeeze(0),               #T5는 어떤 feature을 input으로 받음??? 이걸 collater에서 그대로 쓰면 됨
            'attention_mask' :torch.LongTensor(input.attention_mask).squeeze(0),
            'labels':torch.LongTensor(output.input_ids).squeeze(0)              #
        }

        return item


def collate_fn(items):
        #for sentence in items:
            #print("sentence['input']",sentence['input'])
        input_id=pad_sequence([sentence['input'] for sentence in items], batch_first=True, padding_value=0)
        input_attn_mask=(input_id!=0).long()
        output_id=pad_sequence([sentence['labels'] for sentence in items], batch_first=True, padding_value=0)
        batch={
            "input_ids":input_id,
            "attention_mask":input_attn_mask,
            "labels":output_id
        }
        return batch                #batch.items() 을 input으로 사용해야함. batch item이 위에서 정의한 사전 데이터타입. 각각 value들은 위에서 정의했듯이 패딩된 2차원 tensor

