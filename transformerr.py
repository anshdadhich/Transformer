import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import pandas
import re
import matplotlib.pyplot as mat
from torch.cuda.amp import autocast,GradScaler
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

dataset = pandas.read_csv("C:/Users/anshd/OneDrive/Desktop/python/datasets/Conversation.csv",usecols = ["question","answer"])

dataset_str = dataset.astype(str)
length = dataset_str.map(len) 
dataset["length"] = length.sum(axis = 1)

dataset = dataset.sort_values(by = "length")

datasett = pandas.DataFrame()

datasett["question"] = dataset["question"].astype(str)
datasett["answer"] = dataset["answer"].astype(str)

def remove_punctuation(sentence):
    sentence = re.sub(r"[^\w\s]","",sentence)
    sentence = sentence
    return sentence
    
datasett["question"] = datasett["question"].map(remove_punctuation)
datasett["answer"] = datasett["answer"].map(remove_punctuation)

vocab = ["<PAD>", "<UNK>", "<START>", "<END>"] + list(set(word for sentence in datasett["question"] for word in sentence))

word_to_index = {word:index for index,word in enumerate(vocab)}
index_to_word = {index:word for word,index in word_to_index.items()} 

training_data = []
 
for i in range(0,len(dataset)):
    data = datasett["question"].iloc[i]
    target = datasett["answer"].iloc[i]
    training_data.append((data,target))    

class custom_dataset(Dataset):
    def __init__(self,training_data,word_to_index,index_to_word):
        self.question,self.answer = zip(*training_data) 
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word

    def __len__(self):
        return len(self.question)
    
    def __getitem__(self,index):        
        question = re.sub(r"[^\w\s]", "" ,self.question[index]).lower()
        answer = re.sub(r"[^\w\s]", "" ,self.answer[index]).lower()
        
        sentence =  torch.tensor([self.word_to_index.get("<START>")] + [self.word_to_index.get(word,0) for word in (question.split())] + [self.word_to_index.get("<END>")])
                
        target_sentence = torch.tensor([self.word_to_index.get("<START>")] + [self.word_to_index.get(word,0) for word in (answer.split())] + [self.word_to_index.get("<END>")])
                        
        return (sentence, target_sentence)
    
def collate_batch(batch):
    sentences,target = zip(*batch)

    padded_sentences = []
    padded_target_sentences = []
    
    max_sentence = len(max(sentences,key = len))
    max_target = len(max(target,key = len))
    
    max_length = max(max_sentence,max_target)
    
    for sentence in sentences:
        padding = torch.zeros(max_length - len(sentence))
        sentence = torch.cat([sentence,padding])
        padded_sentences.append(sentence)
    
    for target_sentence in target:
        padding = torch.zeros(max_length - len(target_sentence))
        target_sentence = torch.cat([target_sentence,padding])
        padded_target_sentences.append(target_sentence)  
        
    sentences = torch.stack(padded_sentences).long()
    target = torch.stack(padded_target_sentences).long()
                
    return (sentences,target)
        
dataset = custom_dataset(training_data,word_to_index,index_to_word)

training_data = DataLoader(dataset, batch_size = 32, shuffle = True, collate_fn = collate_batch)

# class multi_headed_attention(nn.Module):
#     def __init__(self,embedding_dim,number_of_heads):
#         super().__init__()

#         assert embedding_dim % number_of_heads == 0
         
#         self.head_dimension = embedding_dim // number_of_heads
 
#         self.embedding_dimension = embedding_dim
#         self.number_of_heads = number_of_heads
        
#         self.weight_query = nn.Linear(self.embedding_dimension,self.embedding_dimension)
#         self.weight_key = nn.Linear(self.embedding_dimension,self.embedding_dimension)
#         self.weight_value = nn.Linear(self.embedding_dimension,self.embedding_dimension)
#         self.weight_output = nn.Linear(self.embedding_dimension,self.embedding_dimension)
        
#     def split_heads(self,x):
#         batch_size,seq_len,embedding_dim = x.size()
#         x = x.view(batch_size,seq_len,self.number_of_heads,self.head_dimension).transpose(1,2).contiguous()
#         return x
    
#     def combine_heads(self,x):
#         batch_size,num_heads,seq_len,head_dimension = x.size()
#         x = x.transpose(1,2).contiguous()
#         x = x.view(batch_size,seq_len,self.embedding_dimension)
#         return x
    
#     def scaled_dot_product(self,Q,V,K,mask = None):
#         attention_scores = torch.matmul(Q, K.transpose(-2,-1))/math.sqrt(self.head_dimension)
        
#         if mask is not None:
#             attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
#         attention_probabilities = F.softmax(attention_scores,dim = -1)
#         attention_based_embeddings = torch.matmul(attention_probabilities,V)        
#         return attention_based_embeddings
    
#     def forward(self,Q,K,V,maskk = None):
#         Q = self.split_heads(self.weight_query(Q))
#         K = self.split_heads(self.weight_key(K))
#         V = self.split_heads(self.weight_value(V))
        
#         atten_embeddings = self.scaled_dot_product(Q,K,V,mask = maskk)
        
#         embeddings = self.weight_output(self.combine_heads(atten_embeddings))
        
#         return embeddings
    
# class positional_encodings(nn.Module):
#     def __init__(self,embedding_dimension,sentence_length = len(max(datasettt,key = len))):
#         super().__init__()
        
#         self.embedding_dimension = embedding_dimension 
#         self.sentence_length = sentence_length
#         self.positional_encoding = torch.empty(sentence_length,embedding_dimension)
        
#         position = torch.arange(0,sentence_length,dtype = torch.float).unsqueeze(1)
        
#         div_term = torch.exp(torch.arange(0,embedding_dimension//2).float() * (-math.log(10000)/ embedding_dimension))
        
#         div_term.unsqueeze(0)
        
#         self.positional_encoding[:,0::2] = torch.sin(position * div_term)
#         self.positional_encoding[:,1::2] = torch.cos(position * div_term)
        

#         self.register_buffer("positional_encodings", self.positional_encoding)

#     def forward(self,x): 
#         return x + self.positional_encoding[:x.size(1),:]
    
# class feed_forward(nn.Module):
#     def __init__(self,embedding_dimension):
#         super().__init__()
        
#         self.fc1 = nn.Linear(embedding_dimension,300)
#         self.out = nn.Linear(300,embedding_dimension)
        
#     def forward(self,x):
#         x = F.relu(self.fc1(x))
#         x = self.out(x)
#         return x
        
        
# class Encoder(nn.Module):
#     def __init__(self,embedding_dimension,number_of_heads):
#         super().__init__()
        
#         self.self_attention = multi_headed_attention(embedding_dimension,number_of_heads)
#         self.feed_forward = feed_forward(embedding_dimension)
#         self.norm1 = nn.LayerNorm(embedding_dimension)
#         self.norm2 = nn.LayerNorm(embedding_dimension)
        
#     def forward(self,x,input_padding_mask):
#         attention_scores = self.self_attention(x,x,x,input_padding_mask= None)
#         x = self.norm1(x + attention_scores)
#         feed_forwarded_x = self.feed_forward(x)
#         x = self.norm2(x + feed_forwarded_x)
#         return x
    
# class Decoder(nn.Module):
#     def __init__(self,embedding_dimension,number_of_heads):
#         super().__init__()
        
#         self.self_attention = multi_headed_attention(embedding_dimension,number_of_heads)
#         self.cross_attention = multi_headed_attention(embedding_dimension,number_of_heads)
#         self.feed_forward = feed_forward(embedding_dimension)
#         self.norm1 = nn.LayerNorm(embedding_dimension)
#         self.norm2 = nn.LayerNorm(embedding_dimension)
#         self.norm3 = nn.LayerNorm(embedding_dimension)

        
#     def forward(self,x,encoder_output,target_mask,input_padding_mask = None):
#         attention_scores = self.self_attention(x,x,x,target_mask)
#         x = self.norm1(x + attention_scores)
#         cross_attention_scores = self.cross_attention(x,encoder_output,encoder_output,input_padding_mask)
#         x = self.norm2(x + cross_attention_scores)
#         feed_forwarded_x = self.feed_forward(x)
#         x = self.norm3(x + feed_forwarded_x)
#         return x       

# class transformer(nn.Module):
#     def __init__(self,embedding_dimension,number_of_heads,number_of_layers):
#         super().__init__()
        
#         self.encoder_embeddings = nn.Embedding(len(vocab),embedding_dimension)
#         self.decoder_embeddings = nn.Embedding(len(vocab),embedding_dimension)
#         self.positional_encodings = positional_encodings(embedding_dimension,sentence_length = len(max(datasettt,key = len)))
        
#         self.Encoder_layers = nn.ModuleList([Encoder(embedding_dimension,number_of_heads) for layer in range(number_of_layers)]) 
        
#         self.Decoder_layers = nn.ModuleList([Decoder(embedding_dimension,number_of_heads) for layer in range(number_of_layers)]) 
#         self.Decoder = Decoder(embedding_dimension,number_of_heads)
        
#         self.output = nn.Linear(embedding_dimension,len(vocab))
        
#     def generate_mask(self,sentence_length):
#         mask = torch.tril(torch.ones(1,sentence_length,sentence_length),diagonal = 1)
#         return mask

#     def create_key_padding_mask(self,sequence):
#          mask = (sequence != 0)
#          mask = mask.float()
#          return mask
 
#     def forward(self,sentence,target = None,input_padding_mask,target_mask = 1):
#         sentence_length = sentence.size(1)
#         sentence = self.encoder_embeddings(sentence)
#         sentence = self.positional_encodings.forward(sentence)

#         input_padding_mask = self.create_key_padding_mask(sentence)
#         target_padding_mask = self.create_key_padding_mask(target)
        
#         if target is None:
#           target = torch.zeros(sentence.size(0),sentence.size(1)).long()

#         target = self.decoder_embeddings(target)
#         target = self.positional_encodings.forward(target)

#         for encoder in self.Encoder_layers:
#             encoder_output = encoder.forward(sentence,input_padding_mask)

#         if target_mask is not None:
#            target_mask = self.generate_mask(sentence_length)
#            target_mask = target_mask & target_padding_mask

#         for Decoder in self.Decoder_layers:
#             target = Decoder.forward(target,encoder_output,target_mask,target_padding_mask)

#         output = self.output(target)
#         return output

# model = transformer(128,4,2)

# loss = nn.CrossEntropyLoss(ignore_index = 0)

# optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)

# scaler = GradScaler()

# losses = []

# for epoch in range(50):
#     for batch,(sentence,target) in enumerate(training_data):
          
#         torch.autograd.set_detect_anomaly(True)

#         optimizer.zero_grad()

#         predicted = model.forward(sentence,target)
        
#         predictedd  = predicted

#         predicted = predicted.view(-1,predicted.size(-1))
#         target = target.view(-1)

#         losss = loss(predicted,target)

#         losses.append(losss.detach().item())

#         scaler.scale(losss).backward()
#         scaler.step(optimizer)
#         scaler.update()
        
#     print(f"loss at epoch {epoch} is {losss}")
  
#     model.eval()
#     with torch.no_grad():
#         predictedd = torch.argmax(predictedd,dim = -1).cpu().tolist()
#         for sentence in predictedd:
#             i = 0
#             print(sentence)
#             for word in sentence:
#                 print(dataset.index_to_word.get(word), end = " ")
#                 i += 1
#                 if(i>10):
#                     break
#             break
#             
#     if(losss.item() < 0.003):
#         break
    
    
# mat.plot(range(len(losses)),losses)
# mat.xlabel("epochs")
# mat.ylabel("loss")
# mat.show()

# torch.save({"model_state_dict" : model.state_dict(),
#             "word_to_index" : dataset.word_to_index,
#             "index_to_word" : dataset.index_to_word,
#             },"transformer.pt")

# loaded = torch.load("transformer.pt")
    
# with torch.no_grad():
#     model.load_state_dict(loaded["model_state_dict"])
#     word_to_index = loaded["word_to_index"]
#     index_to_word = loaded["index_to_word"] 
    
#     text = "the man was walking"
    
#     text = torch.tensor([word_to_index.get(word, word_to_index.get("<UNK>")) for word in text.split()])

#     text = text.unsqueeze(0).long()

#     predicted = torch.argmax(model.forward(text),dim = -1)
#     predicted = predicted.squeeze().tolist()

#     predicted = [index_to_word.get(word,"<UNK>") for word in predicted]
#     predicted = " ".join(predicted)
#     print(predicted)

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embeddings = nn.Embedding(len(vocab),256)
        self.transformer = nn.Transformer(256,4,2,2,1024,batch_first = True,layer_norm_eps = 0.0001)
        self.out = nn.Linear(256,len(vocab))
        
    def generate_mask(self,sentence_length,):
        mask = torch.tril(torch.ones(sentence_length,sentence_length),diagonal = 1)
        return mask.masked_fill(mask == 0,-1e9)

    def create_key_padding_mask(self,sequence):
         mask = (sequence != 0)
         mask = mask.float()
         return mask.masked_fill(mask == 0,-1e9)
        
    def encoder(self,sentence,src_key_padding_mask = None):
        sentence = self.embeddings(sentence.long())
        
        output =  self.transformer.encoder(sentence,src_key_padding_mask = src_key_padding_mask)
                
        return output
    
    def decoder(self,encoder_output,target,memory_key_padding_mask = None, target_key_padding_mask = None,no_peek_mask = None):
        
        target = self.embeddings(target.long())
        decoder_output = self.transformer.decoder(encoder_output,target,memory_key_padding_mask = memory_key_padding_mask,tgt_key_padding_mask = target_key_padding_mask,tgt_mask = no_peek_mask)

        output = self.out(decoder_output)
        
        return output
    
    def scheduled_sampling(self,sentence,target,teacher_forcing_ratio,src_key_padding_mask):
        batch_size,target_length = target.size()
        output = torch.zeros(batch_size,target_length,len(vocab)).to(device)
                
        encoder_output = self.encoder(sentence,src_key_padding_mask = src_key_padding_mask)

        decoder_input = target[:,-1:].to(device)
        
        use_teacher_forcing = teacher_forcing_ratio > random.random()
                
        for t in range(target_length):
           target_mask = self.generate_mask(decoder_input.size(1)).to(device)

           decoder_output = self.decoder(encoder_output,decoder_input, memory_key_padding_mask = src_key_padding_mask, no_peek_mask = target_mask) 
           
           output[:,t] = decoder_output[:,-1].to(device)
           
           if use_teacher_forcing:
              decoder_input = torch.cat([decoder_input,target[:,t].unsqueeze(1)], dim = 1).to(device)
           else:
               last_predicted_word = decoder_output[:,-1].argmax(1).to(device)
               decoder_input = torch.cat([decoder_input,last_predicted_word.unsqueeze(1)],dim = 1).to(device)
        
        return output
              
Model = Transformer().to(device)

pad_index = dataset.word_to_index.get("<PAD>")

loss = nn.CrossEntropyLoss(ignore_index = pad_index)

losses = []

optimizer = torch.optim.Adam(Model.parameters(), lr = 0.0001)
scaler = GradScaler()

for epoch in range(100):

    Model.train()
    for batch,(sentence,target) in enumerate(training_data):
        
        sentence = sentence.to(device)
        target = target.to(device)

        torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()

        batch_size = sentence.size(0)
        sentence_length = sentence.size(1) 
        target_length = target.size(1)
        
        end_token_index = dataset.word_to_index.get("<END>")
        
        with autocast():        
             src_key_padding_mask = Model.create_key_padding_mask(sentence).to(device)
             target_key_padding_mask = Model.create_key_padding_mask(target).to(device)
             target_mask = Model.generate_mask(target_length).to(device)
             
             teacher_forcing_ratio = max(0.5,1-(epoch/50)) 
             
             predicted = Model.scheduled_sampling(sentence,target,teacher_forcing_ratio,src_key_padding_mask)
             
             predictedd = predicted
                
             end_mask = (target != pad_index)
             end_mask = end_mask.unsqueeze(-1).float()
                                                                
             predicted = predicted * end_mask
             
             predicted = predicted.view(-1,predicted.size(-1))
             target = target.view(-1)
     
             losss = loss(predicted,target)
     
             losses.append(losss.detach().item())
     
             torch.nn.utils.clip_grad_norm_(Model.parameters(),max_norm = 0.5)
             
             scaler.scale(losss).backward()
             scaler.step(optimizer)
             scaler.update()
        
        if(losss.item() < 0.003):
            break
        
    print(f"loss at epoch {epoch} is {losss}")

    Model.eval()
    with torch.no_grad():
        predictedd = torch.argmax(predictedd,dim = -1).cpu().tolist()
        for sentence in predictedd:
            i = 0
            print(sentence)
            for word in sentence:
                print(dataset.index_to_word.get(word), end = " ")
                i += 1
                if(i>10):
                    break
            break
            
    if(losss.item() < 0.003):
        break
    
mat.plot(range(len(losses)),losses)
mat.xlabel("epochs")
mat.ylabel("loss")
mat.show()

torch.save({"model_state_dict" : Model.state_dict(),
            "word_to_index" : dataset.word_to_index,
            "index_to_word" : dataset.index_to_word,
            },"transformer.pt")

torch.cuda.empty_cache()
loaded = torch.load("transformer.pt", map_location = "cpu")
    
with torch.no_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    predicted = "<START>"
    last_word = ["<START>"]
    
    Model.load_state_dict(loaded["model_state_dict"])
    Model = Model.to(device)
    
    for parameter in Model.parameters():
        parameter.requires_grad = False
        
    word_to_index = loaded["word_to_index"]
    index_to_word = loaded["index_to_word"] 
    
    Model.eval()

    torch.cuda.empty_cache()
    
    i = 0
    while (i != 10):          
           text = "hello, how are you doing?"
           text = re.sub(r"[^\w\s]","",text).lower()
           
           target = predicted
           
           target_mask = Model.generate_mask(len(target.split()))
           
           text = torch.tensor([word_to_index.get(word, word_to_index.get("<UNK>")) for word in text.split()])
           output = torch.tensor([word_to_index.get(word) for word in target.split()])
       
           text = text.unsqueeze(0).long().to(device)
           output = output.unsqueeze(0).long().to(device)
       
           predictedd = torch.argmax(Model.forward(text,target = output,no_peek_mask = target_mask),dim = -1)
                      
           if (predictedd.size(1) > 1):
               predictedd = predictedd[0,-1].tolist()   
               predicted_list = index_to_word.get(predictedd)
               last_word.append(predicted_list)
               predicted = " ".join(last_word)
               print(predicted)
               torch.cuda.empty_cache()
               i += 1 
           else:
               predictedd = predictedd.squeeze().tolist()
               print(predictedd)
               last_word.append(index_to_word.get(predictedd))
               predicted = " ".join(last_word)
               print(predicted)
               i += 1
               torch.cuda.empty_cache()
            
    print(predicted)
    
    torch.cuda.empty_cache()