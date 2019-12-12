from transformers import *
import torch 
import numpy as np 
from utils.pixelcnnpp_utils import *
import pdb
from torch.nn.utils import weight_norm as wn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


def bert_encoder():
    return BERTEncoder()


def class_embedding(n_classes, embedding_dim):
    return nn.Embedding(n_classes, embedding_dim)


def unconditional(n_classes, embedding_dim):
    return nn.Embedding(n_classes, embedding_dim)


class Embedder(nn.Module):
    def __init__(self, embed_size):
        super(Embedder, self).__init__()
        self.embed_size = embed_size

    def forward(self, class_labels, captions):
        raise NotImplementedError


class BERTEncoder(Embedder):
    '''
    pretrained model used to embed text to a 768 dimensional vector
    '''

    def __init__(self):
        super(BERTEncoder, self).__init__(embed_size=768)
        # self.pretrained_weights = 'bert-base-uncased'
        # self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_weights)
        # self.model = BertModel.from_pretrained(self.pretrained_weights)
        self.pretrained_weights = 'gpt2'
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2Model.from_pretrained('gpt2')
        self.max_len = 50

    def tokenize(self, text_batch):
        text_token_ids = [
            torch.tensor(self.tokenizer.encode(string_, add_special_tokens=False, max_length=self.max_len)) for
            string_ in text_batch]
        padded_input = pad_sequence(text_token_ids, batch_first=True, padding_value=0)
        return padded_input

    def forward(self, class_labels, captions):
        '''
        :param class_labels : torch.LongTensor, class ids
        :param list captions: list of strings, sentences to embed
        :return: torch.tensor embeddings: embeddings of shape (batch_size,embed_size=768)
        '''

        padded_input = self.tokenize(captions)
        device = list(self.parameters())[0].device
        padded_input = padded_input.to(device)
        # takes the mean of the last hidden states computed by the pre-trained BERT encoder and return it
        return self.model(padded_input)[0].mean(dim=1)


class OneHotClassEmbedding(Embedder):

    def __init__(self, num_classes):
        super(OneHotClassEmbedding, self).__init__(embed_size=num_classes)
        self.num_classes = num_classes
        self.weights = nn.Parameter(torch.eye(self.num_classes))

    def forward(self, class_labels, captions):
        '''
        :param class_ids : torch.LongTensor, class ids
        :param list text_batch: list of strings, sentences to embed
        :return: torch.tensor embeddings: embeddings of shape (batch_size,embed_size=768)
        '''
        print(class_labels.size(),'class_label_size')
        print(self.weights[class_labels].size(),'embedding_size')
        return self.weights[class_labels]
        


class UnconditionalClassEmbedding(Embedder):
    def __init__(self):
        super(UnconditionalClassEmbedding, self).__init__(embed_size=1)

    def forward(self, class_labels, captions):
        '''
        :param class_ids : torch.LongTensor, class ids
        :param list text_batch: list of strings, sentences to embed
        :return: torch.tensor embeddings: embeddings of shape (batch_size,embed_size=768)
        '''
        print(captions)
        zero = torch.zeros(class_labels.size(0), 1).to(class_labels.device)
        return zero

class GloveEmbedding(Embedder):
    def __init__(self):
        super(GloveEmbedding, self).__init__(embed_size=900)
        self.embeddings_dict = {}
        with open("glove.840B.300d.txt", 'r') as f:
            i = 0
            for line in f:
                if i > 1000000:
                    break
                values = line.split()
                word = values[0]    
                for j in range(1,len(values)):
                    if(values[j][0].isdigit() == False):
                        values[j] = '0.0' 
                vector = np.asarray(values[1:], np.double)
                self.embeddings_dict[word.lower()] = vector
                i = i + 1

    def forward(self, class_labels, captions):
        '''
        :param class_ids : torch.LongTensor, class ids
        :param list text_batch: list of strings, sentences to embed
        :return: torch.tensor embeddings: embeddings of shape (batch_size,embed_size=300)
        '''
        embeddings_list = []
        for caption in captions:
            caption = caption.replace('-',' ')
            words = caption.split()
            #assume that there is at most 3 words, so embed size will be 900 (300 for each)
            res = np.zeros(300,dtype=np.double)
            for i in range(3):
                if(len(words)-1 < i):
                    res = np.concatenate((res,np.zeros(300,dtype=np.double)),axis=None)
                elif words[i].lower() in self.embeddings_dict:
                    res = np.concatenate((res,self.embeddings_dict[words[i].lower()]),axis=None)
                else:
                    res = np.concatenate((res,np.zeros(300,dtype=np.double)),axis=None)
            embeddings_list.append(torch.from_numpy(res[300:]).double())
        embeddings = torch.stack(embeddings_list).to(class_labels.device)
        return embeddings





