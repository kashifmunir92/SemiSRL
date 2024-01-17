import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from UnlabelData import *
from Utilities import *
import torch.nn.utils.rnn as rnn_utils


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, layer_num):
        super(LanguageModel, self).__init__()
        self.layer_num = layer_num
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim,
                            hidden_dim // 2,
                            layer_num,
                            batch_first=True,
                            bidirectional=True)
        self.line = nn.Linear(hidden_dim, vocab_size)

    def init_weight(self, data):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return (torch.randn(self.layer_num * 2, data.shape[0],
                            self.hidden_dim // 2).to(device),
                torch.randn(self.layer_num * 2, data.shape[0],
                            self.hidden_dim // 2).to(device))

    def forward(self, x, length):
        h = self.init_weight(x)
        embed = self.embed(x)
        packed_embedding = rnn_utils.pack_padded_sequence(embed,
                                                          length,
                                                          batch_first=True)
        out, _ = self.lstm(packed_embedding, h)
        sentence_embeddng, out_len = rnn_utils.pad_packed_sequence(
            out, batch_first=True)
        line = self.line(sentence_embeddng)
        predict = F.log_softmax(line, dim=-1)
        return predict


if __name__ == "__main__":
    batch = 60
    word_2_idx = {'bos': 0, 'eos': 1}
    # read data
    sentences_to_id = []
    sentence = read_file('/SemiSequenceTag/Data/corpus')
    build_voc_size(sentence, word_2_idx)
    for s in sentence:
        s = 'bos ' + s.strip('\n') + ' eos'
        sentences_to_id.append(prepare_sequence(s, word_2_idx))
    training_data = UnlabelData(sentences_to_id)
    dataloader = data.DataLoader(training_data,
                                 batch,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=padd_sentence)
    # build model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LanguageModel(len(word_2_idx) + 3, 100, 50, 2).to(device)
    c = nn.NLLLoss()
    p = torch.optim.Adam(model.parameters(), 0.001)
    x = []
    y = []
    for i in tqdm(range(200)):
        total_loss = 0
        for data, length in dataloader:
            p.zero_grad()
            train = data[:, 0:data.size(-1) - 1].to(device)
            target = data[:, 1:data.size(-1)].reshape(-1).to(device)
            predict = model(train,
                            length).reshape(train.size(0) * train.size(1), -1)
            loss = c(predict, target)
            total_loss = total_loss + loss
            loss.backward()
            p.step()
        x.append(i)
        y.append(total_loss)
    plt.plot(x, y, 'ro-')
    plt.title('loss function')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
