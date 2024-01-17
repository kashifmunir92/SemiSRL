import torch
import torch.nn as nn
import numpy as np
import BuildTransformer
import Dataset
import torch.utils.data as data
from tqdm import tqdm


if __name__ == "__main__":
    # config
    batch = 16
    max_length = 3
    # define model
    sentence1 = torch.randint(1, 3, (100, max_length))
    sentence2 = torch.randint(1, 4, (100, max_length))
    dataset = Dataset.Data(sentence1, sentence2)
    dataloader = data.DataLoader(dataset, batch, shuffle=False, num_workers=4)
    model = BuildTransformer.make_model(3, 4)
    criterion = nn.CrossEntropyLoss()
    optimize = torch.optim.Adam(model.parameters(), 0.0001)
    for i in tqdm(range(50)):
        total_loss = 0
        for i, sentence in enumerate(dataloader):
            x = model(sentence[0], sentence[0], None, None)
            optimize.zero_grad()
            loss = criterion(
                x.reshape(x.size(0)*x.size(1), x.size(-1)), sentence[0].reshape(sentence[0].size(-2)*sentence[0].size(-1)))
            loss.backward()
            optimize.step()
            total_loss += loss
        print('loss is %f' % (total_loss))

    for i, sentence in enumerate(dataloader):
        x = model(sentence[0], sentence[0], None, None)
        v, index = torch.max(x, -1)
        print(sentence[0])
        print(index)
