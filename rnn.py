import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
from torchvision import transforms
BATCH_SIZE = 100
INPUT_STEP = 28
INPUT_SIZE = 28
OUTPUT_SIZE = 10
HIDE_SIZE = 100
data_df = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=data_df,    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=True,
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class RNN(nn.Module):
    def __init__(self,in_dim,lstm_hid,out_dim):
        super(RNN,self).__init__()
        self.lstm = nn.LSTM(in_dim,lstm_hid,batch_first=True)
        self.classifier = nn.Linear(lstm_hid,out_dim)

    def forward(self, x):
        out,state = self.lstm(x)
        return F.softmax(  self.classifier(out[:,-1,:]) ,1 )

if __name__ == "__main__":
    rnn = RNN(INPUT_SIZE,HIDE_SIZE,OUTPUT_SIZE)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
    loss_fun = nn.CrossEntropyLoss()
    for ep in range(10):
        i = 0
        for (x,y_) in train_loader:
            #x = x.view(BATCH_SIZE,INPUT_STEP,INPUT_SIZE)
            x = torch.squeeze(x)
            x = Variable(x)
            y_ = Variable(y_)

            y = rnn(x)
            loss = loss_fun(y,y_)
            _,pred = torch.max(y,1)
            num_correct = (pred == y_).sum()
            i = i+1
            if i % 100==0:
                print(loss.data[0],num_correct.data[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
