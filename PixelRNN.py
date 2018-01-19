import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import re


HIDE_LAYER = 500
CORLOR_CNT = 167
BITCH_SIZE = 50
STEP_SIZE = 100
IMAGE_HIGH = 20
IMAGE_WIDTH = 20
IMAGE_SIZE = 400

def getcolormap():
    with open('colormap.txt') as file:
        lines = file.readlines()
        color_map={}
        for (i,line) in enumerate(lines):
            ans = re.split('\\D', line.strip())
            ans = [ int(ans[i]) for i in range(3) ]
            color_map[i]=ans
        return color_map

g_color_map = getcolormap()


def show_img(im,colormap):
    img = np.array(im)
    img = img.reshape([IMAGE_HIGH, IMAGE_WIDTH])
    out=[]
    for i in range(len(img)):
        out.append([])
        for j in range(len(img[i])):
            out[i].append([])
            out[i][j]=colormap[ img[i][j] ]
    print(np.array(out))
    plt.imshow(np.array(out,dtype=np.ubyte))
    plt.show()


def getbitch(bitch_size,step_size):
    bitch_list=[]
    x = []
    y = []
    with open('pixel_color.txt') as file:
        lines = file.readlines()
        for line in lines:
            ans = re.split('\\D',line.strip())
            ans = [int(ans[i]) for i in range(len(ans))]
            start = 0
            while start+step_size+1 < len(ans):
                core = ans[start:start+step_size]
                x.append(core)
                exp = ans[start+1:start+step_size+1]
                y.append(exp)
                start+=3
                if len(x)==bitch_size:
                    bitch_list.append(x)
                    bitch_list.append(y)
                    x=[]
                    y=[]
    return bitch_list


class PixelRNN(nn.Module):
    def __init__(self,in_dim,lstm_hid,out_dim):
        super(PixelRNN,self).__init__()
        self.lstm = nn.LSTM( lstm_hid ,lstm_hid,batch_first=True,num_layers = 1)
        self.classify = nn.Linear(lstm_hid,out_dim)
        self.embed = nn.Embedding( CORLOR_CNT, lstm_hid )
        self.optimizer = torch.optim.Adagrad(self.parameters(), lr=0.001)
        self.lossfun = nn.CrossEntropyLoss()
    def forward(self,x):
        x = self.embed(x)
        out,state = self.lstm(x)
        outs=[]
        for i in range(out.size()[1]):
           outs.append(self.classify(out[:,i,:]))
        outs = torch.stack(outs,1)
        return outs
    def train(self):
        ls = getbitch(BITCH_SIZE,STEP_SIZE)
        for epol in range(10):
            for i in range( len(ls)//2 ):
                x = Variable( torch.LongTensor(ls[i*2]) )
                y_= Variable( torch.LongTensor(ls[i*2+1]) )
                y = self.forward(x)
                _,pred = torch.max( y,2 )
                y = y.view(-1,CORLOR_CNT)
                y_ =y_.view(-1)
                pred = pred.view(-1)
                loss = self.lossfun(y,y_)
                num_correct = (pred == y_).sum()
                if i%100 == 0:
                    print(loss.data[0],num_correct.data[0])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    
if __name__=="__main__":
    rnn = PixelRNN(1,HIDE_LAYER,CORLOR_CNT)
    rnn.train()




