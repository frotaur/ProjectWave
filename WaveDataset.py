import torch
from torch.utils.data import Dataset
import os
from scipy import io


class WaveDataset(Dataset):
    """
        Dataset class for training transformers on wave equations
    
        Parameters :
        data_directory : str
            String telling the path where the data is. Inside 
            data_directory there should be three folders, sources,
            mics and wave_solution.
    """

    def __init__(self, data_directory):
        super().__init__()

        self.datadir = data_directory

        self.datafiles=[os.path.join(self.datadir,file) for file in os.listdir(self.datadir) if file!='.DS_Store']

        self.length = len(self.datafiles)

        self.datadict = {k : io.loadmat(k) for k in self.datafiles}
        print(f'Initialized dataset with {self.length} examples')



    def __len__(self):
        return self.length
    

    def __getitem__(self, index):
        # data = io.loadmat(self.datafiles[index])
        data = self.datadict[self.datafiles[index]]
        ## Option : combine the data so that its (B,M,T,3) and (N,2)
        ## Or, do it inside the model
        #mics' dimension:(6,2)（M,2);values' dimension:(6,512)(M,T);output dimension:(M,T,3)=(6,512,3)
        mics=torch.from_numpy(data['mics']) # (M,2)
        # TEMPORARY 3-mic
        values = torch.from_numpy(data['values'])[None,:,:] # (1,M,T)

        T = values.shape[-1]

        mics = mics.permute(1,0) #(2,M)
        mics = mics[:,:,None].expand(-1,-1,T) # (2,M,T)

        return torch.from_numpy(data['Org']),torch.cat((values,mics),dim=0)[:,:,:160] #(N,2), (3,M,T)

#.squeeze()
if __name__=='__main__':
    testDataset = WaveDataset('./data')

    print(f'The dataset has {len(testDataset)} examples')
    source, mics, values = testDataset[0]

    print(f''' Source shape : {source.shape}\n
             mics shape : {mics.shape}\n
             values shape : {values.shape}''')