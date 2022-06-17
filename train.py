import numpy as np;
import torch;

from model import EncodeNetwork, LSTM_Network, DecodeNetwork
import torch.utils.data as data

from dataset import R2N2_Data

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    epochs = 50
    batch_size = 20
    
    dataset = R2N2_Data('PlaneRenderings/', 'PlaneVoxels/',k=5)
    #increase number of workers to CPU count
    data_batches = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    #setup the network
    encoder = EncodeNetwork().cuda()
    lstm = LSTM_Network().cuda()
    decoder = DecodeNetwork().cuda()

    #setup loss and optimizer
    nll = torch.nn.NLLLoss()
    adam = torch.optim.Adam([ {'params': encoder.parameters()}, {'params': lstm.parameters()}, {'params': decoder.parameters()}], lr=1e-5)
    
    #scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(adam, T_max=55, eta_min=0, last_epoch=-1, verbose=False)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(adam, [5,10,12,15,18,20,22,25,28,30,32,35,38,40,42,45], gamma=0.5)
    for i in range(epochs):
        print('Epoch:', i+1)
        
        
        training_error = 0
        ct = 0
        #updating by batches 
        for idx,data in enumerate(data_batches):
            ct += 1
            
            adam.zero_grad()
            occ_grid = data['label']
            h0 = torch.zeros((batch_size, 128, 4, 4, 4)).cuda()
            s0 = torch.zeros((batch_size, 128, 4, 4, 4)).cuda()
            for im in data['data']:
                s0,h0 = lstm(encoder(im.cuda()).cuda(),s0.cuda(),h0.cuda())
            decoded = decoder(h0).cuda()
            loss = nll(decoded, occ_grid.cuda()).cuda()
            training_error += loss
            loss.backward()
            adam.step()
        scheduler.step()
        training_error /= ct
        print(f'    Training Error: {training_error}')
        #after done with all batches
        if i%10 == 0:
            chkpt = {'Encoder Params': encoder.state_dict(), 'LSTM Params': lstm.state_dict(), 'Decoder Params': decoder.state_dict() }
            torch.save(chkpt, f'chkpt/epoch{i+1}.pt')
    
    chkpt = {'Encoder Params': encoder.state_dict(), 'LSTM Params': lstm.state_dict(), 'Decoder Params': decoder.state_dict() }
    torch.save(chkpt, f'chkpt/final.pt')
    print('Finally Finished! :}')
        
