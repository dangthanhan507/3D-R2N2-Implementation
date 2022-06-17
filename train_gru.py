import numpy as np;
import torch;

from deep_model import DeepEncodeNetwork, GRU_Network, DeepDecodeNetwork
import torch.utils.data as data

from dataset import R2N2_Data

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    epochs = 120
    batch_size = 20
    
    with open('chkpt_gru/final.pt', 'rb') as f: 
        loaded = torch.load(f, 'cuda')
    
    
    dataset = R2N2_Data('PlaneRenderings/', 'PlaneVoxels/',k=5)
    #increase number of workers to CPU count
    data_batches = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    #setup the network
    encoder = DeepEncodeNetwork().cuda()
    gru = GRU_Network().cuda()
    decoder = DeepDecodeNetwork().cuda()

    encoder.load_state_dict(loaded['Encoder Params'])
    gru.load_state_dict(loaded['GRU Params'])
    decoder.load_state_dict(loaded['Decoder Params'])
    
    
    #setup loss and optimizer
    nll = torch.nn.NLLLoss()
    adam = torch.optim.Adam([ {'params': encoder.parameters()}, {'params': gru.parameters()}, {'params': decoder.parameters()}], lr=1e-5)
    
    scheduler=torch.optim.lr_scheduler.MultiStepLR(adam, [5,10,15,20,25,30,35,40,45], gamma=0.5)
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
            for im in data['data']:
                h0 = gru(encoder(im.cuda()).cuda(),h0.cuda())
            decoded = decoder(h0).cuda()
            loss = nll(decoded, occ_grid.cuda()).cuda()
            training_error += loss
            loss.backward()
            adam.step()
        scheduler.step()
        training_error /= ct
        print(f'    Training Error: {training_error}')
        #after done with all batches
        if (i+1)%10 == 0:
            chkpt = {'Encoder Params': encoder.state_dict(), 'GRU Params': gru.state_dict(), 'Decoder Params': decoder.state_dict() }
            torch.save(chkpt, f'chkpt_gru/epoch{i+1}.pt')
    
    chkpt = {'Encoder Params': encoder.state_dict(), 'GRU Params': gru.state_dict(), 'Decoder Params': decoder.state_dict() }
    torch.save(chkpt, f'chkpt_gru/final.pt')
    print('Finally Finished! :}')
        
