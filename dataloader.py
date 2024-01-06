from torch.utils.data import DataLoader
from transformers import default_data_collator






def loader(data_packed, batch_size, flag):

    shuffle = True if flag == 'train' else False
    
    dataloader = DataLoader( data_packed,
                                batch_size=batch_size,
                                shuffle = shuffle,
                                collate_fn=default_data_collator  # batch
                                )
    return dataloader

def loadercheck(dataloader):
    b = next(iter(dataloader))
    print('loader key:', b.keys())
    print('input_ids:',  b["input_ids"][0][:25])
    print('labels:',b["labels"][0][:25] )


