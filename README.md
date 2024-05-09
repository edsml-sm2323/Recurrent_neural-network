# Recurrent Neural Network
Basic RNN, Long Short-Term Memory (LSTM)，Gated Recurrent Unit (GRU). Use the daily price of gold to predict the future gold price.

## Related files: 
`daily_gold_rate.csv`: Dataset for training the model. 
<div align="center">
  <img src="images/data.png" width="500" />
</div> 

`model.py`: Model structure for RNN, LSTM, GRU. 
<div align="center">
  <img src="images/model.png" width="700" />
</div> 

`dataloader.py`: Create dataset suitable for the above models. 

`train.py`: Training the model and get the .pth files.

## Important concepts:
### Suitable data for model:
`batch_first` = False

Input: (seq_len, batch, feature) 

`batch_first` = True

Input: (batch, seq_len, feature) 

### Related parameters:


### The difference between RNN, LSTM and GRU:



