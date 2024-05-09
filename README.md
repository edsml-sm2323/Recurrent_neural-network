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
`input_size`: 每个时间步 (timestamp) 包含的feature数量。时间步不同于seq_len, 当seq_len=5时，代表有五个时间步。对于本项目来说，一个时间步就是金价的一个交易日，每个交易日只有一个价格，所以input_size应该为1。

`hidden_size`: 

`num_layers`:

`output`


### The difference between RNN, LSTM and GRU:



