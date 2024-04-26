# Recurrent_neural-network
Basic RNN, Long Short-Term Memory (LSTM)，Gated Recurrent Unit (GRU)

## Project 1: Use LSTM to predict gold prices (pytorch)
### Related files: 
Datasets: `annual_gold_rate.csv` and `daily_gold_rate.csv`. 

Model construction and training: `gold_rate_predict.ipynb`. 

Trained model: `best_model.pth`. 

### Specific implementation:
Step 1: Conduct data analysis, find missing values, outliers, etc., and check the correlation between each data. 

Step 2: Use pytorch to build a suitable GoldPriceDataset so that the Dataloader can output data suitable for LSTM. 

Step 3: training and prediction. 

