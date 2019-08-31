# phenoinfer_version2

<br>
## the shared link for data
https://drive.google.com/open?id=1w5Shb8zNGv0E-vYurNEKWzAUoMrkHN9e
you may need to download the data and store it in the root directory
<br>
## preprocess the data
go to the directory phenoinfer/preprocessing
execute      python data_preprocessing.py
<br>

## move to /GraphSAGE/data_preprocessing
excucute     python unsupervised_data_processing.py


## move to /GraphSAGE/GraphSAGE
execute      python ../small_graph/gd-G.json ../small_graph/gd-walks.txt

## move to /graphSAGE?GraphSAGE
execute      python random_rank_predict_inner_product.py
