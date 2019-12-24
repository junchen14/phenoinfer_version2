# phenoinfer_version2

<br>
# go to data directory
## download the data from the following link into phenoifner_version2/data directory

https://drive.google.com/open?id=1w5Shb8zNGv0E-vYurNEKWzAUoMrkHN9e
you may need to download the data and store it in the root directory


## preprocess the data
## there are 5 different types of data: union, intersection, go, mp, uberon
let us have union as example for the following operation
then execute

    sh data_preprocessing.sh



## move your current directory to /GraphSAGE/graphsage
execute

    python random_rank_predict_inner_product.py union


## prediction part
go to the directory of GraphSAGE/graphsage

### but please modify the comment part in the code
execute    

    python test_overfitting.py

