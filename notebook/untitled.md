## Evaluation

Submissions will be evaluated based on their mean F1 score.

### Submission File
For each orderid in the test set, you should predict a space-delimited list of productids for that order. If you wish to predict an empty order, you should submit an explicit 'None' value. You may combine 'None' with product_ids. The spelling of 'None' is case sensitive in the scoring metric. The file should have a header and look like the following:

```
order_id,products  
17,1 2  
34,None  
137,1 2 3  
etc.
```

## Data Description

The dataset for this competition is a relational set of files describing customers' orders over time. The goal of the competition is to predict which products will be in a user's next order. The dataset is anonymized and contains a sample of over 3 million grocery orders from more than 200,000 Instacart users. For each user, we provide between 4 and 100 of their orders, with the sequence of products purchased in each order. We also provide the week and hour of day the order was placed, and a relative measure of time between orders. For more information, see the blog post accompanying its public release.

## Discussion

- 2th　　
https://www.kaggle.com/c/instacart-market-basket-analysis/discussion/38143  
https://github.com/KazukiOnodera/Instacart  
- 3rd　　
https://www.kaggle.com/c/instacart-market-basket-analysis/discussion/38097  
- 4th　　
https://www.kaggle.com/c/instacart-market-basket-analysis/discussion/38102  
- 6th　　
https://www.kaggle.com/c/instacart-market-basket-analysis/discussion/38112  
- 9th　　
https://www.kaggle.com/c/instacart-market-basket-analysis/discussion/38100 
- 11th  
https://www.kaggle.com/c/instacart-market-basket-analysis/discussion/38126  
- 12th　　
https://www.kaggle.com/c/instacart-market-basket-analysis/discussion/38110　　