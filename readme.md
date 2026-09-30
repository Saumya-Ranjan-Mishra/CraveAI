## Metrics Used for Offline Evaluation

We evaluate three rankings: BM25, dense retrieval, and Reciprocal Rank Fusion (RRF). Each ranking is compared against two separate relevance signals:

- **Related items** (`response_item_ids`): the dataset's known related-item labels.
- **User clicks** (`clicked_item_ids`): items with recorded user clicks.

These signals answer different questions, so we report them separately. A click is positive behavioral evidence, but an unclicked item is not necessarily irrelevant; it may not have been shown or chosen.

### Metrics

All metrics use a cutoff `K`, meaning they evaluate only the first K results. The cutoff should reflect the product surface being evaluated. For example, use `K=10` to evaluate the first page of results. A larger cutoff such as `K=100` is useful for measuring deeper retrieval, but does not describe first-page quality.

- **Hit Rate@K**: The fraction of labeled queries for which at least one relevant item appears in the top K. A query contributes 1 if there is a hit and 0 otherwise. For example, Hit Rate@10 of 0.76 means at least one labeled relevant item appeared in the first 10 results for 76% of labeled queries. It does not measure how many relevant items appeared.

- **MRR@K (Mean Reciprocal Rank)**: The average reciprocal rank of the first relevant result. If the first relevant result is at rank `r`, that query contributes `1/r`; if no relevant result appears in the top K, it contributes 0. MRR rewards placing the first relevant result near the top, but ignores other relevant results after it.

- **Recall@K**: The fraction of all labeled relevant items retrieved in the top K:

  `Recall@K = relevant items in top K / all labeled relevant items`

  Queries with many relevant items can have low Recall@K even when many top results are valid. For example, retrieving 10 relevant items from a set of 400 gives Recall@10 of 0.025. We therefore interpret recall in light of the number and completeness of the labels.

- **Precision@K**: The fraction of the first K returned results that are labeled relevant:

  `Precision@K = relevant results in top K / number of results returned up to K`

  For example, if 7 of the first 10 results are relevant, Precision@10 is `0.70`. Unlike Hit Rate@K, it reflects how many results are relevant, not just whether there is at least one. Unlike Recall@K, it does not divide by the full set of relevant items.

  Precision@K is useful for estimating the relevance density of the results users see. It depends on the relevance labels being sufficiently complete: a genuinely suitable food item missing from the labels will be counted as non-relevant.

### What are the metrices we use in this project & How We Interpret the Metrics

For food search, many items may equally satisfy a query. Unless the labels distinguish quality or preference, we do not assume one qualifying item is inherently more relevant than another.

- **Precision@k** indicates how many relevant have appreaded in the top k.
- **Hit Rate@k** indicates how often the first page contains at least one labeled match.
- **MRR@k** indicates how early the first labeled match appears.
- **Recall@K** is a retrieval-depth diagnostic: it indicates how much of the labeled candidate set was retrieved, and is more useful at larger K or when the relevant set is reasonably bounded.

## Results with only item description embeddings (test set with top 100 queries)

**@K = 10**
![alt text](image.png)

**@K = 100**
![alt text](image-1.png)

## Observations

- MRR at k = 100 and 10 remain almost same, so the search results are consistent, irrespective of the value of k.
- MRR onlt at BM25 search is 60% while only dense search is 32.54%. However, with RRF, the MRR is ~65%.
- MRR for based on user past clicked history is too low (3%).
- recall at k is also too low, this is becuase all the item queries has more than 300 - 400 items. For example: when user searches for Kathal Biryani, there are 435 items in the overall picture items that matches that item name, Hence for this case the recall_at_k is very low.
- Since ranking based on the items relevancy is not possible in this case hence NDCG (Normalized Discounted Cummulative Gain), is not the correct matrics to use here.
- We are focusing on the hit rate, which is how many items are among the top k are in the relevat items (for related_items) and items user click on as the metrics to check how the search results are.
- The hit_rate at k = 100 & 10 shows a very good result with 69% in BM25 and 72 in dense search, and with RRF in place that number goes upto 84% while testing against related_items metrics, how ever for the user clicks, the RRF search search results struggles at 38%.

## Conclusions after first Iteration

- We need to improve the dense search MRR results. Looks like the MRR for dense search is very low cauing the fusion rank to suffer.
