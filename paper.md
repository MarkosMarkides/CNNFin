# CNNFin: Testing Image Representations for Financial Prediction

## Abstract

This paper studies whether numerical financial time-series data can become more predictive when represented as images and modeled with a convolutional neural network. The experiment uses BTCUSDT as the prediction target and compares an image-based EfficientNet-B0 model against numerical baselines trained on the same underlying market window. The preliminary results suggest that the image model improves over raw numerical baselines, but the conclusion must be stated carefully because the final robustness checks and corrected baseline tables are still being verified.

## Introduction

Financial machine learning models usually receive market data as numerical features: prices, returns, indicators, and volumes. However, traders and researchers often reason visually through charts, heatmaps, and patterns over time. This raises a simple question: if the same numerical data is converted into structured images, can a vision model extract useful patterns that numerical models miss?

CNNFin is designed to test that question in a controlled way. The goal is not to prove that Bitcoin can be predicted reliably, or that the strategy is profitable after fees. The goal is narrower: to compare different representations of the same information and measure whether an image representation improves classification performance.

## Research Question

The main research question is:

> Can representing financial time-series data as images improve directional prediction compared with models trained directly on numerical inputs?

The contribution of this project is twofold. First, it tests whether image representations increase predictability. Second, it tests whether the specific four-panel image design used in this project is useful for financial classification.

## Literature Review

Financial prediction is a difficult setting for machine learning because the signal is weak, unstable, and easily overstated. Gu, Kelly, and Xiu (2020) show that machine learning can improve return prediction, but they also stress the role of regularization, nonlinear interactions, and strict out-of-sample testing. Kelly and Xiu (2023) make a similar point in their broader survey: financial machine learning is useful, but it is not a shortcut around noisy data or poor experimental design. This paper follows that view. The goal is not to claim that Bitcoin is reliably predictable, but to test whether the same market information becomes more useful when represented differently.

The idea that visual market patterns can be studied systematically is not new. Lo, Mamaysky, and Wang (2000) showed that technical chart patterns can be defined algorithmically and tested statistically. Their work matters here because it turns chart reading from a subjective practice into a measurable representation problem. More recently, Jiang, Kelly, and Xiu (2023) provide one of the strongest precedents for image-based financial prediction. They use price chart images and machine learning image analysis to identify return-predictive patterns that differ from standard momentum and reversal signals. Their result is important for CNNFin because it supports the central premise that visual representations can contain useful predictive structure. It also limits the novelty claim: this paper does not invent financial image prediction.

A second relevant literature converts time series into images before classification. Wang and Oates (2015) introduced Gramian Angular Fields and Markov Transition Fields as ways to encode time series for computer vision models. This directly motivates the GAF panel in CNNFin. In finance, Barra et al. (2020) apply time-series-to-image encoding to financial forecasting, while Chen and Tsai (2020) use candlestick images and GAF-based CNNs for candlestick pattern classification. These papers show that GAF and candlestick image methods are already established. Therefore, the contribution of CNNFin is not the use of GAF itself, but the use of GAF as one part of a combined market image.

There is also direct prior work on converting financial data into image-like inputs for CNNs. Sezer and Ozbayoglu (2018) propose a CNN trading model built from technical-indicator image representations, and their later bar-chart image work further supports the idea that visual encodings can be used for financial classification. Jin and Kwon (2021) add an important caution: chart-image design choices themselves affect prediction performance. This matters because CNNFin is not only testing a CNN. It is testing a representation. If the image model performs better, the result may come from the structure imposed by the image design rather than from the CNN alone.

CNNFin also relates to work using convolutional models on structured market data. Zhang, Zohren, and Roberts (2019) show in DeepLOB that convolutional filters can exploit the spatial structure of limit order book data, although their setting is different from chart-image prediction. This supports the broader idea that market data can have useful local structure when arranged in a suitable matrix. It does not, however, prove that a four-panel chart image is optimal.

The labeling and evaluation choices also need grounding. CNNFin uses a fixed-horizon average future return rather than a trade-entry label. The future average close is compared with the current close, and train-only quantile thresholds convert that return into down, neutral, and up classes. This keeps the target aligned with the research question: predictive classification rather than trading simulation. The evaluation uses macro-F1 because each class should matter independently. Opitz (2022) discusses why macro metrics are often used when class-level performance matters more than aggregate accuracy.

Finally, the CNN backbone should be treated as an engineering choice, not a contribution. EfficientNet was introduced by Tan and Le (2019) as an efficient CNN family with strong transfer-learning performance. Yosinski et al. (2014) show that transferred CNN features can help even across different tasks, but that transferability decreases when the source and target domains are distant. This is the correct way to frame ImageNet pretraining in CNNFin: useful initialization, not evidence that natural-image features are inherently financial.

Taken together, the literature suggests a narrow but meaningful position. Financial image prediction, candlestick images, GAF, and CNNs already exist. CNNFin contributes by testing a specific crypto-oriented four-panel image representation against numerical models trained on the same sample IDs and source window. The paper is therefore best framed as an empirical representation study: it asks whether the way numerical market data is arranged changes what a model can learn.

## Data and Prediction Task

The experiment uses Binance one-hour candles from 2021 through 2025. BTCUSDT is the prediction target. Additional context is provided by ADA, BNB, ETH, LINK, LTC, SOL, TRX, XLM, and XRP against USDT.

The data is split chronologically:

| Split | Years |
|---|---|
| Train | 2021-2023 |
| Validation | 2024 |
| Test | 2025 |

Each sample uses the previous 30 one-hour candles as input. The label is built from the following 12 one-hour closes. The future average close is compared with the current close using a log return:

```text
R_t = log(mean(Close_{t+1:t+12}) / Close_t)
```

The lower and upper thresholds are fitted on the training set only using the 33rd and 66th percentiles of `R_t`. The classes are:

| Class | Meaning |
|---:|---|
| 0 | down |
| 1 | neutral |
| 2 | up |

The validation and test labels use the same thresholds fitted from the training set. This keeps the target definition fixed after training data is observed.

## Image Representation

For each sample, the numerical market window is converted into a four-panel image:

1. altcoin-vs-BTC return divergence
2. BTC candlestick chart
3. BTC Gramian Angular Field
4. technical indicator heatmap

The CNN model sees only this image. The numerical baselines are trained on the same sample IDs and the same source window, so the comparison is based on representation rather than different data.

## Models

The proposed image model is EfficientNet-B0, pretrained on ImageNet and fine-tuned for three-class classification.

The numerical baselines are:

| Model | Input format |
|---|---|
| Logistic Regression | flattened numerical window |
| XGBoost | flattened numerical window |
| MLP | flattened numerical window |
| LSTM | numerical sequence |

The main evaluation metric is macro-F1 on the 2025 test set. Weighted-F1 and accuracy are also reported, but macro-F1 remains the primary metric because the task is multi-class and imbalanced.

## Preliminary Results

The first completed CNN run achieved:

| Model | Test macro-F1 | Test accuracy | Test weighted-F1 |
|---|---:|---:|---:|
| EfficientNet-B0 CNN | 0.36069 | 0.50070 | 0.50051 |

The original raw numerical baseline results were lower:

| Model | Test macro-F1 |
|---|---:|
| MLP | 0.28758 |
| Logistic Regression | 0.26806 |
| LSTM | 0.25574 |
| XGBoost | 0.23136 |

Under this first comparison, the CNN improves over the best raw numerical baseline by about 25 percent in relative macro-F1.

These results are from the V1 triple-barrier setup. They should be treated as historical preliminary results until the V2 average-future-return labels are rebuilt and the models are rerun.

TODO: Insert final corrected regularized-baseline table.

TODO: Insert random-distribution baseline table.

TODO: Insert final confidence intervals and significance comparison.

## Discussion

The preliminary result suggests that image representations can make financial windows easier for a model to use. One possible explanation is that the image construction naturally emphasizes shapes, local structure, relative movement, and visual contrast. Raw numerical vectors may be harder for standard models to interpret, especially across different market regimes.

At the same time, the result should not be overstated. If improved numerical preprocessing closes the gap, then the conclusion is not simply that CNNs are better. A more careful conclusion would be that representation quality matters. Images may help because they transform raw market data into a more structured and scale-stable form.

## Limitations

This project has several limitations. First, it evaluates prediction quality, not trading profitability. Fees, slippage, and execution constraints are not included. Second, balanced train thresholds make the train task well formed, but validation and test distributions may still shift. Third, the results are based on one target asset and one market period. Finally, the final robustness checks still need to be completed.

TODO: Add figure references for example images and confusion matrices.

## Conclusion

CNNFin tests whether numerical financial data becomes more predictive when represented as images. The preliminary evidence shows that an EfficientNet-B0 image model outperforms raw numerical baselines on test macro-F1. This supports the idea that image representations can improve predictability, at least compared with naive numerical inputs.

The final paper should present this finding carefully. The strongest claim is not that Bitcoin is easily predictable, but that the way market data is represented has a measurable effect on model performance. The next step is to verify the corrected baseline results, include a random baseline, and report the final confidence intervals.

## References

Barra, S., Carta, S. M., Corriga, A., Podda, A. S., & Recupero, D. R. (2020). Deep learning and time series-to-image encoding for financial forecasting. *IEEE/CAA Journal of Automatica Sinica, 7*(3), 683-693.

Chen, J.-H., & Tsai, Y.-C. (2020). Encoding candlesticks as images for pattern classification using convolutional neural networks. *Financial Innovation, 6*, 26.

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies, 33*(5), 2223-2273.

Jiang, J., Kelly, B., & Xiu, D. (2023). (Re-)Imag(in)ing price trends. *Journal of Finance*.

Jin, G., & Kwon, O. (2021). Impact of chart image characteristics on stock price prediction with a convolutional neural network. *PLOS ONE, 16*(6), e0253121.

Kelly, B. T., & Xiu, D. (2023). Financial machine learning. *NBER Working Paper No. 31502*.

Lo, A. W., Mamaysky, H., & Wang, J. (2000). Foundations of technical analysis: Computational algorithms, statistical inference, and empirical implementation. *Journal of Finance, 55*(4), 1705-1765.

Opitz, J. (2022). From bias and prevalence to macro F1, kappa, and MCC: A structured overview of metrics for multi-class evaluation.

Sezer, O. B., & Ozbayoglu, A. M. (2018). Algorithmic financial trading with deep convolutional neural networks: Time series to image conversion approach. *Applied Soft Computing, 70*, 525-538.

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of the 36th International Conference on Machine Learning*.

Wang, Z., & Oates, T. (2015). Imaging time-series to improve classification and imputation. *Proceedings of IJCAI 2015*.

Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? *Advances in Neural Information Processing Systems*.

Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep convolutional neural networks for limit order books. *IEEE Transactions on Signal Processing, 67*(11), 3001-3012.
