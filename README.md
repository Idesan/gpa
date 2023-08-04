# Generative Perturbation Analysis for Probabilistic Black-Box Anomaly Attribution

Tsuyoshi Ide (井手 剛), 
tide@us.ibm.com, IBM Thomas J. Watson Research Center.

August 3, 2023

This repository provides a reference implementation of **GPA (generative perturbation analysis)** by the authors, based on a paper presented at KDD 2023:

> Tsuyoshi Id&#233;, Naoki Abe, ``*Generative Perturbation Analysis for Probabilistic Black-Box Anomaly Attribution*,'' Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2023, August 6-10, 2023, Long Beach, California, USA), pp.TBD ([paper](https://ide-research.net/papers/2023_KDD_Ide.pdf), [slides](https://ide-research.net/papers/2023_KDD_Ide_GPA_presentation.pdf), [poster](https://ide-research.net/papers/2023_KDD_Ide_poster.pdf)). 

> @inproceedings{Ide23KDD,
>  title={Generative Perturbation Analysis for Probabilistic Black-Box Anomaly Attribution},
>  author={Tsuyoshi Id\'{e} and Naoki Abe},
>  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 23)},
>  pages={TBD},
>  year={2023}
>}

As the title of the paper suggests, GPA is for *probabilistic* anomaly attribution. The **[demo notebook](GPA_Introduction.ipynb)** provides a readable introduction to the algorithm and an endo-to-end demo using a publicly available dataset. Use `gpa_map(X, y, model)` for the expected attribution score and `gpa_dist(delta_MAP,X,y,model)` for its distribution. 

In addition to the proposed GPA algorithm, I have implemented most of the existing black- or white-box (non-probabilistic) attribution methods for comparison purposes:

1. `lib.gpa_map_gaussian()`: Likelihood compensation (LC) [Ide et al. AAAI 21]
1. `util.LIME_deviation()`: LIME [Ribeiro et al. KDD 16]
1. `util.IG_vec()`: Integrated gradient (IG) [Sundararajan et al. ICML 20]
1. `util.EIG_vec()`: Expected integrated gradient (EIG) [Deng et al. AAAI 21]
1. `util.SV()`: Shapley values (SV) [Strumbelj & Kononenko KAIS 14]
1. Z-score: $Z_i = (x_i^t - m_i)/\sigma_i$


---
[**日本語版はこちら**](GPA_Introduction_JPN.ipynb)
