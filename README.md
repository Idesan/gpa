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

It also provides the implementation of major existing black- or white-box attribution methods:

1. Likelihood compensation (LC) [Ide et al. AAAI 21]
1. LIME [Ribeiro et al. KDD 16]
1. Integrated gradient (IG) [Sundararajan et al. ICML 20]
1. Expected integrated gradient (EIG) [Deng et al. AAAI 21]
1. Shapley values (SV) [Strumbelj & Kononenko KAIS 14]
1. Z-score: $Z_i = (x_i^t - m_i)/\sigma_i$

For the detail, see the [demo notebook](GPA_Introduction.ipynb). Enjoy!

