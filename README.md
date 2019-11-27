·

# FlowScope

Implementation code of the algorithm described in FlowScope: Spotting Money Laundering Based on Graphs. and thanks for the contribution of the coworkers , i.e. , Shenghua Liu and Zifeng Li.

**FlowScope**: Spotting Money Laundering Based on Graphs

**FlowScope** is developped for ML detection based on multipartite graph,  which models the high-volume *flows* of funds passing through chains of bank accounts, and  detects the complete flow of money from source to destination.

## Abstract 

Given a graph of the money transfers between accounts of a bank, how can we detect  money laundering? Money laundering refers to criminals using the bank's services to move massive amounts of illegal money to untraceable destination accounts, in order to inject their illegal money into the legitimate financial system. Existing graph fraud detection approaches focus on dense subgraph detection, without considering the fact that money laundering involves high-volume *flows* of funds through chains of bank accounts, thereby decreasing their detection accuracy.  Instead, we propose to model the transactions using a multipartite graph, and  detect the complete flow of money from source to destination using a scalable algorithm, FlowScope. Theoretical analysis shows that FlowScope provides guarantees in terms of the amount of money that fraudsters can transfer  without being detected. FlowScope outperforms state-of-the-art baselines in accurately detecting the accounts involved in money laundering, in both injected and real-world data settings.

## Run in command line

Code in  FlowScope-AMC:

```
        designed for detecting money laundering with one middle layer M
```

Code in  FlowScope-AM1M2C:

```
        designed for detecting money laundering with two middle layers M1 and M2.  
```

for usage:

```
        python2  FS.py  input_dir output_dir  block_num 
```
       
parameter specifications：

```
        input_dir:  the path to input data; 
        output_dir: the path to output result;
        block_num:  numbers of suspicious transactions you want to detect;
```

  input data: decribe the transaction between two  adjacent layer, which should be named as fs1.csv, fs2.csv ... fsn.csv . 
              

```
        fs1.csv: describes the transactions between source layer and the first middle layer. 
        ...
        fsi.csv: describes the transactions between current i-th layer and the next layer. 
        ...
        fsn.csv: describes the transactions between last middle layer and the destination layer.  
```

  ouput result: the most suspicous accouts in each layer. 

```
        rown.csv: the source  account of the n-th suspicious transactions；
        coln.csv: the destination  account of the n-th suspicious transactions；
        midk_n. csv: the middle account in k-th middle layer of the n-th suspicious transactions；
```



## The results  





## Cite:

```
@inproceedings{li2020flowscope,
  author    = {Xiangfeng Li,  Shenghua Liu,   
        Zifeng Li,   Xiaotian Han, Chuan Shi ,   
        Bryan Hooi,  He Huang,  Xueqi Cheng},
  title     = {FlowScope: Spotting Money Laundering Based on  Graphs },
  booktitle = {Proceedings of the 2017 {ACM} on Conference on Information and Knowledge
               Management, {CIKM} 2017, Singapore, November 06 - 10, 2017},
  pages     = {1539--1548},
  year      = {2020},
  crossref  = {DBLP:conf/cikm/2017},
  url       = {http://doi.acm.org/10.1145/3132847.3133018},
  doi       = {10.1145/3132847.3133018},
  timestamp = {Tue, 07 Nov 2017 16:24:37 +0100},
  biburl    = {http://dblp.org/rec/bib/conf/cikm/LiuHF17},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
@article{liu2018contrast,
  title={A Contrast Metric for Fraud Detection in Rich Graphs},
  author={Liu, Shenghua and Hooi, Bryan and Faloutsos, Christos},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2018},
  publisher={IEEE}
}
```





