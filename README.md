# FlowScope
Implementation code of the algorithm described in FlowScope: Spotting Money Laundering Based on Graphs. and thanks for the contribution of the coworkers , i.e. , Shenghua Liu and Zifeng Li.

Code in  FlowScope-AMC:
            designed for detecting money laundering with one middle layer M
Code in  FlowScope-AM1M2C:
            designed for detecting money laundering with one two layers M1 and M2.  

for usage:
            python2  FS.py  input_dir output_dir  block_num 
       
       
parameter specifications：
  input_dir:  the path to input data; 
  output_dir: the path to output result;
  block_num:  numbers of suspicious transactions you want to detect;
  
  input data: decribe the transaction between two  adjacent layer, which should be named as fs1.csv, fs2.csv ... fsn.csv . 
              
              fs1.csv describes the transactions be source layer and the first middle layer.  
              fsn.csv describes the transactions be source layer and the first middle layer.  
               
  ouput data: the most suspicous accouts in each layer. 

              rown.csv: the source  account of the n-th suspicious transactions；
              coln.csv: the destination  account of the n-th suspicious transactions；
              midk_n. csv: the middle account in k-th middle layer of the n-th suspicious transactions；
             
  

  

