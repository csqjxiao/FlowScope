from __future__ import division
import math
import numpy as np
import random

from scipy import sparse
from sklearn.utils import shuffle
from MinTree import MinTree


np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=160)



def listToSparseMatrix(edgesSource, edgesDest, edge_value ):
    m = max(edgesSource) + 1
    n = max(edgesDest) + 1
    degree = edge_value

    M = sparse.coo_matrix(  ( degree , (edgesSource, edgesDest) ),   shape=(m, n))
    return M.astype('float')



def readData(filepath ):
    
    
    file1 = filepath + 'fs1.csv'
    file2 = filepath + 'fs2.csv'
    file3 = filepath + 'fs3.csv'

    edgesSource1 = []
    edgesDest1 = []
    value_list1 = []
    with open(file1) as f:
        for line in f:
            toks = line.split(',')
            edgesSource1.append(int(toks[0]))
            edgesDest1.append(int(toks[1]))
            value_list1.append(float(toks[2].strip()))

    edgesSource2 = []
    edgesDest2 = []
    value_list2 = []
    with open(file2) as f:
        for line in f:
            toks = line.split(',')
            edgesSource2.append(int(toks[0]))
            edgesDest2.append(int(toks[1]))
            value_list2.append(float(toks[2].strip()))

    edgesSource3 = []
    edgesDest3 = []
    value_list3 = []
    with open(file3) as f:
        for line in f:
            toks = line.split(',')
            edgesSource3.append(int(toks[0]))
            edgesDest3.append(int(toks[1]))
            value_list3.append(float(toks[2].strip()))


    row_length = max( edgesSource1 ) +1
    mid_length1 = max( edgesDest1 ) +1
    mid_length2 = max( edgesSource2 ) +1
    mid_inter_length1 = max( mid_length1 , mid_length2 )

    mid_length1 = max( edgesDest2) +1
    mid_length2 = max( edgesSource3 ) +1
    mid_inter_length2 = max( mid_length1 , mid_length2 )

    col_length = max( edgesDest3 ) +1

    tran1 = sparse.coo_matrix((value_list1, (edgesSource1, edgesDest1 )), shape=(row_length, mid_inter_length1 ))
    tran2 = sparse.coo_matrix((value_list2, (edgesSource2, edgesDest2 )), shape=(mid_inter_length1, mid_inter_length2 ))
    tran3 = sparse.coo_matrix((value_list3, (edgesSource3, edgesDest3 )), shape=(mid_inter_length2, col_length ))

    return tran1, tran2, tran3


def del_row_col(M, rowSet ,colSet):
    M = M.tolil()

    (rs, cs) = M.nonzero()
    for i in range(len(rs)):
        if rs[i] in rowSet or cs[i] in colSet:
            M[rs[i], cs[i]] = 0
    return M.tolil()


def detectMultiple(M1, M2,M3, numToDetect  ):

    Mcur1 = M1.copy().tolil()
    Mcur2 = M2.copy().tolil()
    Mcur3 = M3.copy().tolil()
    res = []

    for i in range(numToDetect):

        ((rowSet, midSet1, midSet2, colSet), score) = fastGreedyDecreasing( Mcur1, Mcur2 ,Mcur3  )
        res.append(     [  (rowSet, midSet1, midSet2, colSet ) , score  ]    )

        Mcur1 = del_row_col(Mcur1, rowSet, midSet1)                   
        Mcur2 = del_row_col(Mcur1, midSet1, midSet2)                   
        Mcur2 = del_row_col(Mcur2, midSet2, colSet)                   

    return res


def fastGreedyDecreasing(M1, M2, M3 ):
    alpha = 4
    print 'start  greedy '
    (row_length, mid_length1) = M1.shape
    (mid_length1, mid_length2) = M2.shape
    (mid_length2, col_length) = M3.shape

    M1 = M1.tolil()  
    M2 = M2.tolil()  
    M3 = M3.tolil()  

    M_tran_1 = M1.transpose().tolil()  
    M_tran_2 = M2.transpose().tolil()  
    M_tran_3 = M3.transpose().tolil()  

    rowSet = set(range(0, row_length))
    midSet1 = set(range(0, mid_length1))
    midSet2 = set(range(0, mid_length2))
    colSet = set(range(0, col_length))

    

    curScore1 = M1.sum()
    curScore2 = M2.sum()

    
    bestAveScore = -10000000000  
    

    rowDeltas = np.squeeze(M1.sum(axis=1).A)  

    midDeltas1 = np.squeeze(M1.sum(axis=0).A)  
    midDeltas2 = np.squeeze(M2.sum(axis=1).A)  
    

    midDeltas3 = np.squeeze(M2.sum(axis=0).A)  
    midDeltas4 = np.squeeze(M3.sum(axis=1).A)  
    

    
    

    colDeltas = np.squeeze(M3.sum(axis=0).A)  
    mid_min1 = []
    mid_max1 = []
    for (m1, m2) in zip(midDeltas1, midDeltas2):
        temp = min(m1, m2)
        temp2 = max( m1, m2 )
        mid_min1.append( temp )
        mid_max1.append( temp2 )


    mid_min1 = np.array( mid_min1 )
    mid_max1 = np.array( mid_max1 )
    new_mid_priority1 = (1 +alpha ) * mid_min1 - alpha * mid_max1
    new_mid_tree1 =  MinTree ( new_mid_priority1 )

    mid_min2 = []
    mid_max2 = []

    for (m1, m2) in zip(midDeltas3, midDeltas4):
        temp = min(m1, m2)
        temp2 = max( m1, m2 )
        mid_min2.append(temp)
        mid_max2.append( temp2 )

    mid_min2 = np.array( mid_min2 )
    mid_max2 = np.array( mid_max2 )
    new_mid_priority2 = (1 +alpha ) * mid_min2 - alpha * mid_max2
    new_mid_tree2 =  MinTree ( new_mid_priority2 )



    rowTree = MinTree(rowDeltas)
    midTree1 = MinTree(midDeltas1)
    midTree2 = MinTree(midDeltas2)

    midTree3 = MinTree(midDeltas3)
    midTree4 = MinTree(midDeltas4)


    
    
    

    colTree = MinTree(colDeltas)

    numDeleted = 0
    deleted = []
    bestNumDeleted = 0
    

    
    

    curScore1_1 = sum(mid_min1)
    curScore1_2 = sum(mid_min2)

    curScore2_1 = sum(abs(midDeltas1 - midDeltas2))
    curScore2_2 = sum(abs(midDeltas3 - midDeltas4))
    

    while rowSet and colSet and midSet1 and midSet2:  
        
        (nextRow, rowDelt) = rowTree.getMin()  
        (nextCol, colDelt) = colTree.getMin()  

        (nextmid1, midDelt1) = new_mid_tree1.getMin()  
        (nextmid2, midDelt2) = new_mid_tree2.getMin()  

        row_weight = rowDelt * (1 + alpha)
        col_weight = colDelt * (1 + alpha)
        mid_weight1 = midDelt1
        mid_weight2 = midDelt2


        min_weight = min (row_weight, col_weight)
        min_weight = min(min_weight, mid_weight1)
        min_weight = min(min_weight, mid_weight2)
        

        
        
        
        
        
        
        
        
        
        if min_weight == row_weight:
            
                      
            
            

            
            for j in M1.rows[nextRow]:  

                new_mid_value = midTree1.changeVal(j, -M1[nextRow, j])
                if new_mid_value == float('inf'):
                    continue
                temp_mid2 = midTree2.index_of(j)

                if (new_mid_value) < mid_min1[j]:  
                    curScore1_1 -= (mid_min1[j] - new_mid_value)
                    mid_min1[j] = new_mid_value

                mid_min_value = min( new_mid_value ,temp_mid2 )
                mid_delta_value = abs(new_mid_value - temp_mid2)
                new_mid_value = mid_min_value  - alpha * mid_delta_value
                new_mid_tree1.setter( j , new_mid_value )



                curScore2_1 = curScore2_1 - abs(midDeltas1[j] - midDeltas2[j])
                curScore2_1 = curScore2_1 + abs(new_mid_value - midDeltas2[j])
                midDeltas1[j] = new_mid_value


            rowSet -= {nextRow}
            rowTree.changeVal(nextRow, float('inf'))  
            deleted.append((0, nextRow))

        elif min_weight == col_weight:
            
            
            
            

            
            for i in M_tran_3.rows[nextCol]:

                new_mid_value = midTree4.changeVal(i, -M_tran_3[nextCol, i])
                if new_mid_value== float('inf'):
                    continue
                
                

                temp_mid3 = midTree3.index_of(i)
                mid_min_value = min( new_mid_value ,temp_mid3 )
                mid_delta_value = abs(new_mid_value - temp_mid3 )
                new_mid_value = mid_min_value  - alpha * mid_delta_value
                new_mid_tree2.setter( i , new_mid_value )


                if (new_mid_value) < mid_min2[i]:  
                    curScore1_2 -= (mid_min2[i] - new_mid_value)
                    mid_min2[i] = new_mid_value

                curScore2_2 = curScore2_2 - abs(midDeltas3[i] - midDeltas4[i])
                curScore2_2 = curScore2_2 + abs(new_mid_value - midDeltas3[i])
                midDeltas4[i] = new_mid_value
                

            colSet -= {nextCol}
            colTree.changeVal(nextCol, float('inf'))  
            deleted.append((1, nextCol))
        elif min_weight == mid_weight1:
            
            
            
            
                
            
            
            curScore1_1 -= mid_min1[nextmid1]
            curScore2_1 -= abs(  midDeltas1[nextmid1] - midDeltas2[nextmid1]  )

            mid_min1[nextmid1] = float('inf')
            midDeltas1[nextmid1] = float('inf')
            midDeltas2[nextmid1] = float('inf')

            
            
            

            for j in M2.rows[nextmid1]:  
                
                new_mid_value = midTree3.changeVal(j, -M2[nextmid1, j])
                if new_mid_value == float('inf'):
                    continue

                temp_mid4 = midTree4.index_of(j)
                mid_min_value = min( new_mid_value ,temp_mid4 )
                mid_delta_value = abs(new_mid_value - temp_mid4)
                new_mid_value = mid_min_value  - alpha * mid_delta_value
                new_mid_tree2.setter( j , new_mid_value )

                if (new_mid_value) < mid_min2[j]:  
                    curScore1_2 -= (mid_min2[j] - new_mid_value)
                    mid_min2[j] = new_mid_value
                curScore2_2 -= abs( midDeltas3[j] - midDeltas4[j] )
                curScore2_2 += abs( new_mid_value - midDeltas4[j] )
                midDeltas3[j] = new_mid_value

                
                


            
            for j in M_tran_1.rows[nextmid1]:  
                rowTree.changeVal(j, -M_tran_1[nextmid1, j])

            midSet1 -= {nextmid1}
            midTree1.changeVal(nextmid1, float('inf'))  
            midTree2.changeVal(nextmid1, float('inf'))  
            new_mid_tree1.changeVal(nextmid1, float('inf'))  
            deleted.append((2, nextmid1))

        elif min_weight == mid_weight2:
            
            
            
            
            
                
            
            
            curScore1_2 -= mid_min2[nextmid2]
            curScore2_2 -= abs(midDeltas3[nextmid2] - midDeltas4[nextmid2])

            mid_min2[nextmid2] = float('inf')
            midDeltas3[nextmid2] = float('inf')
            midDeltas4[nextmid2] = float('inf')

            
            
            
            
            for j in M3.rows[nextmid2]:  
                
                colTree.changeVal(j, -M3[nextmid2, j]  )
            
            
            
            
            for j in M_tran_2.rows[nextmid2]:  
                
                new_mid_value = midTree2.changeVal(j, -M_tran_2[nextmid2, j])
                if  new_mid_value  == float( 'inf' ):
                    continue
                temp_mid1 = midTree1.index_of(j)
                mid_min_value = min( new_mid_value ,temp_mid1 )
                mid_delta_value = abs(new_mid_value - temp_mid1)
                new_mid_value = mid_min_value  - alpha * mid_delta_value
                new_mid_tree1.setter( j , new_mid_value )

                if (new_mid_value) < mid_min1[j]:  
                    curScore1_1 -= (mid_min1[j] - new_mid_value)
                    mid_min1[j] = new_mid_value

                curScore2_1 -= abs( midDeltas2[j] - midDeltas1[j] )
                curScore2_1 += abs( new_mid_value - midDeltas1[j] )
                midDeltas2[j] = new_mid_value
                
                

            
            midSet2 -= {nextmid2}
            midTree3.changeVal(nextmid2, float('inf'))  
            midTree4.changeVal(nextmid2, float('inf'))  
            new_mid_tree2.changeVal(nextmid2, float('inf'))  
            deleted.append((3, nextmid2) )

        
        
        
        numDeleted += 1

        curAveScore1_1 = curScore1_1 / (len(rowSet) + len(midSet1) + len(midSet2))
        curAveScore2_1 = curScore2_1 / (len(rowSet) + len(midSet1) + len(midSet2))
        curAveScore1_2 = curScore1_2 / (len(midSet1) + len(midSet2) + len(colSet))
        curAveScore2_2 = curScore2_2 / (len(midSet1) + len(midSet2) + len(colSet))

        curAveScore1 = curAveScore1_1 + curAveScore1_2
        curAveScore = curAveScore1 - alpha * curAveScore2_1 - alpha * curAveScore2_2
        
        
        if curAveScore >= bestAveScore:  
            bestNumDeleted = numDeleted
            bestAveScore = curAveScore
            temp1_1 = curAveScore1_1
            temp1_2 = curAveScore1_2
            temp2_1 = curAveScore2_1
            temp2_2 = curAveScore2_2

    print 'best number deleted : ', bestNumDeleted
    
    print 'nodes number remain in rowSet, mid1Set, mid2Set, colSet', len(rowSet), len(midSet1),  len(midSet2), len(colSet)
    print 'min value of each tree :  ', rowTree.getMin(), midTree1.getMin(),midTree2.getMin(), midTree3.getMin(), midTree4.getMin(),colTree.getMin()
    print 'best score: ', bestAveScore

    
    finalRowSet = set(range(row_length))
    finalMidSet1 = set(range(mid_length1))
    finalMidSet2 = set(range(mid_length2))
    finalColSet = set(range(col_length))

    for i in range(bestNumDeleted):  
        if deleted[i][0] == 0:
            finalRowSet.remove(deleted[i][1])
        elif deleted[i][0] == 1:
            finalColSet.remove(deleted[i][1])
        elif deleted[i][0] == 2:
            finalMidSet1.remove(deleted[i][1])
        elif deleted[i][0] == 3:
            finalMidSet2.remove(deleted[i][1])
    # print  'detected row, mid1, mid2, col: ',
    # print finalRowSet, finalMidSet1,  finalMidSet2, finalColSet

    return ((finalRowSet, finalMidSet1,  finalMidSet2, finalColSet), bestAveScore)

