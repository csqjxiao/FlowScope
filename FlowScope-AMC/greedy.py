from __future__ import division
import numpy as np
from scipy import sparse

from MinTree import MinTree


# return a sparse matrix from source data
def listToSparseMatrix(edgesSource, edgesDest, edge_value ):
    m = max(edgesSource) + 1
    n = max(edgesDest) + 1
    degree = edge_value

    M = sparse.coo_matrix(  ( degree , (edgesSource, edgesDest) ),   shape=(m, n))
    return M.astype('float')




# reads edges from file and returns sparse matrix, each row in file should be in form of:  source node, end node, edge weight
def readData(filepath ):
    file1 = filepath + 'fs1.csv'
    file2 = filepath + 'fs2.csv'


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

    row_length = max( edgesSource1 ) +1
    mid_length1 = max( edgesDest1 ) +1
    mid_length2 = max( edgesSource2 ) +1
    mid_length = max( mid_length1 , mid_length2 )
    col_length = max( edgesDest2 ) +1

    tran1 = sparse.coo_matrix((value_list1, (edgesSource1, edgesDest1 )), shape=(row_length, mid_length ))
    tran2 = sparse.coo_matrix((value_list2, (edgesSource2, edgesDest2 )), shape=(mid_length, col_length ))

    return tran1, tran2
# delete a block in the matrix
def del_block(M, rowSet ,colSet):
    M = M.tolil()

    (rs, cs) = M.nonzero()
    for i in range(len(rs)):
        if rs[i] in rowSet or cs[i] in colSet:
            M[rs[i], cs[i]] = 0
    return M.tolil()

# detect a series of dense block in the graph
def detectMultiple(M1, M2, numToDetect  ):

    Mcur1 = M1.copy().tolil()
    Mcur2 = M2.copy().tolil()
    res = []

    for i in range(numToDetect):

        ((rowSet, midSet, colSet), score) = fastGreedyDecreasing(Mcur1, Mcur2    )

        res.append(     [  (rowSet, midSet,  colSet ) , score  ]    )

        Mcur1 = del_block(Mcur1, rowSet, midSet)
        Mcur2 = del_block(Mcur2, midSet, colSet)

    return res


# delete the node nearly greedy
def fastGreedyDecreasing(M1, M2    ):
    alpha = 4
    print 'start  greedy '
    (row_length, mid_length) = M1.shape
    (mid_length, col_length) = M2.shape

    M1 = M1.tolil()
    M2 = M2.tolil()

    M_tran_1 = M1.transpose().tolil()
    M_tran_2 = M2.transpose().tolil()

    rowSet = set(range(0, row_length))
    midSet = set(range(0, mid_length))
    colSet = set(range(0, col_length))

    bestAveScore = float('-inf')

    rowDeltas = np.squeeze(M1.sum(axis=1).A)
    midDeltas1 = np.squeeze(M1.sum(axis=0).A)
    midDeltas2 = np.squeeze(M2.sum(axis=1).A)
    midDeltas = midDeltas1 + midDeltas2

    colDeltas = np.squeeze(M2.sum(axis=0).A)
    mid_min = []
    mid_max  = []
    for (m1, m2) in zip(midDeltas1, midDeltas2):
        temp = min(m1, m2)
        temp2 = max( m1, m2)
        mid_min.append(temp)
        mid_max.append(  temp2 )

    mid_min = np.array( mid_min )
    mid_max = np.array( mid_max )
    new_mid_priority = (1 +alpha ) * mid_min - alpha * mid_max
    new_mid_tree =  MinTree ( new_mid_priority )

    rowTree = MinTree(rowDeltas)
    midTree1 = MinTree(midDeltas1)
    midTree2 = MinTree(midDeltas2)
    midTree = MinTree(midDeltas)
    colTree = MinTree(colDeltas)

    numDeleted = 0
    deleted = []
    bestNumDeleted = 0

    row_sum = 0
    col_sum = 0
    mid_sum1 = 0
    mid_sum2 = 0

    temp1 = 0
    temp2 = 0

    curScore1 = sum(mid_min)
    curScore2 = sum(abs(midDeltas1 - midDeltas2))

    curAveScore1 = curScore1 / (len(rowSet) + len(midSet) + len(colSet))
    curAveScore2 = curScore2 / (len(rowSet) + len(midSet) + len(colSet))

    print 'initial score', curScore1, curScore2 , curAveScore1, curAveScore2
    while rowSet and colSet and midSet:  # repeat deleting until one node set in null

        (nextRow, rowDelt) = rowTree.getMin()  #  node of min weight in row
        (nextCol, colDelt) = colTree.getMin()  #  node of min weight in col
        (nextmid, midDelt) = new_mid_tree.getMin()  # node of min weight in mid
        row_weight = rowDelt * (1 + alpha)
        col_weight = colDelt * (1 + alpha)
        mid_weight = midDelt

        min_weight = min(row_weight , col_weight )
        min_weight = min(min_weight , mid_weight )
        if min_weight == row_weight:
            row_sum += rowDelt
            for j in M1.rows[nextRow]:  # update  the  weight of connected nodes
                new_mid_value = midTree1.changeVal(j, -M1[nextRow, j])

                if new_mid_value == float('inf'):
                    continue
                temp_mid2 = midTree2.index_of(j)
                mid_min_value = min( new_mid_value ,temp_mid2 )
                mid_delta_value = abs(new_mid_value - temp_mid2)
                new_mid_value = mid_min_value  - alpha * mid_delta_value
                new_mid_tree.setter( j , new_mid_value )

                if (new_mid_value) < mid_min[j]:  # if new_mid_value  of node j < min_mid_value  of node j
                    curScore1 -= (mid_min[j] - new_mid_value)
                    mid_min[j] = new_mid_value

                curScore2 = curScore2 - abs(midDeltas1[j] - midDeltas2[j])
                curScore2 = curScore2 + abs(new_mid_value - midDeltas2[j])
                midDeltas1[j] = new_mid_value

            rowSet -= {nextRow}
            rowTree.changeVal(nextRow, float('inf'))
            deleted.append((0, nextRow))

        elif min_weight == col_weight:

            col_sum += colDelt
            for i in M_tran_2.rows[nextCol]:
                new_mid_value = midTree2.changeVal(i, -M_tran_2[nextCol, i])
                if new_mid_value == float('inf'):
                    continue

                temp_mid1 = midTree1.index_of(i)
                mid_min_value = min( new_mid_value ,temp_mid1 )
                mid_delta_value = abs(new_mid_value - temp_mid1)
                new_mid_value = mid_min_value  - alpha * mid_delta_value
                new_mid_tree.setter( i , new_mid_value )

                if (new_mid_value) < mid_min[i]:
                    curScore1 -= (mid_min[i] - new_mid_value)
                    mid_min[i] = new_mid_value

                curScore2 = curScore2 - abs(midDeltas1[i] - midDeltas2[i])
                curScore2 = curScore2 + abs(new_mid_value - midDeltas1[i])
                midDeltas2[i] = new_mid_value

            colSet -= {nextCol}
            colTree.changeVal(nextCol, float('inf'))
            deleted.append((1, nextCol))
        elif min_weight == mid_weight:

            curScore1 -= mid_min[nextmid]
            curScore2 -= abs(midDeltas1[nextmid] - midDeltas2[nextmid])

            mid_min[nextmid] = 0
            midDeltas1[nextmid] = 0
            midDeltas2[nextmid] = 0

            mid_sum1 += midTree1.index_of(nextmid)
            mid_sum2 += midTree2.index_of(nextmid)

            for j in M2.rows[nextmid]:
                colTree.changeVal(j, -M2[nextmid, j])

            for j in M_tran_1.rows[nextmid]:
                rowTree.changeVal(j, -M_tran_1[nextmid, j])

            midSet -= {nextmid}
            midTree.changeVal(nextmid, float('inf'))
            midTree1.changeVal(nextmid, float('inf'))
            midTree2.changeVal(nextmid, float('inf'))
            new_mid_tree.changeVal(nextmid, float('inf'))
            deleted.append((2, nextmid))

        numDeleted += 1
        if (len(rowSet) + len(midSet) + len(colSet)) > 0:
            curAveScore1 = curScore1 / (len(rowSet) + len(midSet) + len(colSet))
        else:
            curAveScore1 = 0
        if (len(rowSet) + len(midSet) + len(colSet))> 0:
            curAveScore2 = curScore2 / (len(rowSet) + len(midSet) + len(colSet))
        else:
            curAveScore2 = 0

        curAveScore = curAveScore1 - alpha * curAveScore2

        if curAveScore >= bestAveScore:
            bestNumDeleted = numDeleted
            bestAveScore = curAveScore
            temp1 = curAveScore1
            temp2 = curAveScore2

    print 'best delete number : ', bestNumDeleted
    print 'nodes number remaining', len(rowSet), len(midSet), len(colSet)
    print  'matrix mass remaining:  ', curScore1, curScore2
    print 'min value of the tree :  ', '  row   ', rowTree.getMin(), ' mid  ', midTree1.getMin(), midTree2.getMin(),  new_mid_tree.getMin(),  '  col ', colTree.getMin()


    print 'best score : ', bestAveScore, temp1, temp2
    finalRowSet = set(range(row_length))
    finalMidSet = set(range(mid_length))
    finalColSet = set(range(col_length))

    for i in range(bestNumDeleted):
        if deleted[i][0] == 0:
            finalRowSet.remove(deleted[i][1])
        elif deleted[i][0] == 1:
            finalColSet.remove(deleted[i][1])
        elif deleted[i][0] == 2:
            finalMidSet.remove(deleted[i][1])

    return (finalRowSet, finalMidSet, finalColSet), bestAveScore


