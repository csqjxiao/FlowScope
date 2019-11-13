import time
start_time = time.time()
from greedy import *
import  pandas as pd


def getScore(from_acc, to_acc, all_ab_set, middle_set=[] ):
    
    temp = from_acc.append(to_acc)
    if len(middle_set) > 0:
        temp = temp.append(middle_set)
    detected_ab = list(set(temp))

    detect_length = len(detected_ab)
    real_detect = set(detected_ab) & set(all_ab_set)  
    ab_length = len(real_detect)

    precision = ab_length / float(detect_length)
    recall = ab_length / float(len(all_ab_set))

    f1_score = 0 if (precision + recall == 0) else 2 * precision * recall / (precision + recall)

    return [ab_length, precision, recall, f1_score]

def getScore_money (  from_acc , mid1_acc, mid2_acc, to_acc, ab_from_acc_list, ab_to_acc_list, all_ab_money , tran_matrix1, tran_matrix2 , tran_matrix3 ):

    from_acc_list  = list( from_acc )
    mid1_acc_list  = list( mid1_acc )
    mid2_acc_list  = list( mid2_acc )
    to_acc_list = list( to_acc )

    block_mass1 = get_block_mass( from_acc_list, mid1_acc_list, tran_matrix1 )
    real_ab_mass1 = get_block_mass_traverse(from_acc_list, mid1_acc_list,  ab_from_acc_list[0], ab_to_acc_list[0], tran_matrix1  )

    block_mass2 = get_block_mass( mid1_acc_list, mid2_acc_list, tran_matrix2 )
    real_ab_mass2 = get_block_mass_traverse(mid1_acc_list, mid2_acc_list,  ab_from_acc_list[1], ab_to_acc_list[1], tran_matrix2  )

    block_mass3 = get_block_mass( mid2_acc_list, to_acc_list, tran_matrix3 )
    real_ab_mass3 = get_block_mass_traverse(mid2_acc_list, to_acc_list,  ab_from_acc_list[2], ab_to_acc_list[2], tran_matrix3  )

    precision = ( real_ab_mass1  + real_ab_mass2 + real_ab_mass3 ) / float( block_mass1 +  block_mass2 + block_mass3  )
    recall = ( real_ab_mass1  + real_ab_mass2 + real_ab_mass3 )  / float( all_ab_money )

    print 'the mass of detected block: ', block_mass1, block_mass2, block_mass3
    print 'the mass of true abnormal: ',real_ab_mass1, real_ab_mass2, real_ab_mass3

    block_mass = block_mass1 +  block_mass2+  block_mass3
    real_ab_mass = real_ab_mass1 +  real_ab_mass2 +  real_ab_mass3

    f1_score = 0 if (precision + recall == 0) else 2 * precision * recall / (precision + recall)
    return [ block_mass, real_ab_mass, all_ab_money,  precision, recall, f1_score]


def get_block_mass( from_acc, to_acc, matrix  ) :
    from_acc = list(  from_acc )
    to_acc = list( to_acc )

    degree_matrix = matrix
    degree_matrix_row = degree_matrix.tocsr()[from_acc, :]
    degree_matrix_column = degree_matrix_row.tocsc()[:, to_acc]
    block_mass = degree_matrix_column.sum()
    return block_mass

def get_block_mass_traverse( from_acc_list, to_acc_list, ab_from_acc_list, ab_to_acc_list, matrix  ) :
    
    from_acc = list(from_acc_list)
    to_acc = list(to_acc_list)
    block_mass = 0


    temp_block_mass = 0
    temp_from = set(from_acc) & set(ab_from_acc_list )
    temp_to = set(to_acc) & set(ab_to_acc_list )
    print 'detected  from : ', len(temp_from), temp_from
    print 'detected to : ', len(temp_to), temp_to

    for row_idx in temp_from:  
        for col_idx in temp_to:  #
            temp_block_mass += matrix[row_idx, col_idx]
    print 'detected block  mass : ', temp_block_mass
    block_mass += temp_block_mass

    return block_mass



def dense_flow(tran1 , tran2, tran3,  block_num    ):

    tran1 = tran1.tolil()
    tran2 = tran2.tolil()
    tran3 = tran3.tolil()


    M_backup1 = tran1.copy()
    M_backup2 = tran2.copy()
    M_backup3 = tran3.copy()

    M_copy1 = tran1.tolil()
    M_copy2 = tran2.tolil()
    M_copy3 = tran3.tolil()

    lwRes = detectMultiple(M_copy1, M_copy2, M_copy3, block_num   )
    return  lwRes
