from run_greedy import *
import sys
import pandas as pd

if __name__ == '__main__':

    tran1, tran2 , tran3 = readData( sys.argv[1] )  
    print "finished reading data: shape = ", tran1.shape, tran2.shape, tran3.shape
    print 'mass of each matrix:', long(  tran1.sum()  ), long ( tran2.sum()  ), long ( tran3.sum()  )
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    block_num = int(sys.argv[3])


    result = dense_flow(tran1, tran2, tran3, block_num   )
    print 'the  information of detected dense flow '
    for i in range(block_num ):
        row = result[i][0][0]
        mid1 = result[i][0][1]
        mid2 = result[i][0][2]
        col = result[i][0][3]
        best_score = result[i][1]

        print 'block number:' , i+1
        print '\t the length of row , mid1, mid2, and col :', len(row ), len( mid1 ),len( mid2 ), len( col )
        print '\t row :', row, '\n\t mid1: ',mid1, '\n\t mid2: ',mid2, '\n\t col: ', col
        print '\t best score: ', best_score
        row_csv = pd.DataFrame( {'row':row } )
        mid1_csv = pd.DataFrame( {'mid1': mid1 } )
        mid2_csv = pd.DataFrame( {'mid2':mid2 } )
        col_csv = pd.DataFrame( {'col':col } )

        row_csv.to_csv(  output_dir + '/row_'  + str( i +1) + '. csv', index= None  )
        mid1_csv.to_csv(  output_dir + '/mid1_'  + str( i +1) + '. csv', index= None  )
        mid2_csv.to_csv(  output_dir + '/mid2_'  + str( i +1) + '. csv', index= None  )
        col_csv.to_csv(  output_dir + '/col_'  + str( i +1) + '. csv', index= None  )

