from run_greedy import *
import sys
import pandas as pd

if __name__ == '__main__':

    tran1, tran2 = readData( sys.argv[1] )  # read the datasss
    print "finished reading data: shape = ", tran1.shape, tran2.shape
    print 'the mass of each matrix :', long(  tran1.sum()  ), long ( tran2.sum()  )
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    block_num = int(sys.argv[3])
    res = dense_flow(tran1, tran2,  block_num )
    print 'the  information of detected dense flow '
    for i in range( block_num ):
        block_info = res[i]
        row = block_info[0][0]
        mid = block_info[0][1]
        col = block_info[0][2]
        best_score = block_info[1]

        print 'block number:' , i+1
        print '\t the length of row , mid1, mid2, and col :', len(row ), len( mid ), len( col )
        print '\t row :', row, '\n\t mid: ',mid,  '\n\t col: ', col
        print '\t best score: ', best_score
        row_csv = pd.DataFrame(   {'row': row   } )
        mid_csv = pd.DataFrame(   {  'mid': mid   } )
        col_csv = pd.DataFrame(   { 'col':col  } )

        row_csv.to_csv(  output_dir + '/row'  + str( i +1) + '. csv', index= None  )
        mid_csv.to_csv(  output_dir + '/mid'  + str( i +1) + '. csv', index= None  )
        col_csv.to_csv(  output_dir + '/col'  + str( i +1) + '. csv', index= None  )




