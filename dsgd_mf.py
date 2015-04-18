#!/usr/bin/env python
"""This script implements DSGD-MF using spark"""


from pyspark import SparkContext, SparkConf
import sys
import numpy as np
import math
from operator import itemgetter


FILEPATH = sys.argv[6]
NUM_WORKERS = int(sys.argv[2])
NUM_ITERATIONS = int(sys.argv[3])/ NUM_WORKERS
NUM_FACTORS = int(sys.argv[1])
BETA = float(sys.argv[4])
LAMBDA = float(sys.argv[5])
W_PATH = sys.argv[7]
H_PATH = sys.argv[8]

def createratingvector(key, value, row_slice, col_slice):
    """find out the rowindex and colindex from (0... highest user/movieID) using
    the (movieid, userid). And use (rowIndex, colIndex) to find out the stratum
    it locates."""
    userid = key[0]
    movieid = key[1]
    wi = userid - 1
    hj = movieid - 1
    rowblock = wi / row_slice
    colblock = hj / col_slice
    if rowblock >= NUM_WORKERS:
        rowblock = NUM_WORKERS - 1
    if colblock >= NUM_WORKERS:
        colblock = NUM_WORKERS - 1
    if rowblock == colblock:
        return ((0, rowblock), (colblock, wi, hj, value))
    elif rowblock > colblock:
        return ((colblock + NUM_WORKERS - rowblock, rowblock), (colblock, wi, hj, value))
    else:
        return ((colblock - rowblock, rowblock), (colblock, wi, hj, value))



def updatematrix(rowblock, tuple_val, w_mat, h_mat, numrows, numcols, gamma, totalups, wn, vn):
    """retrieve the corresponding rows in w and h according to rowblock and colblock
     assigned to the stratum of ratings"""
    tuple_val = list(tuple_val)
    colblock = tuple_val[0][0]
    (rowlow, rowhigh) = getslicefromlowtohigh(rowblock, NUM_WORKERS, numrows)
    (collow, colhigh) = getslicefromlowtohigh(colblock, NUM_WORKERS, numcols)
    # create a new copy of the subset of w and h
    w_block = w_mat[rowlow:rowhigh, ]
    h_block = h_mat[collow:colhigh, ]
    numups = 0
    for record in tuple_val:
        epsilon = math.pow(gamma + totalups + numups, -BETA)
        row_no = record[1] - rowlow
        col_no = record[2] - collow
        predict_rating = np.dot(w_block[row_no,], h_block[col_no,])
        ni = wn[(record[1], rowblock, colblock)]
        nj = vn[(record[2], rowblock, colblock)]
        q = w_block[row_no,] - epsilon * (
            (-2) * (record[3] - predict_rating) * h_block[col_no,]
            + 2 * LAMBDA * w_block[row_no,]* 1.0 / ni)
        h_block[col_no,] = h_block[col_no,] - epsilon * (
            (-2) * (record[3] - predict_rating) * w_block[row_no,]
            + 2 * LAMBDA * h_block[col_no,] * 1.0 / nj)
        w_block[row_no,] = q
        numups += 1
    return ((rowblock, colblock, numups), (w_block, h_block))


def getslicefromlowtohigh(rowblock, totalblock, numrows):
    """using the block no to get the ranges of rows from lowestIndex
     to highestIndex"""
    row_slice = numrows / totalblock
    if rowblock < totalblock - 1:
        return (rowblock * row_slice, (rowblock + 1) * row_slice)
    else:
        return (rowblock * row_slice, numrows)



def readratingfromtestfile(line):
    """read the rating file,each rating is converted into
    a tuple with keys are (userid, movieid), value is the rating"""
    line = line.split(",")
    return ((int(line[0]), int(line[1])), int(line[2]))




if __name__ == "__main__":
    conf = SparkConf().setAppName("PythonSGD")
    sc = SparkContext()
    # read file
    rdd = sc.textFile(FILEPATH).map(readratingfromtestfile)
    # get unique and sorted movie_ids and user_ids
    movie_ids = rdd.map(lambda (key, value): key[1]).reduceByKey(lambda a, b: 1)\
                .sortByKey().keys()
    user_ids = rdd.map(lambda (key, value): (key[0], 1)).reduceByKey(lambda a, b: 1)\
                .sortByKey().keys()
    NUM_MOVIES = movie_ids.count()
    NUM_USERS = user_ids.count()
    MOVIE_SLICE = NUM_MOVIES / NUM_WORKERS
    USER_SLICE = NUM_USERS/ NUM_WORKERS

    # compute the stratum no for each (userid,movieid) pair.
    rddIntoBlock = rdd.map(
        lambda (key, value): createratingvector(key, value, USER_SLICE,
                                                MOVIE_SLICE, NUM_WORKERS))
    # count number of ratings for each user/ moview at each (rowblock, colblock) strata
    w_n = rddIntoBlock.map(lambda (key, value): ((value[1], key[1], value[0]), 1)).\
                           keys().countByValue()
    h_n = rddIntoBlock.map(lambda (key, value): ((value[2], key[1], value[0]), 1)).\
                           keys().countByValue()
    # initialize the w and h as numpy dense array
    w = np.random.rand(NUM_USERS, NUM_FACTORS)
    h = np.random.rand(NUM_MOVIES, NUM_FACTORS)
    w_br = sc.broadcast(w)
    h_br = sc.broadcast(h)
    w_n_br = sc.broadcast(w_n)
    h_n_br = sc.broadcast(h_n)
    totalupdates = 0

    for i in xrange(NUM_ITERATIONS):
        for j in xrange(NUM_WORKERS):
            # only select the blocks that belong to this stratum
            block_rdd = rddIntoBlock.filter(lambda (key, value): key[0] == j).map(
                lambda (key, value): (key[1], value)).groupByKey().partitionBy(NUM_WORKERS)

            # sequentially update the w, h for each block in this stratum
            block_mat = block_rdd.map(lambda (key, value):
                                      updatematrix(key, value, w_br.value, h_br.value, NUM_WORKERS,
                                                   NUM_USERS, NUM_MOVIES, 100, totalupdates, BETA,
                                                   LAMBDA, w_n_br.value, h_n_br.value)).collect()
            # remove the w_br and h_br in memory
            w_br.unpersist()
            h_br.unpersist()
            # aggregate all the number of updates made in the current iteration
            numUpdates = sum(x[0][2] for x in block_mat)
            totalupdates += numUpdates
            # sort the rowBlock according to the rowBlock_no
            row_list = sorted([(x[0][0], x[1][0]) for x in block_mat], key=itemgetter(0))
            row = [x[1] for x in row_list]
            # sort the colBlock according to the colBlock_no
            col_list = sorted([(x[0][1], x[1][1]) for x in block_mat], key=itemgetter(0))
            col = [x[1] for x in col_list]
            # combine the rowblocks into one np array
            w = np.vstack(row)
            h = np.vstack(col)
            # broadcast the w, h to all workers again
            w_br = sc.broadcast(w)
            h_br = sc.broadcast(h)
    np.savetxt(W_PATH, w, delimiter=',', fmt="%f")
    np.savetxt(H_PATH, h.T, delimiter=',', fmt="%f")




