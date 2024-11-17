def split_seq(seq, num_pieces):
    start = 0
    for i in range(num_pieces):
        stop = start + len(seq[i:: num_pieces])
        yield seq[start:stop]
        start = stop
