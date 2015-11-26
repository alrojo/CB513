# Protein vectors are tirplets.
# read matrix of one_hot encoded amin acids.
# 1) load protein vectors
# 2) Create mapping from triplet to protein vector
# 3) For each amino acid sequence do
#       a) Find all triplets. Ensure that len(triplets) == len(seq) by padding both ends with 'X'
#       b) Map all triplets to protein vector
# 4) return matrix


import gzip
import numpy as np


# read protvec and return tiplet -> protvec dict
def read_protvec(fn):
    protvec = np.genfromtxt(fn, delimiter='\t')
    protvec = protvec[:, 1:].astype('float32')  # remove aa names
    aa_triplet2protvec = {}
    with open(fn, 'r') as f:
        aa_triplets = [l.split('\t')[0] for l in f.readlines()]

    for idx, triplet in enumerate(aa_triplets):
        aa_triplet2protvec[triplet] = protvec[idx]

    return aa_triplet2protvec


def load_protvec_encoding(protein_file, protein_vector_file, protvec_dim = 100):
    # amino acid encodings see
    # http://www.princeton.edu/~jzthree/datasets/ICML2014/dataset_readme.txt
    aas = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q',
           'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X']
    aa2num = {aa: idx for idx, aa in enumerate(aas)}
    num2aa = {idx: aa for idx, aa in enumerate(aas)}

    ## load protein vector
    aatriplet2protvec = read_protvec(protein_vector_file)


    # could just make X_in an ar
    # load gz proteins
    with gzip.open(protein_file, 'rb') as f:
        X_in = np.load(f)
        X_in = np.reshape(X_in, (5534, 700, 57))

    X_ort_onehot = X_in[:, :, :21]   # load orthogonal columns
    mask = X_in[:, :, 30] * -1 + 1

    X_num = X_ort_onehot.argmax(axis=-1) # convert to numeric rep.
    num_seq, seq_len = X_num.shape
    X_out = np.zeros((num_seq, seq_len, protvec_dim), dtype='float32')
    found, not_found = 0, 0
    # iterate over sequences
    for seq_row in range(num_seq):
        this_seq_len = int(np.sum(mask[seq_row]))
        this_seq_unmasked = X_num[seq_row, :this_seq_len]

        # this is ugly and slow but easier to debug i think...
        this_seq_str = [num2aa[i] for i in this_seq_unmasked]

        # pad sequence with two 'X' to have correct number of triplets
        this_seq_str = ['X'] + this_seq_str + ['X']

        # create triplets
        triplets = []
        for i in range(1, this_seq_len+1):
            triplets += ["".join(this_seq_str[i-1:i+2])]
        assert len(triplets) == this_seq_len
        # do a random check....
        # XABCDEF 0= XAB, 1=ABC, 2= BCD etc
        assert triplets[2] == "".join(this_seq_str[2:5])

        # use aatriplet2protvec to look up all triplets
        for seq_pos in range(this_seq_len):
            triplet = triplets[seq_pos]
            # try look up. Otherwise use '<unk>'
            try:
                protvec = aatriplet2protvec[triplet]
                found += 1
            except KeyError:
                protvec = aatriplet2protvec['<unk>']
                not_found += 1
            X_out[seq_row, seq_pos] = protvec

    print "Num Triplet found    :", found
    print "Num Triplet not found:", not_found

    # sanity check
    assert np.sum(mask) == (found + not_found)

    return X_out


if __name__ == '__main__':
    protein_file = 'cullpdb+profile_6133_filtered.npy.gz'
    protein_vector_file = 'protVec_100d_3grams_clean.csv'
    X = load_protvec_encoding(protein_file, protein_vector_file)
    print X.shape
    print X[101, :10, :10]





