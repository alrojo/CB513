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


def seq2triplets(seq):
    # pad sequence with two 'X' to have correct number of triplets
    seq_len = len(seq)
    seq_padded = ['X'] + seq + ['X']
    triplets = []
    for i in range(1, seq_len+1):
        triplets += ["".join(seq_padded[i-1:i+2])]
    return triplets


def load_protvec_encoding(X_in, mask, protein_vector_file, protvec_dim = 100):
    # amino acid encodings see
    # http://www.princeton.edu/~jzthree/datasets/ICML2014/dataset_readme.txt
    aas = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q',
           'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X']
    num2aa = {idx: aa for idx, aa in enumerate(aas)}

    ## load protein vector
    aatriplet2protvec = read_protvec(protein_vector_file)
    X_ort_onehot = X_in[:, :, :21]   # load orthogonal columns


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
        triplets = seq2triplets(this_seq_str)
        #print triplets
        # use aatriplet2protvec to look up all triplets
        for seq_pos in range(this_seq_len):
            triplet = triplets[seq_pos]
            print triplet,
            # try look up. Otherwise use '<unk>'
            try:
                protvec = aatriplet2protvec[triplet]
                #print "Found:", triplet
                found += 1
            except KeyError:
                protvec = aatriplet2protvec['<unk>']
                #print "Not found", triplet
                not_found += 1
            X_out[seq_row, seq_pos] = protvec


    print "Num Triplet found    :", found
    print "Num Triplet not found:", not_found

    # sanity check
    assert np.sum(mask) == (found + not_found)

    return X_out



## Testing --- very little....
def test_seq2triplets():
    seq = list('EHK')
    triplets_true = ['XEH', 'EHK', 'HKX']
    triplets_fun = seq2triplets(seq)
    assert triplets_true == triplets_fun


def test_load_protvec_encoding(protein_vector_file):
    aas = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q',
           'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X']
    aa2num = {aa: idx for idx, aa in enumerate(aas)}
    seq = list('EHK')
    seq_num = [aa2num[aa] for aa in seq]
    num_aa = len(aas)
    X = np.zeros((2, 3, num_aa))
    for row in range(2):
        for col in range(3):
            X[row, col, seq_num[col]] = 1

    mask = np.ones((2,3))
    mask[1,2] = 0  # mask last position in seq2


    X_fun = load_protvec_encoding(X, mask, protein_vector_file)

    ## triplets are for seq EHK are 'XEH', 'EHK', 'HKX'
    ## for the masked sequence we have [XEH, EHX]
    XEH	= "-0.040274	-0.131796	0.001296	0.126828	-0.100334	-0.099206	-0.025372	0.092139	0.078937	0.130191	0.030734	-0.061886	-0.006869	-0.027613	0.03856	-0.011865	0.107084	0.037546	-0.095911	0.046329	-0.072519	0.109772	-0.126555	0.041689	-0.034009	-0.016592	-0.181516	0.088839	0.08461	0.049713	0.088948	-0.01107	-0.012233	-0.139802	0.056793	0.012647	-0.09859	0.080558	-0.039157	0.131668	0.020648	-0.000104	0.052428	-0.085358	-0.097828	-0.146377	0.008134	-0.020106	-0.075406	0.038728	0.064586	-0.057677	0.102028	0.005715	0.013417	-0.008224	-0.044459	-0.077354	0.001212	0.003074	0.057253	0.022256	0.01945	-0.126424	0.061443	-0.038428	0.077202	-0.154751	0.09359	0.102762	0.006394	0.054582	-0.143053	-0.089576	0.00577	-0.177765	0.19629	0.122234	0.1121	0.046496	-0.179245	0.081885	-0.063401	-0.022518	0.128442	-0.057264	-0.010756	-0.054602	-0.015483	0.058533	-0.010063	0.113258	0.001118	0.021575	0.082484	-0.054278	-0.015756	0.116589	-0.031181	0.040217"
    EHK =  "0.020481	0.173989	-0.03861	0.08149	0.075322	-0.017179	0.111902	0.063626	-0.094459	0.119148	0.131103	-0.063715	0.021452	0.090283	0.009241	0.080508	0.105047	-0.143445	0.083745	-0.14124	-0.087696	0.004191	-0.119396	0.06372	0.03968	0.012651	-0.019076	-0.056619	-0.004592	-0.046646	-0.034477	0.001795	-0.051988	0.034241	0.094871	0.047674	0.151395	-0.107013	-0.031548	0.08612	0.05873	0.013363	-0.015174	-0.140363	0.019804	-0.065965	0.058439	-0.013402	0.107643	0.285637	-0.175445	0.079577	0.028941	-0.150677	-0.013795	-0.12161	0.074128	0.0383	0.026246	-0.084032	-0.029617	-0.006227	0.184241	0.022002	0.045034	0.000205	0.034911	0.040973	0.051888	-0.042757	-0.001686	0.093547	-0.073569	0.065407	-0.023812	-0.049088	0.140886	0.154521	-0.039286	-0.160079	-0.023498	-0.043347	-0.151132	-0.086797	0.086302	-0.027177	-0.060353	-0.102455	-0.044422	0.007159	0.03522	0.081977	-0.009477	-0.095076	0.100908	-0.00127	0.13496	0.074342	-0.096401	0.101643"
    unk =	"-0.0365576142793988	0.0150848404067197	-0.0421630539345713	-0.0791904971264365	-0.00188294882847041	-0.0083460720601238	-0.00675786881078695	-0.026734716954023	-0.0630853951149427	0.0684358414014147	0.00526172679045092	-0.0335999732537577	-0.00455342152961977	0.00180430017683467	-0.0237892017020336	0.0107441864500442	0.037585730327144	-0.0336586401414678	-0.0249018978779841	-0.0231389745800177	-0.00361985179045094	-0.0317930816755084	-0.0616961456675507	0.0458140505083996	0.00446027641467728	-0.0255747749778956	-0.0488459209770116	-0.00101535411140585	0.0292818015030946	0.00037209659593285	-0.0142972078912467	-0.0429948863837311	0.0293992107648098	-0.0181502973032715	0.0669065242042438	0.0219559259504864	-0.0448854582228119	0.0181411231211318	-0.00868478647214858	-0.0189086151635721	0.00249231576038905	0.0042003215075155	0.0342378280282936	-0.0268296264367816	-0.0025461195844386	0.00482850353669321	-0.053351735632184	-0.00447327486737398	-0.0186593258178603	0.0342339108090186	-0.0269484575596816	-0.0140385459770115	0.00325713561007947	-0.0576905234305923	0.0371601086427939	-0.0592895155835546	0.000929333775420014	-0.0129883458222812	-0.0153933040450929	-0.0240505654288241	-0.00515014710433246	0.0776796380415559	0.00226443247126438	-0.0300370659814324	0.0745039966843501	-0.0188211025641025	-0.0346046162687886	0.0209826118479222	0.0531284751326261	-0.0271630249778956	-0.0474824709328029	-0.0217271354995579	-0.00839073264809902	-0.015792401856764	-0.004186840959328	-0.0504500591290893	0.0132622290008842	0.000581241158267018	-0.0138131879973475	-0.00837482305481874	-0.0681864021883288	0.0183128757736516	0.032521590959328	0.0195699266136163	0.0274142185013262	-0.0252078950044208	-0.0173655896330681	-0.0145892749778957	0.00343636682139712	0.0144756736295315	0.00890529000884171	0.0285740992484528	-0.0552389189876215	-0.0144206358311229	0.0349314709328028	-0.00874192992926612	-0.0584642891246685	-0.0515562371794871	-0.0408943401856765	0.0936843799734748"
    XEH = np.array(map(float, XEH.split('\t')))
    EHK = np.array(map(float, EHK.split('\t')))
    unk = np.array(map(float, unk.split('\t')))
    seq_true = np.concatenate([[XEH, EHK, unk]])
    X_true = np.concatenate([[seq_true, seq_true]])
    X_true[1,2,:] = 0
    X_true[1,1] = unk


    assert 1e-6>np.sum(X_fun[0,0,:] - XEH)
    assert 1e-6>np.sum(X_fun[0,1,:] - EHK)
    assert 1e-6>np.sum(X_fun[0,2,:] - unk)
    assert 1e-6>np.sum(X_fun[1,0,:] - XEH)
    assert 1e-6>np.sum(X_fun[1,1,:] - unk)
    assert 1e-6>np.sum(X_fun[1,2,:])


if __name__ == '__main__':
   test_seq2triplets()
   test_load_protvec_encoding(protein_vector_file='protVec_100d_3grams_clean.csv')




