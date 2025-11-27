import sys


def codon_aa(c):
    """Convert codon into amino acid.
    
    :param c: Input codon.
    :type c: string
    :return: The amino acid.
    :rtype: string
    """
    if len(c) != 3:
        print(f"Error of converting codon into amino acid: "
              f"the input codon length is not 3.\n"
              f"The input was '{c}', "
              f"will be marked as 'x' in the output.")
        # sys.exit()
    if c in ['GTT', 'GTC', 'GTA', 'GTG', 'GUU', 'GUC', 'GUA', 'GUG']:
        aa = 'V' # Val
    elif c in ['GCT', 'GCC', 'GCA', 'GCG', 'GCU']:
        aa = 'A' # Ala
    elif c in ['GAT', 'GAC', 'GAU']:
        aa = 'D' # Asp
    elif c in ['GAA', 'GAG']:
        aa = 'E' # Glu
    elif c in ['GGT', 'GGC', 'GGA', 'GGG', 'GGU']:
        aa = 'G' # Gly
    elif c in ['TTT', 'TTC', 'UUU', 'UUC']:
        aa = 'F' # Phe
    elif c in ['TTA', 'TTG', 'UUA', 'UUG', 'CTT', 'CTC', 'CTA', 'CTG', 'CUU', 'CUC', 'CUA', 'CUG']:
        aa = 'L' # Leu
    elif c in ['TCT', 'TCC', 'TCA', 'TCG', 'UCU', 'UCC', 'UCA', 'UCG', 'AGT', 'AGC', 'AGU']:
        aa = 'S' # Ser
    elif c in ['TAT', 'TAC', 'UAU', 'UAC']:
        aa = 'Y' # Tyr
    elif c in ['TGT', 'TGC', 'UGU', 'UGC']:
        aa = 'C' # Cys
    elif c in ['TGG', 'UGG']:
        aa = 'W' # Trp
    elif c in ['CCT', 'CCC', 'CCA', 'CCG', 'CCU']:
        aa = 'P' # Pro
    elif c in ['CAT', 'CAC', 'CAU']:
        aa = 'H' # His
    elif c in ['CAA', 'CAG']:
        aa = 'Q' # Gln
    elif c in ['CGT', 'CGC', 'CGA', 'CGG', 'CGU', 'AGA', 'AGG']:
        aa = 'R' # Arg
    elif c in ['ATT', 'ATC', 'ATA', 'AUU', 'AUC', 'AUA']:
        aa = 'I' # Ile
    elif c in ['ATG', 'AUG']:
        aa = 'M' # Met (Starting codon)
    elif c in ['ACT', 'ACC', 'ACA', 'ACG', 'ACU']:
        aa = 'T' # Thr
    elif c in ['AAT', 'AAC', 'AAU']:
        aa = 'N' # Asn
    elif c in ['AAA', 'AAG']:
        aa = 'K' # Lys
    elif c in ['TAA', 'TAG', 'UAA', 'UAG', 'TGA', 'UGA']:
        aa = '*' # STOP
    else:
        aa = 'U' # nonsense
    return aa


def translation(seq, seq_shift=0):
    """Convert the input sequence into protein sequence.
    (Works on both DNA or RNA sequences!)

    :param seq: Input sequence.
    :type seq: string
    :param seq_shift: Sequence shift of translation. [0,1,2,-1] defaults to 0.
    :type seq_shift: int, optional
    :return: The protein sequence.
    :rtype: string
    """
    if seq_shift not in [0,1,2,-1]:
        print(f"Error of translation seq_shift setting, "
              f"'{seq_shift}' was used. (only accept 0, 1, 2 or -1)")
        sys.exit()
    count = len(seq) // 3 # codon count.
    if seq_shift != 0:
        count -= 1
        if seq_shift == -1:
            seq_shift = 2
    aa = ''
    loc1 = 0 + seq_shift
    loc2 = 3 + seq_shift
    for i in range(count):
        codon = seq[loc1:loc2]
        # print(codon)
        aa += codon_aa(codon)
        loc1 += 3
        loc2 += 3
    return aa


# Reverse translation from amino acids to DNA
import random


def reverse_translation(ptotein_seq):

    dna_seq = ""
    for aa in ptotein_seq:
        if aa == 'A':
            dna_seq += random.choice(['GCT', 'GCC', 'GCA', 'GCG'])
        elif aa == 'C':
            dna_seq += random.choice(['TGT', 'TGC'])
        elif aa == 'D':
            dna_seq += random.choice(['GAT', 'GAC'])
        elif aa == 'E':
            dna_seq += random.choice(['GAA', 'GAG'])
        elif aa == 'F':
            dna_seq += random.choice(['TTT', 'TTC'])
        elif aa == 'G':
            dna_seq += random.choice(['GGT', 'GGC', 'GGA', 'GGG'])
        elif aa == 'H':
            dna_seq += random.choice(['CAT', 'CAC'])
        elif aa == 'I':
            dna_seq += random.choice(['ATT', 'ATC', 'ATA'])
        elif aa == 'K':
            dna_seq += random.choice(['AAA', 'AAG'])
        elif aa == 'L':
            dna_seq += random.choice(['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'])
        elif aa == 'M':
            dna_seq += 'ATG'
        elif aa == 'N':
            dna_seq += random.choice(['AAT', 'AAC'])
        elif aa == 'P':
            dna_seq += random.choice(['CCT', 'CCC', 'CCA', 'CCG'])
        elif aa == 'Q':
            dna_seq += random.choice(['CAA', 'CAG'])
        elif aa == 'R':
            dna_seq += random.choice(['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'])
        elif aa == 'S':
            dna_seq += random.choice(['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'])
        elif aa == 'T':
            dna_seq += random.choice(['ACT', 'ACC', 'ACA', 'ACG'])
        elif aa == 'V':
            dna_seq += random.choice(['GTT', 'GTC', 'GTA', 'GTG'])
        elif aa == 'W':
            dna_seq += 'TGG'
        elif aa == 'Y':
            dna_seq += random.choice(['TAT', 'TAC'])
        elif aa == '*':
            dna_seq += random.choice(['TAA', 'TAG', 'TGA'])
        else:
            dna_seq += '???'  # Represents an unknown or invalid amino acid

    return dna_seq