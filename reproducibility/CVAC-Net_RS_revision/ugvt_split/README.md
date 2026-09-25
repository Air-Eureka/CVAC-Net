# UGVT split

`train_sequences_165.txt` and `test_sequences_45.txt` are the exact, disjoint paired-sequence membership lists used in the CVAC-Net manuscript. Each name identifies a synchronized aerial/ground pair. The lists, rather than a random seed alone, define this split.

The historical split procedure recorded Python random seed 1, but the original filesystem enumeration order was not retained. The manuscript reports each physical view correctly. Older internal result-folder abbreviations should not be used to infer physical view identity.
