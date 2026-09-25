# Exact UGVT split used by CVAC-Net

`train_sequences_165.txt` and `test_sequences_45.txt` are the authoritative sorted membership lists for the 210 paired UGVT sequences used in the article. A name denotes a synchronized aerial/ground pair, not a single-view sequence. The lists are disjoint and contain 165 and 45 names, respectively.

The historical splitting script recorded Python random seed 1, but obtained its initial names from unsorted filesystem enumeration. `reconstruct_candidate_seeds.py` supplies a portable check: sort the 210 released names, shuffle with `random.Random(seed)`, then take the first 165 as training and the remaining 45 as test. Among the checked candidate seeds 0, 1 and 42, only seed 1 reproduces the released membership. Because the historical enumeration order was not retained, **the lists rather than the seed are the exact split definition**.

Run the check in this directory:

```bash
python reconstruct_candidate_seeds.py
```

The script writes `candidate_seed_reconstruction.json` and checks counts, overlap, and membership. It does not download or modify the original dataset. The split seed is distinct from model-training seeds.

For evaluation, initialize each view's tracker from its ground-truth box in the first frame and evaluate later frames under one-pass evaluation. Report success rate (area under the IoU-threshold success curve), precision at 20-pixel center error, and normalized precision at 0.20 normalized center error. Average sequences equally within each physical view, then average the aerial and ground views equally.
