#!/usr/bin/env python
# Author: 2021 Mohammad Alanjary
# Wageningen University and Research (WUR)
# Bioinformatics department
#
# License: A copy of the GPLv3 can also be found at: <http://www.gnu.org/licenses/>.

# Generator for modified Condensation - Adenylation domain interface

import argparse, subprocess, os, shutil, tempfile
import numpy as np
import random
import multiprocessing as mp
import time, os, itertools

gapPenalty = 0.05

# ---- Amino acid to E-desciptor conversion ----
# E-descriptor dictionary: DOI 10.1007/s00894-001-0058-5
# KEY = aa 1 letter code. VAL = (E1, E2, E3, E4, E5)
# Interpretations E1 - hydrophobic/hydrophilic, E2 - Side chain length
# E3 - Alpha-helix freq, E4 / E5 = other combinations (partial specific volumes, relative abundance of amino acids, B-strand forming propensity)

ED = {    'A':(0.008, 0.134, -0.475, -0.039, 0.181),  'R':(0.171, -0.361, 0.107, -0.258, -0.364), 'N':(0.255, 0.038, 0.117, 0.118, -0.055),
          'D':(0.303, -0.057, -0.014, 0.225, 0.156),   'C':(-0.132, 0.174, 0.07, 0.565, -0.374), 'Q':(0.149, -0.184, -0.03, 0.035, -0.112),
          'E':(0.221, -0.28, -0.315, 0.157, 0.303),    'G':(0.218, 0.562, -0.024, 0.018, 0.106), 'H':(0.023, -0.177, 0.041, 0.28, -0.021),
          'I':(-0.353, 0.071, -0.088, -0.195, -0.107),  'L':(-0.267, 0.018, -0.265, -0.274, 0.206), 'K':(0.243, -0.339, -0.044, -0.325, -0.027),
          'M':(-0.239, -0.141, -0.155, 0.321, 0.077),  'F':(-0.329, -0.023, 0.072, -0.002, 0.208), 'P':(0.173, 0.286, 0.407, -0.215, 0.384),
          'S':(0.199, 0.238, -0.015, -0.068, -0.196),   'T':(0.068, 0.147, -0.015, -0.132, -0.274), 'W':(-0.296, -0.186, 0.389, 0.083, 0.297),
          'Y':(-0.141, -0.057, 0.425, -0.096, -0.091), 'V':(-0.274, 0.136, -0.187, -0.196, -0.299)}

blosum62 = {
    ('W', 'F'): 1, ('L', 'R'): -2, ('S', 'P'): -1, ('V', 'T'): 0,
    ('Q', 'Q'): 5, ('N', 'A'): -2, ('Z', 'Y'): -2, ('W', 'R'): -3,
    ('Q', 'A'): -1, ('S', 'D'): 0, ('H', 'H'): 8, ('S', 'H'): -1,
    ('H', 'D'): -1, ('L', 'N'): -3, ('W', 'A'): -3, ('Y', 'M'): -1,
    ('G', 'R'): -2, ('Y', 'I'): -1, ('Y', 'E'): -2, ('B', 'Y'): -3,
    ('Y', 'A'): -2, ('V', 'D'): -3, ('B', 'S'): 0, ('Y', 'Y'): 7,
    ('G', 'N'): 0, ('E', 'C'): -4, ('Y', 'Q'): -1, ('Z', 'Z'): 4,
    ('V', 'A'): 0, ('C', 'C'): 9, ('M', 'R'): -1, ('V', 'E'): -2,
    ('T', 'N'): 0, ('P', 'P'): 7, ('V', 'I'): 3, ('V', 'S'): -2,
    ('Z', 'P'): -1, ('V', 'M'): 1, ('T', 'F'): -2, ('V', 'Q'): -2,
    ('K', 'K'): 5, ('P', 'D'): -1, ('I', 'H'): -3, ('I', 'D'): -3,
    ('T', 'R'): -1, ('P', 'L'): -3, ('K', 'G'): -2, ('M', 'N'): -2,
    ('P', 'H'): -2, ('F', 'Q'): -3, ('Z', 'G'): -2, ('X', 'L'): -1,
    ('T', 'M'): -1, ('Z', 'C'): -3, ('X', 'H'): -1, ('D', 'R'): -2,
    ('B', 'W'): -4, ('X', 'D'): -1, ('Z', 'K'): 1, ('F', 'A'): -2,
    ('Z', 'W'): -3, ('F', 'E'): -3, ('D', 'N'): 1, ('B', 'K'): 0,
    ('X', 'X'): -1, ('F', 'I'): 0, ('B', 'G'): -1, ('X', 'T'): 0,
    ('F', 'M'): 0, ('B', 'C'): -3, ('Z', 'I'): -3, ('Z', 'V'): -2,
    ('S', 'S'): 4, ('L', 'Q'): -2, ('W', 'E'): -3, ('Q', 'R'): 1,
    ('N', 'N'): 6, ('W', 'M'): -1, ('Q', 'C'): -3, ('W', 'I'): -3,
    ('S', 'C'): -1, ('L', 'A'): -1, ('S', 'G'): 0, ('L', 'E'): -3,
    ('W', 'Q'): -2, ('H', 'G'): -2, ('S', 'K'): 0, ('Q', 'N'): 0,
    ('N', 'R'): 0, ('H', 'C'): -3, ('Y', 'N'): -2, ('G', 'Q'): -2,
    ('Y', 'F'): 3, ('C', 'A'): 0, ('V', 'L'): 1, ('G', 'E'): -2,
    ('G', 'A'): 0, ('K', 'R'): 2, ('E', 'D'): 2, ('Y', 'R'): -2,
    ('M', 'Q'): 0, ('T', 'I'): -1, ('C', 'D'): -3, ('V', 'F'): -1,
    ('T', 'A'): 0, ('T', 'P'): -1, ('B', 'P'): -2, ('T', 'E'): -1,
    ('V', 'N'): -3, ('P', 'G'): -2, ('M', 'A'): -1, ('K', 'H'): -1,
    ('V', 'R'): -3, ('P', 'C'): -3, ('M', 'E'): -2, ('K', 'L'): -2,
    ('V', 'V'): 4, ('M', 'I'): 1, ('T', 'Q'): -1, ('I', 'G'): -4,
    ('P', 'K'): -1, ('M', 'M'): 5, ('K', 'D'): -1, ('I', 'C'): -1,
    ('Z', 'D'): 1, ('F', 'R'): -3, ('X', 'K'): -1, ('Q', 'D'): 0,
    ('X', 'G'): -1, ('Z', 'L'): -3, ('X', 'C'): -2, ('Z', 'H'): 0,
    ('B', 'L'): -4, ('B', 'H'): 0, ('F', 'F'): 6, ('X', 'W'): -2,
    ('B', 'D'): 4, ('D', 'A'): -2, ('S', 'L'): -2, ('X', 'S'): 0,
    ('F', 'N'): -3, ('S', 'R'): -1, ('W', 'D'): -4, ('V', 'Y'): -1,
    ('W', 'L'): -2, ('H', 'R'): 0, ('W', 'H'): -2, ('H', 'N'): 1,
    ('W', 'T'): -2, ('T', 'T'): 5, ('S', 'F'): -2, ('W', 'P'): -4,
    ('L', 'D'): -4, ('B', 'I'): -3, ('L', 'H'): -3, ('S', 'N'): 1,
    ('B', 'T'): -1, ('L', 'L'): 4, ('Y', 'K'): -2, ('E', 'Q'): 2,
    ('Y', 'G'): -3, ('Z', 'S'): 0, ('Y', 'C'): -2, ('G', 'D'): -1,
    ('B', 'V'): -3, ('E', 'A'): -1, ('Y', 'W'): 2, ('E', 'E'): 5,
    ('Y', 'S'): -2, ('C', 'N'): -3, ('V', 'C'): -1, ('T', 'H'): -2,
    ('P', 'R'): -2, ('V', 'G'): -3, ('T', 'L'): -1, ('V', 'K'): -2,
    ('K', 'Q'): 1, ('R', 'A'): -1, ('I', 'R'): -3, ('T', 'D'): -1,
    ('P', 'F'): -4, ('I', 'N'): -3, ('K', 'I'): -3, ('M', 'D'): -3,
    ('V', 'W'): -3, ('W', 'W'): 11, ('M', 'H'): -2, ('P', 'N'): -2,
    ('K', 'A'): -1, ('M', 'L'): 2, ('K', 'E'): 1, ('Z', 'E'): 4,
    ('X', 'N'): -1, ('Z', 'A'): -1, ('Z', 'M'): -1, ('X', 'F'): -1,
    ('K', 'C'): -3, ('B', 'Q'): 0, ('X', 'B'): -1, ('B', 'M'): -3,
    ('F', 'C'): -2, ('Z', 'Q'): 3, ('X', 'Z'): -1, ('F', 'G'): -3,
    ('B', 'E'): 1, ('X', 'V'): -1, ('F', 'K'): -3, ('B', 'A'): -2,
    ('X', 'R'): -1, ('D', 'D'): 6, ('W', 'G'): -2, ('Z', 'F'): -3,
    ('S', 'Q'): 0, ('W', 'C'): -2, ('W', 'K'): -3, ('H', 'Q'): 0,
    ('L', 'C'): -1, ('W', 'N'): -4, ('S', 'A'): 1, ('L', 'G'): -4,
    ('W', 'S'): -3, ('S', 'E'): 0, ('H', 'E'): 0, ('S', 'I'): -2,
    ('H', 'A'): -2, ('S', 'M'): -1, ('Y', 'L'): -1, ('Y', 'H'): 2,
    ('Y', 'D'): -3, ('E', 'R'): 0, ('X', 'P'): -2, ('G', 'G'): 6,
    ('G', 'C'): -3, ('E', 'N'): 0, ('Y', 'T'): -2, ('Y', 'P'): -3,
    ('T', 'K'): -1, ('A', 'A'): 4, ('P', 'Q'): -1, ('T', 'C'): -1,
    ('V', 'H'): -3, ('T', 'G'): -2, ('I', 'Q'): -3, ('Z', 'T'): -1,
    ('C', 'R'): -3, ('V', 'P'): -2, ('P', 'E'): -1, ('M', 'C'): -1,
    ('K', 'N'): 0, ('I', 'I'): 4, ('P', 'A'): -1, ('M', 'G'): -3,
    ('T', 'S'): 1, ('I', 'E'): -3, ('P', 'M'): -2, ('M', 'K'): -1,
    ('I', 'A'): -1, ('P', 'I'): -3, ('R', 'R'): 5, ('X', 'M'): -1,
    ('L', 'I'): 2, ('X', 'I'): -1, ('Z', 'B'): 1, ('X', 'E'): -1,
    ('Z', 'N'): 0, ('X', 'A'): 0, ('B', 'R'): -1, ('B', 'N'): 3,
    ('F', 'D'): -3, ('X', 'Y'): -1, ('Z', 'R'): 0, ('F', 'H'): -1,
    ('B', 'F'): -3, ('F', 'L'): 0, ('X', 'Q'): -1, ('B', 'B'): 4
}

def aa2EDmat(aaseq, m=None, matched=None):
    EDmat = []
    for i, aa in enumerate(aaseq.upper()):
        #skip sites that are not in matched list
        if m and matched and m[i] not in matched:
            continue
        if aa in ED.keys():
            EDmat.append(ED[aa])
        else:
            print("error: '%s' is not a valid amino acid"%aa)
            EDmat.append([0, 0, 0, 0, 0])
    return np.array(EDmat)

def aadists(seq1,seq2,map1,map2):
    matchedsites = set(map1) & set(map2)
    matchedsites = matchedsites - set([-1])
    seq1ED = aa2EDmat(seq1)
    seq2ED = aa2EDmat(seq2)
    # seq1ED = aa2EDmat(seq1, map1, matchedsites) #use mapped to ignore unaligned positions
    # seq2ED = aa2EDmat(seq2, map2, matchedsites)
    #
    # dists = np.linalg.norm(seq1ED-seq2ED, axis=1)

    # Get E-descriptor euclidean distance for matching ref locations only
    dists = []
    for i in map1:
        if i in map2:
            dists.append(np.linalg.norm(seq1ED[map1.index(i)]-seq2ED[map2.index(i)]))

    #Penalty for mis-matching locations
    # mmsites = len(set(map1)+set(map2))-len(matchedsites)
    # penalty = mmsites*gapPenalty / len(dists)
    # dists = [d + penalty for d in dists]

    return dists

#Generate random varients of amino acid sequences from ClusterMap dictionary
def seqmutator(seq1, seq2=None, map1=None, map2=None, maxsites=0.5, maxseqs=1e6, mode=0):
    aapool = list(ED.keys())
    seqlist = list(seq1)
    seqlen = len(seqlist)
    if maxsites < 1:
        #Use fraction of length if its a ratio
        maxsites = np.round(maxsites * seqlen)
    maxsites = int(maxsites)
    results = [set() for i in range(maxsites)] #Set storing from 1-maxsite sets
    #Generate all possible site location combinations
    allsites = list(itertools.combinations(list(range(0, seqlen)), maxsites))
    filtpos = {map1.index(p): map2.index(p) for p in list(set(map1) & set(map2)) if p > 0} #maped shared sites

    #limits for sampling
    maxcombos = 20 ** maxsites
    if mode == 1:
        maxcombos = 2 ** maxsites
    print("Mode: %s; Maxsites: %s; Mapped: %s; SiteCombinations: %s; Maxseqs/combo: %d; EstSeqCoverage: %.6f%%" %
          (mode, maxsites, len(filtpos.keys()), len(allsites), maxseqs/len(allsites), 100*(maxseqs/len(allsites))/maxcombos))
    for sites in allsites:
        #evenly sample form every site combination with at least 10 iterations
        for iblank in range(int(maxseqs / len(allsites))+10):
            seqlist = list(seq1) #reset to original seq
            if iblank > maxcombos:
                #Stop generating if possibilities are mostly sampled
                break
            ### Mode 0 - totally random replacement
            if mode == 0:
                #Generate random replacements and strore each replacement sequence to get gradient of # sites
                for si, s in enumerate(sites):
                    seqlist[s] = random.choice(aapool)
                    results[si].add(''.join(seqlist))
            ### Mode 1 - Mossaic use mapped residue from seq 2 otherwise use random aa
            if mode == 1:
                #Generate random replacement positions with k=maxsites, strore each replacement sequence to get gradient of parsimony
                for si, s in enumerate(sites):
                    if s in filtpos.keys():
                        seqlist[s] = seq2[filtpos[s]]
                    else:
                        seqlist[s] = random.choice(aapool)
                    results[si].add(''.join(seqlist))
    return results

# ---- Align reference C-A interface and annotate coordinates to reference interfaces -----
def alignCA(infasta, refalign=False, refcoords=False, tempdir=None, scoreonly=False, regions="A2", outpath=None, maxseqs=1e6, maxsites=0.5, mode=0, mcpu=1, hml=100, hmlonly=False):
    """Find interface regions relative to reference for query cluster A, First seq in fasta (upstream seq),
    and cluster B 2nd seq in fasta (used for Adomain swap)"""
    # Note: if Cluster C (3rd seq in fasta is present) this will be used for final sequence optimization
    # Otherwise Cluster C will be auto-generated based on alignment position
    if not(refalign and refcoords):
        scriptdir = os.path.dirname(os.path.realpath(__file__))
        refalign = os.path.join(scriptdir, "refalign.fasta")
        # Locations of interfaces relative to in refalign
        refcoords = {"C1": (215, 227), "C2": (275, 283), "C3": (316, 319), "C4": (352, 356), "C5": (436, 463),
                     "A1": (822, 825), "A2": (844, 856), "A3": (898, 905), "A4": (918, 931), "A5": (957, 964),
                     "A6": (996, 1004), "A7": (1015, 1021)}

    if not outpath:
        outpath = "mutants-%d" % time.time()

    regions = regions.split(",")
    for region in regions:
        if region not in refcoords.keys():
            print("Error invalid region: %s"%region)
            exit(1)

    tmpfolder = tempfile.mkdtemp(dir=tempdir)
    print("Running in: %s"%tmpfolder)
    queryfile = os.path.join(tmpfolder, "query.fasta")
    mapfile = os.path.join(tmpfolder, "query.fasta.map")
    treefile = os.path.join(tmpfolder, "query.tree")

    shutil.copy(infasta, queryfile)
    cwd = os.getcwd()
    # Change working directory for MAFFT alignment output (and corresponding map and tree files)
    os.chdir(tmpfolder)
    try:
        with open("aligned.fasta", "w") as ofil:
            subprocess.call(["mafft", "--quiet", "--localpair", "--maxiterate", "1000", "--mapout", "--treeout", "--add", "query.fasta", refalign],stdout=ofil)
    except:
        print("failed to execute alignment")
    os.chdir(cwd)

    # parse map file and return dictionary of mapped sequences for clusterA/B/C
    with open(mapfile, "r") as fil:
        clusterMap = [None, None, None]
        i = -1
        for line in fil:
            if line.startswith("#"):
                continue #skip comments
            if line.startswith(">"):
                #Log new cluster in map (cluster A, B, C : i=0,1,2)
                i += 1
                clusterMap[i] = {"seq": "", "refind": [], "qind": [], "intlocs": dict.fromkeys(refcoords.keys(), None), "reflocs": dict.fromkeys(refcoords.keys(), None),
                                 "qlocs": dict.fromkeys(refcoords.keys(), None), "intseqs": dict.fromkeys(refcoords.keys(), None)}
                continue
            aaseq,qcol,refcol = line.split(", ")
            clusterMap[i]["qind"].append(int(qcol))
            clusterMap[i]["seq"] += aaseq
            if "-" in refcol:
                clusterMap[i]["refind"].append(-1)
            else:
                clusterMap[i]["refind"].append(int(refcol))

    # Translate reference locations to query locations for cluster C and extract sequences
    for i in range(0, 3):
        for k, loc in refcoords.items():
            # get all overlapping positions
            OLinds = set(range(*loc)) & set(clusterMap[i]["refind"])
            indstrt = clusterMap[i]["refind"].index(min(OLinds))
            indend = clusterMap[i]["refind"].index(max(OLinds))
            subseq = clusterMap[i]["seq"][indstrt:indend]

            clusterMap[i]["intlocs"][k] = (indstrt, indend)
            clusterMap[i]["reflocs"][k] = clusterMap[i]["refind"][indstrt:indend]
            clusterMap[i]["qlocs"][k] = clusterMap[i]["qind"][indstrt:indend]
            clusterMap[i]["intseqs"][k] = subseq

    if scoreonly:
        print("#Cluster C vs Native A-dom regions in Cluster A & Native C-dom regions in Cluster B:")
        print("#Region-label:\tsubSeq-C\tsubSeq-A\tmean-distance\tsum-distance\tmax-distance\tmax-index")
        alldists = [[],[]] # A and C domain distances (mean, sumdist, maxdist, count of distances)
        for k,subseq1 in clusterMap[2]["intseqs"].items():
            #Detect which cluster based on region key (default Cluster A for A domain):
            clustInd = 0
            if k.startswith("C"):
                clustInd = 1

            map1 = clusterMap[2]["reflocs"][k]
            subseq2 = clusterMap[clustInd]["intseqs"][k]
            map2 = clusterMap[clustInd]["reflocs"][k]

            dists = aadists(subseq1, subseq2, map1, map2)
            if dists:
                maxdist = np.max(dists)
                sumdist = np.sum(dists)
                alldists[clustInd].append(sumdist)
                print("%s:\t%s\t%s\t%s\t%s\t%s\t%s" % (k, subseq1, subseq2, np.mean(dists), sumdist, maxdist, dists.index(maxdist)+1))
            else:
                print("%s:\t%s\t%s\tn.a." % (k, subseq1, subseq2))
        print("#C-dom cumulative score: %s" % np.sum(alldists[1]))
        print("#A-dom cumulative score: %s" % np.sum(alldists[0]))
    else:
        hmlrecs = {}
        for region in regions:
            outdir = os.path.join(outpath, "mutants_region-%s" % region)

            if not os.path.exists(outdir):
                os.makedirs(outdir)

            print("Generating mutants for region %s..." % region)
            subseq1 = clusterMap[2]["intseqs"][region]
            map1 = clusterMap[2]["reflocs"][region]

            # Detect which cluster based on region key (default Cluster A for A domain):
            clustInd = 0
            if region.startswith("C"):
                clustInd = 1

            subseq2 = clusterMap[clustInd]["intseqs"][region]
            map2 = clusterMap[clustInd]["reflocs"][region]

            mutants = seqmutator(subseq1, seq2=subseq2, map1=map1, map2=map2, maxseqs=maxseqs, maxsites=maxsites, mode=mode)
            for k, v in enumerate(mutants):
                #parallel calculation of distances if mcpu > 1
                print("Calculating distances for %s mutants with %s sites changed..." % (len(v), k+1))
                dists = []
                if mcpu > 1:
                    pool = mp.Pool(mcpu)
                    for si, seq in enumerate(v):
                        dists.append(pool.apply_async(aadists, args=(seq, subseq2, map1, map2)))
                    pool.close()
                    pool.join()
                    dists = [x.get() for x in dists]
                else:
                    for si, seq in enumerate(v):
                        dists.append(aadists(seq, subseq2, map1, map2))
                # Store and output results
                tsvrecs = []
                for si, seq in enumerate(v):
                    seqtitle = "Seq_%s_%s-%s" % (region, k+1, si)
                    aadist = list(dists[si])
                    maxdist = np.max(aadist)
                    sumdist = np.sum(aadist)
                    meandists = np.mean(aadist)
                    tsvrecs.append([seqtitle, seq, meandists, sumdist, maxdist, aadist.index(maxdist) + 1])
                print("Sorting mutants...")
                tsvrecs.sort(key=lambda q: q[3])
                #output all
                if not hmlonly:
                    with open(os.path.join(outdir, "mutant_%s.tsv" % (k+1)), "w") as ofil, open(os.path.join(outdir, "mutant_%s.fasta" % (k+1)), "w") as fofil:
                        ofil.write("#Seq_title\tSeq_region_%s\tmean-distance\tsum-distance\tmax-distance\tmax-index\tsiteschanged\n" % region)
                        for seqtitle, seq, meandists, sumdist, maxdist, d in tsvrecs:
                            fullseq = clusterMap[2]["seq"]
                            strt, stop = clusterMap[2]["intlocs"][region]
                            fullseq = fullseq[:strt] + seq + fullseq[stop + 1:]
                            ofil.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (seqtitle, seq, meandists, sumdist, maxdist, d, k+1))
                            fofil.write(">%s|region=%s|mean-d=%s|sum-d=%s|max-d=%s|sites=%s\n%s\n" % (seqtitle, region, meandists, sumdist, maxdist, k + 1, fullseq))
                if len(tsvrecs) > 3*hml:
                    hmlrecs[region] = tsvrecs[:hml] + random.sample(tsvrecs[hml:-1*hml], hml) + tsvrecs[-1*hml:]
                else:
                    hmlrecs[region] = tsvrecs
                #output High Medium Low
                with open(os.path.join(outdir, "mutant_hml_%s.tsv" % (k+1)), "w") as ofil, open(os.path.join(outdir, "mutant_hml_%s.fasta" % (k+1)), "w") as fofil:
                    ofil.write("#Seq_title\tSeq_region_%s\tmean-distance\tsum-distance\tmax-distance\tmax-index\tsiteschanged\n" % region)
                    for seqtitle, seq, meandists, sumdist, maxdist, d in hmlrecs[region]:
                        fullseq = clusterMap[2]["seq"]
                        strt, stop = clusterMap[2]["intlocs"][region]
                        fullseq = fullseq[:strt] + seq + fullseq[stop + 1:]
                        ofil.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (seqtitle, seq, meandists, sumdist, maxdist, d, k+1))
                        fofil.write(">%s|region=%s|mean-d=%s|sum-d=%s|max-d=%s|sites=%s\n%s\n" % (seqtitle, region, meandists, sumdist, maxdist, k + 1, fullseq))
        #Make and output combined mutants
        print("Generating combined region mutants...")
        outdir = os.path.join(outpath, "mutants_combined")
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        combined = [] #title,fullseq,sum-sum,max,region-ind
        for region in regions:
            #ensure lists of high-med-low are same length by repeating shorter lists
            hmllen = len(hmlrecs[region])
            if hmllen < 3*hml:
                temprecs = hmlrecs[region]*np.ceil((3.0*hml)/hmllen) # repeat X times to get above 3*hml values
                temprecs.sort(key=lambda q: q[3]) # re-sort
                hmlrecs[region] = temprecs[:3*hml] # truncate
            if not len(combined):
                #First addition
                fullseq = clusterMap[2]["seq"]
                strt, stop = clusterMap[2]["intlocs"][region]
                for seqtitle, seq, meandists, sumdist, maxdist, d in hmlrecs[region]:
                    combined.append([seqtitle, fullseq[:strt] + seq + fullseq[stop + 1:], sumdist, maxdist, "%s_%s"%(region,d)])
            else:
                for ri,rec in enumerate(hmlrecs[region]):
                    fullseq = combined[ri][1]
                    strt, stop = clusterMap[2]["intlocs"][region]
                    seqtitle = "%s_%s" % (combined[ri][0], rec[0].replace("Seq_", ""))
                    #Compare maxdists
                    if rec[4] > combined[ri][3]:
                        maxdist = rec[4]
                        d = "%s_%s" % (region, rec[5])
                    else:
                        maxdist = combined[ri][3]
                        d = combined[ri][4]
                    #add sumdist
                    sumdist = combined[ri][2] + rec[3]
                    combined[ri] = [seqtitle, fullseq[:strt] + rec[1] + fullseq[stop + 1:], sumdist, maxdist, d]
        #Output combined mutatns
        with open(os.path.join(outdir, "combined_mutants_%s.tsv" % "-".join(regions)), "w") as ofil, open(os.path.join(outdir, "combined_mutants_%s.fasta" % "-".join(regions)), "w") as fofil:
            ofil.write("#Seq_title\tsum-distance\tmax-distance\tmax-region-index\n")
            for seqtitle, fullseq, sumdist, maxdist, d in combined:
                ofil.write("%s\t%s\t%s\t%s\n" % (seqtitle, sumdist, maxdist, d))
                fofil.write(">%s|regions=%s|sum-d=%s|max-d=%s\n%s\n" % (seqtitle, "-".join(regions), sumdist, maxdist, fullseq))

# Commandline Execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Align NRPS C-A sequences from fasta file to highlight and 
                                                    optimize interface regions""")
    parser.add_argument("input", help="Fasta sequence with 2-3 records. Cluster A: First seq in fasta (upstream seq) "
                                      "B: 2nd seq in fasta (used for Adomain swap) "
                                      "C: Re-engineered seq (required)")
    parser.add_argument("-td", "--tempdir", help="Directory to create temporary results folder (system default)", default=None)
    parser.add_argument("-r", "--regions", help="Region(s) to be mutated and scored (e.g. A2)", default="A2")
    parser.add_argument("-mms", "--maxmutantseqs", help="Maximum mutant seqs to generate for each round (default: 1e6)", type=float, default=1e6)
    parser.add_argument("-maxs", "--maxsites", help="Maximum sites changed (default: 0.5, 50%% of sequence). Integers over 1 interpreted as exact maximum", type=float, default=0.5)
    parser.add_argument("-o", "--outputdir", help="Saves all output in directory (defaut = mutants_[timestamp])", default=None)
    parser.add_argument("-mm", "--mutatemode", help="Mutation mode. Totally random = 0 (default), Mossaic = 1", type=int, default=0)
    parser.add_argument("-c", "--cpus", help="Number of cpus to use for parallel distance calculation (default: max)", default=mp.cpu_count())
    parser.add_argument("-hml", "--highmedlow", help="Output top X, bottom X and med (X random remaining) scoring mutants. default=100", default=100)
    parser.add_argument("-hmlonly", "--highmedlowonly", help="Only output high med low (discard full list of mutants to save hard disk space)", action='store_true', default=False)
    parser.add_argument("-so", "--scoreonly", help="Only show similarity score of regions for clusterC to native regions", action='store_true', default=False)
    args = parser.parse_args()
    alignCA(args.input, tempdir=args.tempdir, scoreonly=args.scoreonly, regions=args.regions, outpath=args.outputdir, maxseqs=args.maxmutantseqs, maxsites=args.maxsites, mode=args.mutatemode, mcpu=args.cpus, hml=args.highmedlow, hmlonly=args.highmedlowonly)
