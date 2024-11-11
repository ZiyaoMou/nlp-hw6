#!/usr/bin/env python3

# This file illustrates how you might experiment with the HMM interface.
# You can paste these commands in at the Python prompt, or execute `test_ic.py` directly.
# A notebook interface is nicer than the plain Python prompt, so we provide
# a notebook version of this file as `test_ic.ipynb`, which you can open with
# `jupyter` or with Visual Studio `code` (run it with the `nlp-class` kernel).

import logging, math, os
from pathlib import Path

import torch
from torch import tensor

from corpus import TaggedCorpus
from eval import model_cross_entropy, write_tagging
from hmm import HiddenMarkovModel
from crf import ConditionalRandomField

# Set up logging.
log = logging.getLogger("test_ic")       # For usage, see findsim.py in earlier assignment.
logging.root.setLevel(level=logging.INFO)
logging.basicConfig(level=logging.INFO)  # could change INFO to DEBUG
# torch.autograd.set_detect_anomaly(True)    # uncomment to improve error messages from .backward(), but slows down

# Switch working directory to the directory where the data live.  You may want to edit this line.
os.chdir("../data")

# Get vocabulary and tagset from a supervised corpus.
icsup = TaggedCorpus(Path("icsup"), add_oov=False)
log.info(f"Ice cream vocabulary: {list(icsup.vocab)}")
log.info(f"Ice cream tagset: {list(icsup.tagset)}")

# Two ways to look at the corpus ...
os.system("cat icsup")   # call the shell to look at the file directly

log.info(icsup)          # print the TaggedCorpus python object we constructed from it

# Make an HMM.
log.info("*** Hidden Markov Model (HMM) test\n")
hmm = HiddenMarkovModel(icsup.tagset, icsup.vocab)
# Change the transition/emission initial probabilities to match the ice cream spreadsheet,
# and test your implementation of the Viterbi algorithm.  Note that the spreadsheet 
# uses transposed versions of these matrices.
hmm.B = tensor([[0.7000, 0.2000, 0.1000],    # emission probabilities
                [0.1000, 0.2000, 0.7000],
                [0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000]])
hmm.A = tensor([[0.8000, 0.1000, 0.1000, 0.0000],   # transition probabilities
                [0.1000, 0.8000, 0.1000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000],
                [0.5000, 0.5000, 0.0000, 0.0000]])
log.info("*** Current A, B matrices (using initalizations from the ice cream spreadsheet)")
hmm.printAB()

# Try it out on the raw data from the spreadsheet, available in `icraw``.
log.info("*** Viterbi results on icraw with hard coded parameters")
icraw = TaggedCorpus(Path("icraw"), tagset=icsup.tagset, vocab=icsup.vocab)
write_tagging(hmm, icraw, Path("icraw_hmm.output"))  # calls hmm.viterbi_tagging on each sentence
os.system("cat icraw_hmm.output")   # print the file we just created, and remove it

# Did the parameters that we guessed above get the "correct" answer, 
# as revealed in `icdev`?
icdev = TaggedCorpus(Path("icdev"), tagset=icsup.tagset, vocab=icsup.vocab)
log.info(f"*** Compare to icdev corpus:\n{icdev}")
from eval import viterbi_error_rate
viterbi_error_rate(hmm, icdev, show_cross_entropy=False)

# Now let's try your training code, running it on supervised data.
# To test this, we'll restart from a random initialization.
# (You could also try creating this new model with `unigram=true`, 
# which will affect the rest of the notebook.)
hmm = HiddenMarkovModel(icsup.tagset, icsup.vocab)
log.info("*** A, B matrices as randomly initialized close to uniform")
hmm.printAB()

log.info("*** Supervised training on icsup")
cross_entropy_loss = lambda model: model_cross_entropy(model, icsup)
hmm.train(corpus=icsup, loss=cross_entropy_loss, tolerance=0.0001)
log.info("*** A, B matrices after training on icsup (should "
         "match initial params on spreadsheet [transposed])")
hmm.printAB()

# Now that we've reached the spreadsheet's starting guess, let's again tag
# the spreadsheet "sentence" (that is, the sequence of ice creams) using the
# Viterbi algorithm.
log.info("*** Viterbi results on icraw")
icraw = TaggedCorpus(Path("icraw"), tagset=icsup.tagset, vocab=icsup.vocab)
write_tagging(hmm, icraw, Path("icraw_hmm.output"))  # calls hmm.viterbi_tagging on each sentence
os.system("cat icraw_hmm.output")   # print the file we just created, and remove it

# Next let's use the forward algorithm to see what the model thinks about 
# the probability of the spreadsheet "sentence."
log.info("*** Forward algorithm on icraw (should approximately match iteration 0 "
             "on spreadsheet)")
for sentence in icraw:
    prob = math.exp(hmm.logprob(sentence, icraw))
    log.info(f"{prob} = p({sentence})")

# Finally, let's reestimate on the icraw data, as the spreadsheet does.
# We'll evaluate as we go along on the *training* perplexity, and stop
# when that has more or less converged.
log.info("*** Reestimating on icraw (perplexity should improve on every iteration)")
negative_log_likelihood = lambda model: model_cross_entropy(model, icraw)  # evaluate on icraw itself
hmm.train(corpus=icraw, loss=negative_log_likelihood, tolerance=0.0001)

log.info("*** A, B matrices after reestimation on icraw"
         "should match final params on spreadsheet [transposed])")
hmm.printAB()

# Now let's try out a randomly initialized CRF on the ice cream data. Notice how
# the initialized A and B matrices now hold non-negative potentials,
# rather than probabilities that sum to 1.

log.info("*** Conditional Random Field (CRF) test\n")
crf = ConditionalRandomField(icsup.tagset, icsup.vocab)
log.info("*** Current A, B matrices (potentials from small random parameters)")
crf.printAB()

# Now let's try your training code, running it on supervised data. To test this,
# we'll restart from a random initialization. 
# 
# Note that the logger reports the CRF's *conditional* cross-entropy, 
# log p(tags | words) / n.  This is much lower than the HMM's *joint* 
# cross-entropy log p(tags, words) / n, but that doesn't mean the CRF
# is worse at tagging.  The CRF is just predicting less information.
log.info("*** Supervised training on icsup")
cross_entropy_loss = lambda model: model_cross_entropy(model, icsup)
crf.train(corpus=icsup, loss=cross_entropy_loss, lr=0.1, tolerance=0.0001)
log.info("*** A, B matrices after training on icsup")
crf.printAB()

# Let's again tag the spreadsheet "sentence" (that is, the sequence of ice creams) 
# using the Viterbi algorithm (this may not match the HMM).
log.info("*** Viterbi results on icraw with trained parameters")
icraw = TaggedCorpus(Path("icraw"), tagset=icsup.tagset, vocab=icsup.vocab)
write_tagging(hmm, icraw, Path("icraw_crf.output"))  # calls hmm.viterbi_tagging on each sentence
os.system("cat icraw_crf.output")   # print the file we just created, and remove it
