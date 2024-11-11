#!/usr/bin/env python3

# This file illustrates how you might experiment with the HMM interface.
# You can paste these commands in at the Python prompt, or execute `test_en.py` directly.
# A notebook interface is nicer than the plain Python prompt, so we provide
# a notebook version of this file as `test_en.ipynb`, which you can open with
# `jupyter` or with Visual Studio `code` (run it with the `nlp-class` kernel).

import logging
import math
import os
from pathlib import Path

import torch

from corpus import TaggedCorpus
from eval import eval_tagging, model_cross_entropy, viterbi_error_rate
from hmm import HiddenMarkovModel
from crf import ConditionalRandomField

# Set up logging.
logging.root.setLevel(level=logging.INFO)
log = logging.getLogger("test_en")       # For usage, see findsim.py in earlier assignment.
logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)  # could change INFO to DEBUG
# torch.autograd.set_detect_anomaly(True)    # uncomment to improve error messages from .backward(), but slows down

# Switch working directory to the directory where the data live.  You may need to edit this line.
os.chdir("../data")

entrain = TaggedCorpus(Path("ensup"), Path("enraw"))                               # all training
ensup =   TaggedCorpus(Path("ensup"), tagset=entrain.tagset, vocab=entrain.vocab)  # supervised training
endev =   TaggedCorpus(Path("endev"), tagset=entrain.tagset, vocab=entrain.vocab)  # evaluation
print(f"{len(entrain)=}  {len(ensup)=}  {len(endev)=}")

known_vocab = TaggedCorpus(Path("ensup")).vocab    # words seen with supervised tags; used in evaluation
log.info(f"Tagset: f{list(entrain.tagset)}")

# Make an HMM.  Let's do some pre-training to approximately maximize the
# regularized log-likelihood on supervised training data.  In other words, the
# probabilities at the M step will just be supervised count ratios.
#
# On each epoch, you will see two progress bars: first it collects counts from
# all the sentences (E step), and then after the M step, it evaluates the loss
# function, which is the (unregularized) cross-entropy on the training set.
# 
# The parameters don't actually matter during the E step because there are no
# hidden tags to impute.  The first M step will jump right to the optimal
# solution.  The code will try a second epoch with the revised parameters, but
# the result will be identical, so it will detect convergence and stop.
#
# We arbitrarily choose λ=1 for our add-λ smoothing at the M step, but it would
# be better to search for the best value of this hyperparameter.

log.info("*** Hidden Markov Model (HMM)")
hmm = HiddenMarkovModel(entrain.tagset, entrain.vocab)  # randomly initialized parameters  
loss_sup = lambda model: model_cross_entropy(model, eval_corpus=ensup)
hmm.train(corpus=ensup, loss=loss_sup, λ=1.0,
          save_path="ensup_hmm.pkl") 

# Now let's throw in the unsupervised training data as well, and continue
# training as before, in order to increase the regularized log-likelihood on
# this larger, semi-supervised training set.  It's now the *incomplete-data*
# log-likelihood.
# 
# This time, we'll use a different evaluation loss function: we'll stop when the
# *tagging error rate* on a held-out dev set stops getting better.  Also, the
# implementation of this loss function (`viterbi_error_rate`) includes a helpful
# side effect: it logs the *cross-entropy* on the held-out dataset as well, just
# for your information.
# 
# We hope that held-out tagging accuracy will go up for a little bit before it
# goes down again (see Merialdo 1994). (Log-likelihood on training data will
# continue to improve, and that improvement may generalize to held-out
# cross-entropy.  But getting accuracy to increase is harder.)

hmm = HiddenMarkovModel.load("ensup_hmm.pkl")  # reset to supervised model (in case you're re-executing this bit)
loss_dev = lambda model: viterbi_error_rate(model, eval_corpus=endev, 
                                            known_vocab=known_vocab)
hmm.train(corpus=entrain, loss=loss_dev, λ=1.0,
          save_path="entrain_hmm.pkl")

# You can also retry the above workflow where you start with a worse supervised
# model (like Merialdo).  Does EM help more in that case?  It's easiest to rerun
# exactly the code above, but first make the `ensup` file smaller by copying
# `ensup-tiny` over it.  `ensup-tiny` is only 25 sentences (that happen to cover
# all tags in `endev`).  Back up your old `ensup` and your old `*.pkl` models
# before you do this.

# More detailed look at the first 10 sentences in the held-out corpus,
# including Viterbi tagging.
def look_at_your_data(model, dev, N):
    for m, sentence in enumerate(dev):
        if m >= N: break
        viterbi = model.viterbi_tagging(sentence.desupervise(), endev)
        counts = eval_tagging(predicted=viterbi, gold=sentence, 
                              known_vocab=known_vocab)
        num = counts['NUM', 'ALL']
        denom = counts['DENOM', 'ALL']
        
        log.info(f"Gold:    {sentence}")
        log.info(f"Viterbi: {viterbi}")
        log.info(f"Loss:    {denom - num}/{denom}")
        xent = -model.logprob(sentence, endev) / len(sentence)  # measured in nats
        log.info(f"Cross-entropy: {xent/math.log(2)} nats (= perplexity {math.exp(xent)})\n---")

look_at_your_data(hmm, endev, 10)

# Now let's try supervised training of a CRF (this doesn't use the unsupervised
# part of the data, so it is comparable to the supervised pre-training we did
# for the HMM).  We will use SGD to approximately maximize the regularized
# log-likelihood. 
#
# As with the semi-supervised HMM training, we'll periodically evaluate the
# tagging accuracy (and also print the cross-entropy) on a held-out dev set.
# We use the default `eval_interval` and `tolerance`.  If you want to stop
# sooner, then you could increase the `tolerance` so the training method decides
# sooner that it has converged.
# 
# We arbitrarily choose reg = 1.0 for L2 regularization, learning rate = 0.05,
# and a minibatch size of 10, but it would be better to search for the best
# value of these hyperparameters.
#
# Note that the logger reports the CRF's *conditional* cross-entropy, log p(tags
# | words) / n.  This is much lower than the HMM's *joint* cross-entropy log
# p(tags, words) / n, but that doesn't mean the CRF is worse at tagging.  The
# CRF is just predicting less information.

log.info("*** Conditional Random Field (CRF)\n")
crf = ConditionalRandomField(entrain.tagset, entrain.vocab)  # randomly initialized parameters  
crf.train(corpus=ensup, loss=loss_dev, reg=1.0, lr=0.05, minibatch_size=10,
          save_path="ensup_crf.pkl")

# Let's examine how the CRF does on individual sentences. 
# (Do you see any error patterns here that would inspire additional CRF features?)

look_at_your_data(crf, endev, 10)
