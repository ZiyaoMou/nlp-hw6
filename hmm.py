#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Starter code for Hidden Markov Models.

from __future__ import annotations
from ast import Index
import logging
from math import inf, log, exp, prod
from pathlib import Path
from typing import Callable, List, Optional, cast
from sympy import N
from typeguard import typechecked

import torch
from torch import Tensor, cuda, nn
from jaxtyping import Float

from tqdm import tqdm # type: ignore
import pickle

from integerize import Integerizer
from corpus import BOS_TAG, BOS_WORD, EOS_TAG, EOS_WORD, Sentence, Tag, TaggedCorpus, IntegerizedSentence, Word

TorchScalar = Float[Tensor, ""] # a Tensor with no dimensions, i.e., a scalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

###
# HMM tagger
###
class HiddenMarkovModel:
    """An implementation of an HMM, whose emission probabilities are
    parameterized using the word embeddings in the lexicon.
    
    We'll refer to the HMM states as "tags" and the HMM observations 
    as "words."
    """
    
    # As usual in Python, attributes and methods starting with _ are intended as private;
    # in this case, they might go away if you changed the parametrization of the model.

    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 unigram: bool = False):
        """Construct an HMM with initially random parameters, with the
        given tagset, vocabulary, and lexical features.
        
        Normally this is an ordinary first-order (bigram) HMM.  The `unigram` flag
        says to fall back to a zeroth-order HMM, in which the different
        positions are generated independently.  (The code could be extended to
        support higher-order HMMs: trigram HMMs used to be popular.)"""

        # We'll use the variable names that we used in the reading handout, for
        # easy reference.  (It's typically good practice to use more descriptive names.)

        # We omit EOS_WORD and BOS_WORD from the vocabulary, as they can never be emitted.
        # See the reading handout section "Don't guess when you know."

        if vocab[-2:] != [EOS_WORD, BOS_WORD]:
            raise ValueError("final two types of vocab should be EOS_WORD, BOS_WORD")

        self.k = len(tagset)       # number of tag types
        self.V = len(vocab) - 2    # number of word types (not counting EOS_WORD and BOS_WORD)
        self.unigram = unigram     # do we fall back to a unigram model?

        self.tagset = tagset
        self.vocab = vocab

        # Useful constants that are referenced by the methods
        self.bos_t: Optional[int] = tagset.index(BOS_TAG)
        self.eos_t: Optional[int] = tagset.index(EOS_TAG)
        if self.bos_t is None or self.eos_t is None:
            raise ValueError("tagset should contain both BOS_TAG and EOS_TAG")
        assert self.eos_t is not None    # we need this to exist
        self.eye: Tensor = torch.eye(self.k)  # identity matrix, used as a collection of one-hot tag vectors

        self.init_params()     # create and initialize model parameters
 
    def init_params(self) -> None:
        """Initialize params to small random values (which breaks ties in the fully unsupervised case).  
        We respect structural zeroes ("Don't guess when you know").
            
        If you prefer, you may change the class to represent the parameters in logspace,
        as discussed in the reading handout as one option for avoiding underflow; then name
        the matrices lA, lB instead of A, B, and construct them by logsoftmax instead of softmax."""

        ###
        # Randomly initialize emission probabilities.
        # A row for an ordinary tag holds a distribution that sums to 1 over the columns.
        # But EOS_TAG and BOS_TAG have probability 0 of emitting any column's word
        # (instead, they have probability 1 of emitting EOS_WORD and BOS_WORD (respectively), 
        # which don't have columns in this matrix).
        ###
        WB = 0.01*torch.rand(self.k, self.V)  # choose random logits
        self.B = WB.softmax(dim=1)            # construct emission distributions p(w | t)
        self.B[self.eos_t, :] = 0             # EOS_TAG can't emit any column's word
        self.B[self.bos_t, :] = 0             # BOS_TAG can't emit any column's word
        
        ###
        # Randomly initialize transition probabilities, in a similar way.
        # Again, we respect the structural zeros of the model.
        ###
        rows = 1 if self.unigram else self.k
        WA = 0.01*torch.rand(rows, self.k)
        WA[:, self.bos_t] = -inf    # correct the BOS_TAG column
        self.A = WA.softmax(dim=1)  # construct transition distributions p(t | s)
        if self.unigram:
            # A unigram model really only needs a vector of unigram probabilities
            # p(t), but we'll construct a bigram probability matrix p(t | s) where 
            # p(t | s) doesn't depend on s. 
            # 
            # By treating a unigram model as a special case of a bigram model,
            # we can simply use the bigram code for our unigram experiments,
            # although unfortunately that preserves the O(nk^2) runtime instead
            # of letting us speed up to O(nk) in the unigram case.
            self.A = self.A.repeat(self.k, 1)   # copy the single row k times  

    def printAB(self) -> None:
        """Print the A and B matrices in a more human-readable format (tab-separated)."""
        print("Transition matrix A:")
        col_headers = [""] + [str(self.tagset[t]) for t in range(self.A.size(1))]
        print("\t".join(col_headers))
        for s in range(self.A.size(0)):   # rows
            row = [str(self.tagset[s])] + [f"{self.A[s,t]:.3f}" for t in range(self.A.size(1))]
            print("\t".join(row))
        print("\nEmission matrix B:")        
        col_headers = [""] + [str(self.vocab[w]) for w in range(self.B.size(1))]
        print("\t".join(col_headers))
        for t in range(self.A.size(0)):   # rows
            row = [str(self.tagset[t])] + [f"{self.B[t,w]:.3f}" for w in range(self.B.size(1))]
            print("\t".join(row))
        print("\n")

    def M_step(self, λ: float) -> None:
        """Set the transition and emission matrices (A, B), using the expected
        counts (A_counts, B_counts) that were accumulated by the E step.
        The `λ` parameter will be used for add-λ smoothing.
        We respect structural zeroes ("don't guess when you know")."""

        # we should have seen no emissions from BOS or EOS tags
        assert self.B_counts[self.eos_t:self.bos_t, :].any() == 0, 'Your expected emission counts ' \
                'from EOS and BOS are not all zero, meaning you\'ve accumulated them incorrectly!'

        # Update emission probabilities (self.B).
        self.B_counts += λ          # smooth the counts (EOS_WORD and BOS_WORD remain at 0 since they're not in the matrix)
        self.B = self.B_counts / self.B_counts.sum(dim=1, keepdim=True)  # normalize into prob distributions
        self.B[self.eos_t, :] = 0   # replace these nan values with structural zeroes, just as in init_params
        self.B[self.bos_t, :] = 0

        # we should have seen no "tag -> BOS" or "BOS -> tag" transitions
        assert self.A_counts[:, self.bos_t].any() == 0, 'Your expected transition counts ' \
                'to BOS are not all zero, meaning you\'ve accumulated them incorrectly!'
        assert self.A_counts[self.eos_t, :].any() == 0, 'Your expected transition counts ' \
                'from EOS are not all zero, meaning you\'ve accumulated them incorrectly!'
                
        # Update transition probabilities (self.A).  
        # Don't forget to respect the settings self.unigram and λ.
        # See the init_params() method for a discussion of self.A in the
        # unigram case.
        # print(self.A_counts)
        # print(self.B_counts)
        self.A_counts += λ
        # print(self.A_counts)
        if not self.unigram:
            self.A = self.A_counts / self.B_counts.sum(dim=1, keepdim=True)
            self.A[self.bos_t] = self.A_counts[self.bos_t] / self.A_counts[self.bos_t].sum()
            self.A[:, self.bos_t] = 0
            self.A[self.eos_t] = 0
            # 1 - sum of A in other columns
            self.A[:, self.eos_t] = 0
            self.A[:, self.eos_t] = torch.clamp(1 - self.A.sum(dim=1), min=0)
            self.A[self.eos_t] = 0
            self.A[self.bos_t, self.eos_t] = 0
        else:
            self.A_counts[[self.bos_t, self.eos_t]] = 0
            self.A = self.A_counts.sum(dim=0) / (self.A_counts.sum() + 1)
            self.A[self.bos_t] = 1 / (self.A_counts.sum() + 1)
            self.A = self.A.repeat(self.k, 1)
        # print(self.A)
        # print(self.B)

    def _zero_counts(self):
        """Set the expected counts to 0.  
        (This creates the count attributes if they didn't exist yet.)"""
        self.A_counts = torch.zeros((self.k, self.k), requires_grad=False)
        self.B_counts = torch.zeros((self.k, self.V), requires_grad=False)

    def train(self,
              corpus: TaggedCorpus,
              loss: Callable[[HiddenMarkovModel], float],
              λ: float = 0,
              tolerance: float = 0.001,
              max_steps: int = 50000,
              save_path: Optional[Path] = Path("my_hmm.pkl")) -> None:
        """Train the HMM on the given training corpus, starting at the current parameters.
        We will stop when the relative improvement of the development loss,
        since the last epoch, is less than the tolerance.  In particular,
        we will stop when the improvement is negative, i.e., the development loss 
        is getting worse (overfitting).  To prevent running forever, we also
        stop if we exceed the max number of steps."""
        
        if λ < 0:
            raise ValueError(f"{λ=} but should be >= 0")
        elif λ == 0:
            λ = 1e-20
            # Smooth the counts by a tiny amount to avoid a problem where the M
            # step gets transition probabilities p(t | s) = 0/0 = nan for
            # context tags s that never occur at all, in particular s = EOS.
            # 
            # These 0/0 probabilities are never needed since those contexts
            # never occur.  So their value doesn't really matter ... except that
            # we do have to keep their value from being nan.  They show up in
            # the matrix version of the forward algorithm, where they are
            # multiplied by 0 and added into a sum.  A summand of 0 * nan would
            # regrettably turn the entire sum into nan.      
      
        dev_loss = loss(self)   # evaluate the model at the start of training
        
        old_dev_loss: float = dev_loss     # loss from the last epoch
        step: int = 0   # total number of sentences the model has been trained on so far      
        while step < max_steps:
            
            # E step: Run forward-backward on each sentence, and accumulate the
            # expected counts into self.A_counts, self.B_counts.
            #
            # Note: If you were using a GPU, you could get a speedup by running
            # forward-backward on several sentences in parallel.  This would
            # require writing the algorithm using higher-dimensional tensor
            # operations, allowing PyTorch to take advantage of hardware
            # parallelism.  For example, you'd update alpha[j-1] to alpha[j] for
            # all the sentences in the minibatch at once (with appropriate
            # handling for short sentences of length < j-1).  

            self._zero_counts()
            for sentence in tqdm(corpus, total=len(corpus), leave=True):
                isent = self._integerize_sentence(sentence, corpus)
                self.E_step(isent)

            # M step: Update the parameters based on the accumulated counts.
            self.M_step(λ)
            
            # Evaluate with the new parameters
            dev_loss = loss(self)   # this will print its own log messages
            if dev_loss >= old_dev_loss * (1-tolerance):
                # we haven't gotten much better, so perform early stopping
                break
            old_dev_loss = dev_loss            # remember for next eval batch
        
        # For convenience when working in a Python notebook, 
        # we automatically save our training work by default.
        if save_path: self.save(save_path)
  
    def _integerize_sentence(self, sentence: Sentence, corpus: TaggedCorpus) -> IntegerizedSentence:
        """Integerize the words and tags of the given sentence, which came from the given corpus."""

        if corpus.tagset != self.tagset or corpus.vocab != self.vocab:
            # Sentence comes from some other corpus that this HMM was not set up to handle.
            raise TypeError("The corpus that this sentence came from uses a different tagset or vocab")

        return corpus.integerize_sentence(sentence)

    @typechecked
    def logprob(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Compute the log probability of a single sentence under the current
        model parameters.  If the sentence is not fully tagged, the probability
        will marginalize over all possible tags.  

        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet.
                
        The corpus from which this sentence was drawn is also passed in as an
        argument, to help with integerization and check that we're integerizing
        correctly."""

        # Integerize the words and tags of the given sentence, which came from the given corpus.
        isent = self._integerize_sentence(sentence, corpus)
        return self.forward_pass(isent)

    def E_step(self, isent: IntegerizedSentence, mult: float = 1) -> None:
        """Runs the forward backward algorithm on the given sentence. The forward step computes
        the alpha probabilities.  The backward step computes the beta probabilities and
        adds expected counts to self.A_counts and self.B_counts.  
        
        The multiplier `mult` says how many times to count this sentence. 
        
        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet."""

        # Forward-backward algorithm.
        log_Z_forward = self.forward_pass(isent)
        log_Z_backward = self.backward_pass(isent, mult=mult)
        
        # Check that forward and backward passes found the same total
        # probability of all paths (up to floating-point error).
        assert torch.isclose(log_Z_forward, log_Z_backward), f"backward log-probability {log_Z_backward} doesn't match forward log-probability {log_Z_forward}!"

    @typechecked
    def forward_pass(self, isent: IntegerizedSentence) -> TorchScalar:
        """Run the forward algorithm on a tagged, untagged, or partially tagged sentence.
        Return log Z (the log of the forward probability) as a TorchScalar. If the sentence 
        is not fully tagged, the forward probability will marginalize over all possible tags.  
        
        This method stores the alpha probabilities and log_Z into self for subsequent use 
        in the backward pass."""

        # Initialize alpha with -inf for all tags, except BOS_TAG
        n = len(isent) - 2  # length of sentence without BOS and EOS
        alpha = torch.full((len(isent), self.k), -float('inf'))
        alpha[0][self.bos_t] = 0  # BOS_TAG starts with log-prob of 0

        # Forward pass through each position in the sentence except the EOS step
        for j in range(1, n + 1):
            word = isent[j][0]
            prev_tag = isent[j - 1][1]
            current_tag = isent[j][1]

            # Compute alpha[j] based on tagging status of j and j-1
            if current_tag is None:
                # j is not tagged, sum over all possible tags for j
                alpha[j] = torch.logsumexp(
                    alpha[j - 1].unsqueeze(1) + torch.log(self.A) + torch.log(self.B[:, word]).unsqueeze(0), dim=0
                )
            elif prev_tag is None:
                # j is tagged, but j-1 is not tagged
                alpha[j][current_tag] = torch.logsumexp(
                    alpha[j - 1] + torch.log(self.A[:, current_tag]) + torch.log(self.B[current_tag, word]), dim=0
                )
            else:
                # Both j and j-1 are tagged
                alpha[j][current_tag] = (
                    alpha[j - 1][prev_tag]
                    + torch.log(self.A[prev_tag, current_tag])
                    + torch.log(self.B[current_tag, word])
                )

        # Final step for EOS tag (j = n + 1), omit B matrix
        if isent[n][1] is None:  # n is not tagged
            alpha[n + 1][self.eos_t] = torch.logsumexp(alpha[n] + torch.log(self.A[:, self.eos_t]), dim=0)
        else:  # n is tagged
            alpha[n + 1][self.eos_t] = alpha[n][isent[n][1]] + torch.log(self.A[isent[n][1], self.eos_t])

        # Calculate log Z, the log-probability of observing the given sentence
        log_Z = alpha[n + 1][self.eos_t]
        self.alpha = alpha  # Save alpha for backward pass
        self.log_Z = log_Z  # Save log Z
        return log_Z

    @typechecked
    def backward_pass(self, isent: IntegerizedSentence, mult: float = 1) -> TorchScalar:
        """Run the backwards algorithm from the handout on a tagged, untagged, 
        or partially tagged sentence.  Return log Z (the log of the backward
        probability). 
        
        As a side effect, add the expected transition and emission counts (times
        mult) into self.A_counts and self.B_counts.  These depend on the alpha
        values and log Z, which were stored for us (in self) by the forward
        pass."""

        # Pre-allocate beta just as we pre-allocated alpha.
        beta = [torch.empty(self.k) for _ in isent]
        beta[-1] = torch.log(self.eye[self.eos_t])  # vector that is one-hot at EOS_TAG
        n = len(isent) - 2
        for j in range(n+1, 0, -1):
            # tag1, tag2 = isent[j-1][1], isent[j][1]
            # if tag1 is None and tag2 is None:
            #     for t in range(self.k):
            #         beta[j-1][t] = torch.logsumexp(torch.log(self.A[t, :]) + beta[j], dim=0)
            #         if j != n+1:
            #             beta[j-1][t] = torch.logsumexp(beta[j-1][t] + torch.log(self.B[t, isent[j][0]]), dim=0)
            # elif tag1 is None:
            #     for t in range(self.k):
            #         beta[j-1][t] = torch.logsumexp(torch.log(self.A[t, tag2]) + beta[j][tag2], dim=0)
            #         if j != n+1:
            #             beta[j-1][t] = torch.logsumexp(beta[j-1][t] + torch.log(self.B[tag2, isent[j][0]]), dim=0)
            # elif tag2 is None:
            #     beta[j-1][tag1] = torch.logsumexp(torch.log(self.A[tag1, :]) + beta[j], dim=0)
            #     if j != n+1:
            #         beta[j-1][tag1] = torch.logsumexp(beta[j-1][tag1] + torch.log(self.B[tag1, isent[j][0]]), dim=0)
            # else:
            #     beta[j-1][tag1] = torch.log(self.A[tag1, tag2]) + beta[j][tag2]
            #     if j != n+1:
            #         beta[j-1][tag1] = beta[j-1][tag1] + torch.log(self.B[tag2, isent[j][0]])
            if j != n+1:
                tag1, tag2 = isent[j-1][1], isent[j][1]
                if tag1 is None and tag2 is None:
                    for t in range(self.k):
                        beta[j-1][t] = torch.logsumexp(torch.log(self.A[t, :]) + torch.log(self.B[:, isent[j][0]]) + beta[j], dim=0)
                elif tag1 is None:
                    for t in range(self.k):
                        beta[j-1][t] = torch.log(self.A[t, tag2]) + torch.log(self.B[tag2, isent[j][0]]) + beta[j][tag2]
                elif tag2 is None:
                    beta[j-1][tag1] = torch.logsumexp(torch.log(self.A[tag1, :]) + torch.log(self.B[:, isent[j][0]]) + beta[j], dim=0)
                else:
                    beta[j-1][tag1] = torch.log(self.A[tag1, tag2]) + torch.log(self.B[tag2, isent[j][0]]) + beta[j][tag2]
            else:
                tag1, tag2 = isent[j-1][1], isent[j][1]
                if tag1 is None and tag2 is None:
                    for t in range(self.k):
                            beta[j-1][t] = torch.logsumexp(torch.log(self.A[t, :]) + beta[j], dim=0)
                elif tag1 is None:
                    for t in range(self.k):
                        beta[j-1][t] = torch.log(self.A[t, tag2]) + beta[j][tag2]
                elif tag2 is None:
                    beta[j-1][tag1] = torch.logsumexp(torch.log(self.A[tag1, :]) + beta[j], dim=0)
                else:
                    beta[j-1][tag1] = torch.log(self.A[tag1, tag2]) + beta[j][tag2]
            # print("beta", j-1, torch.exp(beta[j-1])) #TODO:
            # if max(beta[j-1]) > 0:
            #     exit()
        for j in range(1, n+1):
            tag1, tag2 = isent[j-1][1], isent[j][1]
            if tag2 is None:
                for s in range(self.k): # tag of j
                    self.B_counts[s, isent[j][0]] += torch.exp(self.alpha[j][s] + beta[j][s] - self.log_Z) * mult
                    if tag1 is None:
                        for t in range(self.k): # tag of j-1
                            self.A_counts[t, s] += torch.exp(self.alpha[j-1][t] + torch.log(self.A[t,s]) + torch.log(self.B[s,isent[j][0]]) + beta[j][s] - self.log_Z) * mult
                    else:
                        self.A_counts[tag1, s] = torch.exp(self.alpha[j-1][tag1] + torch.log(self.A[tag1, s]) + torch.log(self.B[s,isent[j][0]]) + beta[j][s] - self.log_Z) * mult
            else:
                self.B_counts[tag2, isent[j][0]] += torch.exp(self.alpha[j][tag2] + beta[j][tag2] - self.log_Z) * mult
                if tag1 is None:
                    for t in range(self.k):
                        self.A_counts[t, tag2] += torch.exp(self.alpha[j-1][t] + torch.log(self.A[t,tag2]) + torch.log(self.B[tag2,isent[j][0]]) + beta[j][tag2] - self.log_Z) * mult
                else:
                    self.A_counts[tag1, tag2] += torch.exp(self.alpha[j-1][tag1] + torch.log(self.A[tag1,tag2]) + torch.log(self.B[tag2,isent[j][0]]) + beta[j][tag2] - self.log_Z) * mult
        # print('B_counts', self.B_counts)
        # print('A_counts', self.A_counts)
        return beta[0][self.bos_t]

    def viterbi_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        """Find the most probable tagging for the given sentence, according to the
        current model."""

        # Note: This code is mainly copied from the forward algorithm.
        # We just switch to using max, and follow backpointers.
        # The code continues to use the name alpha, rather than \hat{alpha}
        # as in the handout.

        # We'll start by integerizing the input Sentence. You'll have to
        # deintegerize the words and tags again when constructing the return
        # value, since the type annotation on this method says that it returns a
        # Sentence object, and that's what downstream methods like eval_tagging
        # will expect.  (Running mypy on your code will check that your code
        # conforms to the type annotations ...)

        isent = self._integerize_sentence(sentence, corpus)

        # See comments in log_forward on preallocation of alpha.
        alpha        = [torch.empty(self.k)                  for _ in isent]  
        backpointers = [torch.empty(self.k, dtype=torch.int) for _ in isent]
        n = len(isent) - 2
        tags: List[int] = [0 for _ in range(n+2)]    # you'll put your integerized tagging here
        alpha[0] = torch.log(self.eye[self.bos_t])  # vector that is one-hot at BOS_TAG
        for j in range(1, len(isent)):
            for t in range(self.k):
                # print(alpha[j-1].shape, self.A[:,t].shape, self.B[t, isent[j][0]].shape)
                if j != n+1:
                    prod = alpha[j-1] + torch.log(self.A[:,t]) + torch.log(self.B[t,isent[j][0]])
                else:
                    prod = alpha[j-1] + torch.log(self.A[:,t])
                # print('0', torch.exp(alpha[j-1]), self.A[:,t], self.B[t,isent[j][0]])
                # print('1', torch.exp(prod))
                alpha[j][t] = torch.max(prod)
                backpointers[j][t] = torch.argmax(prod)
            # print(torch.exp(alpha[j]))

        # Now follow backpointers to find the most probable tag sequence.
        tags[n+1] = self.eos_t
        for j in range(n+1, 0, -1):
            tags[j-1] = backpointers[j][tags[j]]
        # Make a new tagged sentence with the old words and the chosen tags
        # (using self.tagset to deintegerize the chosen tags).
        return Sentence([(word, self.tagset[tags[j]]) for j, (word, tag) in enumerate(sentence)])

    def posterior_decoding(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        isent = self._integerize_sentence(sentence, corpus)
        n = len(isent) - 2
        tags = [0] * (n + 2)

        # Perform forward and backward passes to obtain posterior probabilities
        self.forward_pass(isent)
        self.backward_pass(isent)
        
        # Posterior decoding for each tag independently
        for j in range(1, n + 1):
            posterior_probs = torch.exp(self.alpha[j] + self.beta[j] - self.log_Z)
            tags[j] = posterior_probs.argmax().item()

        return Sentence([(word, self.tagset[tags[j]]) for j, (word, tag) in enumerate(sentence)])

    def save(self, model_path: Path) -> None:
        logger.info(f"Saving model to {model_path}")
        torch.save(self, model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved model to {model_path}")

    @classmethod
    def load(cls, model_path: Path, device: str = 'cpu') -> HiddenMarkovModel:
        model = torch.load(model_path, map_location=device)\
            
        # torch.load is similar to pickle.load but handles tensors too
        # map_location allows loading tensors on different device than saved
        if model.__class__ != cls:
            raise ValueError(f"Type Error: expected object of type {cls.__name__} but got {model.__class__.__name__} " \
                             f"from saved file {model_path}.")

        logger.info(f"Loaded model from {model_path}")
        return model