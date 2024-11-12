#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Evaluation of taggers.
import logging
from pathlib import Path
from math import nan, exp
from typing import Counter, Tuple, Optional, Callable, Union

import torch
from torch import nn as nn
from tqdm import tqdm # type: ignore

from corpus import Sentence, Word, EOS_WORD, BOS_WORD, OOV_WORD, TaggedCorpus
from hmm import HiddenMarkovModel
from integerize import Integerizer

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def viterbi_tagger(model: HiddenMarkovModel, eval_corpus: TaggedCorpus) -> Callable[[Sentence], Sentence]:
    def tagger(input:Sentence) -> Sentence:
        return model.viterbi_tagging(input, eval_corpus)
    return tagger

def model_cross_entropy(model: HiddenMarkovModel,
                        eval_corpus: TaggedCorpus) -> float:
    """Return cross-entropy per token of the model on the given evaluation corpus.
    That corpus may be either supervised or unsupervised.
    Warning: Return value is in nats, not bits."""
    logprob = 0.0
    token_count = 0
    for gold in tqdm(eval_corpus, total=len(eval_corpus)):
        logprob += model.logprob(gold, eval_corpus).item()
        token_count += len(gold) - 1    # count EOS but not BOS
    cross_entropy = -logprob / token_count
    log.info(f"Cross-entropy: {cross_entropy:.4f} nats (= perplexity {exp(cross_entropy):.3f})")
    return cross_entropy

def viterbi_error_rate(model: HiddenMarkovModel,
                     eval_corpus: TaggedCorpus,
                     known_vocab: Optional[Integerizer[Word]] = None,
                     show_cross_entropy = True) -> float:
    """Return the error rate of Viterbi tagging with the given model on the given 
    evaluation corpus, after logging cross-entropy (optionally) and a breakdown 
    of accuracy."""

    if show_cross_entropy:
        model_cross_entropy(model, eval_corpus)  # call this for its side effect (logging)
    return tagger_error_rate(viterbi_tagger(model, eval_corpus),
                             eval_corpus,
                             known_vocab=known_vocab)

def tagger_error_rate(tagger: Callable[[Sentence], Sentence],
                     eval_corpus: TaggedCorpus,
                     known_vocab: Optional[Integerizer[Word]] = None) -> float:
    """Return the error rate of the given generic tagger on the given evaluation corpus,
    after printing cross-entropy and a breakdown of accuracy (using the logger)."""

    counts: Counter[Tuple[str, str]] = Counter()  # keep running totals here
    for gold in tqdm(eval_corpus, total=len(eval_corpus)):
        predicted = tagger(gold.desupervise())
        counts += eval_tagging(predicted, gold, known_vocab)   # += works on dictionaries

    def fraction(c:str) -> float:
        num = counts['NUM',c]
        denom = counts['DENOM',c]
        return nan if denom==0 else num / denom

    categories = ['ALL', 'KNOWN', 'SEEN', 'NOVEL']
    if known_vocab is None:
        categories.remove('KNOWN')
    results = [f"{c.lower()}: {(fraction(c)):.3%}" for c in categories]            
    log.info(f"Tagging accuracy: {', '.join(results)}")

    return 1 - fraction('ALL')  # loss value (the error rate)

def eval_tagging(predicted: Sentence, 
                 gold: Sentence, 
                 known_vocab: Optional[Integerizer[Word]]) -> Counter[Tuple[str, str]]:
    """Returns a dictionary with several performance counts,
    comparing the predicted tagging to the gold tagging of the same sentence.

    known_vocab is an optional category that is broken out during scoring; it
    may be anything but usually consists of the word types seen in the
    supervised part of the corpus."""

    counts: Counter[Tuple[str, str]] = Counter()
    for ((word, tag), (goldword, goldtag)) in zip(predicted, gold):
        assert word == goldword or word == OOV_WORD   # sentences being compared should have the same words!
        if word is BOS_WORD or word is EOS_WORD:  # not fair to get credit for these
            continue
        if goldtag is None:                # no way to score if we don't know answer
            continue
        
        if word == OOV_WORD:                      category = 'NOVEL'
        elif known_vocab and word in known_vocab: category = 'KNOWN'
        else:                                     category = 'SEEN'    

        for c in (category, 'ALL'):
            counts['DENOM', c] += 1      # denominator of accuracy in category c
            if tag == goldtag:
                counts['NUM', c] += 1    # numerator of accuracy in category c

    return counts

def write_tagging(model_or_tagger: Union[HiddenMarkovModel, Callable[[Sentence], Sentence]],
                        eval_corpus: TaggedCorpus,
                        output_path: Path) -> None:
    if isinstance(model_or_tagger, HiddenMarkovModel):
        tagger = viterbi_tagger(model_or_tagger, eval_corpus)
    else:
        tagger = model_or_tagger
    with open(output_path, 'w') as f:
        for gold in tqdm(eval_corpus, total=len(eval_corpus)):
            predicted = tagger(gold.desupervise())
            f.write(str(predicted)+"\n")

def compare_models(
        model1: HiddenMarkovModel,
        model2: HiddenMarkovModel,
        eval_corpus: TaggedCorpus,
        known_vocab: Optional[Integerizer[Word]] = None) -> None:
    """Compare two models on a development corpus and log differences where
    the second model makes corrections that the first model missed."""

    counts_fixed = 0
    counts_total = 0
    corrections = []

    for sentence in tqdm(eval_corpus, total=len(eval_corpus)):
        predicted1 = model1.viterbi_tagging(sentence.desupervise(), eval_corpus)
        predicted2 = model2.viterbi_tagging(sentence.desupervise(), eval_corpus)

        for idx, ((word, tag1), (word, tag2), (gold_word, gold_tag)) in enumerate(zip(predicted1, predicted2, sentence)):
            if word != gold_word:
                continue  # Skip OOV cases and BOS/EOS

            if tag2 == gold_tag and tag1 != gold_tag:
                counts_fixed += 1
                corrections.append({
                    'word': word,
                    'position': idx,
                    'gold_tag': gold_tag,
                    'tag_model1': tag1,
                    'tag_model2': tag2,
                    'context': sentence[max(0, idx - 2):idx + 3]
                })

            counts_total += 1

    # Log results
    log.info(f"Total corrections made by model2: {counts_fixed}/{counts_total}")
    for correction in corrections[0: 10]:
        log.info(f"Correction: Word '{correction['word']}' at position {correction['position']}")
        log.info(f"   Gold Tag: {correction['gold_tag']}, Model1 Tag: {correction['tag_model1']}, Model2 Tag: {correction['tag_model2']}")
        log.info(f"   Context: {[w for w, t in correction['context']]}")

    return counts_fixed / counts_total  # percentage of corrected errors