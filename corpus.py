#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Module for working with a corpus of tagged sentences, in a particular format.

# This code is adapted from Probs.py in the language modeling homework.
# The main difference is that now we are dealing with tags as well as words.
# Also, we've encapsulated this as a corpus of tagged sentences.
# Also, we support integerization.

# TODO: It would be nice to add support for weighting the examples.
# For example, sentences from the supervised file could be returned
# with a higher weight or sampled more often, so that they are more
# important in the objective.

import logging
from pathlib import Path
##### TYPE DEFINITIONS (USED FOR TYPE ANNOTATIONS)
from typing import Counter, Iterable, Iterator, List, NewType, Optional, Tuple
from more_itertools import peekable
from integerize import Integerizer

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

Word = NewType('Word', str)          # subtype of str
Tag = NewType('Tag', str)            # subtype of str
TWord = Tuple[Word, Optional[Tag]]   # a (word, tag) pair where the tag might be None

IntegerizedTWord = Tuple[int, Optional[int]]
IntegerizedSentence = List[IntegerizedTWord]

# Special words and tags
OOV_WORD: Word = Word("_OOV_")
BOS_WORD: Word = Word("_BOS_WORD_")
EOS_WORD: Word = Word("_EOS_WORD_")
BOS_TAG: Tag = Tag("_BOS_TAG_")
EOS_TAG: Tag = Tag("_EOS_TAG_")

# Seed the random number generator consistently, for the sake of
# draw_sentences_forever.
import random
random.seed(1234)

class Sentence(List[TWord]):
    def __init__(self, sentence: Optional[List[TWord]] = None):
        if sentence:
            super().__init__(tword for tword in sentence)
        else:
            super().__init__()

    def __str__(self) -> str:
        return " ".join([word if tag is None else f"{word}/{tag}" for (word, tag) in self[1:-1]])

    def desupervise(self):
        """Make a new version of the sentence, with the tags removed 
        except for BOS_TAG and EOS_TAG."""
        sentence = Sentence()
        [sentence.append(
            (word, tag if tag == BOS_TAG or tag == EOS_TAG else None)) for word, tag in self]
        return sentence

    def is_supervised(self) -> bool:
        """Is the given sentence fully supervised?"""
        return all(tag is not None for _, tag in self)


class TaggedCorpus:
    """Class for a corpus of tagged sentences.
    This is read from one or more files, where each sentence is 
    a single line in the following format:
        Papa/N ate/V the/D caviar/N with/P a/D spoon/N ./.
    Some or all of the tags may be omitted:
        Papa ate the caviar with a spoon.

    The tagset and vocab attributes are publicly visible integerizers.
    The objects that we return from the corpus will use strings, but 
    we provide utility functions to run them through these integerizers.
    """

    def __init__(self, *files: Path,
                 tagset: Optional[Integerizer[Tag]] = None, 
                 vocab: Optional[Integerizer[Word]] = None,
                 vocab_threshold: int = 1, 
                 add_oov: bool = True):
        """Wrap the given set of files as a corpus. 
        Use the tagset and/or vocab from the parent corpus, if given.
        Otherwise they are derived as follows from the data in `files`:

        The tagset consists of all tags that have appeared at least once.
        Words must appear at least vocab_threshold times to be in the vocabulary.
        We only include OOV in the vocabulary if the corpus includes any OOV words, or if add_oov is True.

        We include EOS and BOS words and tags.
        But note that in an HMM model, only EOS_TAG is an event that is randomly generated.
        And in a CRF model, none of these are randomly generated.
        So, we include them at the end of the tagset so that they can be easily omitted.
        """

        super().__init__()
        self.files = files

        if tagset is None or vocab is None:
            # Read the corpus to harvest the tagset and vocabulary
            self.tagset: Integerizer[Tag] = Integerizer()
            self.vocab: Integerizer[Word] = Integerizer()
        
            word_counts: Counter[Word] = Counter()
            for word, tag in self.get_tokens(oovs=False):
                if word == EOS_WORD:      # note: word will never be BOS_WORD
                    continue              # skip EOS for purposes of getting vocab
                word_counts[word] += 1    # count words to see later if we pass the threshold
                if tag is not None:
                    self.tagset.add(tag)  # no threshold for tags
            log.info(f"Read {sum(word_counts.values())} tokens from {', '.join(file.name for file in files)}")

            for word, count in word_counts.items():
                if count >= vocab_threshold:
                    self.vocab.add(word)
                else:
                    self.vocab.add(OOV_WORD)

            if add_oov:
                self.vocab.add(OOV_WORD)
            self.tagset.add(EOS_TAG)
            self.tagset.add(BOS_TAG)
            self.vocab.add(EOS_WORD)
            self.vocab.add(BOS_WORD)

        # Install any tagset and/or vocab that were provided as arguments
        # (may overwrite the one that was computed above).
        if tagset is None:
            log.info(f"Created {len(self.tagset)} tag types")
        else:
            self.tagset = tagset

        if vocab is None:
            log.info(f"Created {len(self.vocab)} word types")
        else:
            self.vocab = vocab

        # cache this value (maybe None) so we don't have to keep looking it up
        self.oov_w = self.vocab.index(OOV_WORD)

        # check that installed tagset and vocab (if any) will suffice
        if tagset is not None or vocab is not None:
            for word, tag in self.get_tokens(oovs=False):
                self.integerize_word(word)  # make sure it doesn't throw exception with given vocab
                if tag is not None:
                    self.integerize_tag(tag)  # make sure it doesn't throw exception with given tagset

    def __str__(self) -> str:
        return "\n".join(str(sentence) for sentence in self)

    # Methods for reading the corpus.
    # We return non-integerized versions to make debugging easier;
    # the caller can integerize them using utility methods that we also provide.
    # 
    # (But this design is a bit inefficient, since it re-integerizes the same example
    # each time we visit it during SGD.)

    def __iter__(self) -> Iterator[Sentence]:
        """Iterate over all the sentences in the corpus, in order."""
        return iter(self.get_sentences())

    def __len__(self) -> int:
        """Number of sentences in the corpus."""
        self._num_sentences: int
        try:
            return self._num_sentences
        except AttributeError:
            self._num_sentences = sum(1 for _ in self)
            return self._num_sentences

    def num_tokens(self) -> int: 
        """Number of tokens in the corpus, including EOS tokens."""
        self._num_tokens: int
        try:
            return self._num_tokens
        except AttributeError:
            self._num_tokens = sum(1 for _ in self.get_tokens())
            return self._num_tokens

    def get_tokens(self, oovs: bool = True) -> Iterable[TWord]:
        """Iterate over the tokens in the corpus.  Tokens are whitespace-delimited.
        If oovs is True, then words that are not in vocab are replaced with OOV.
        There is no BOS token, but each sentence is terminated with EOS."""
        for file in self.files:
            with open(file) as f:
                for line in f:
                    for token in line.split():
                        word: Word
                        tag: Optional[Tag]                    # declare type to help the type checker
                        if "/" in token:
                            w, t = token.split("/")   
                            word, tag = Word(w), Tag(t)       # for example, "caviar/Noun"
                        else:
                            word, tag = Word(token), None     # for example, "caviar" without a tag
                        if (not oovs) or word in self.vocab:
                            yield word, tag       # keep the word
                        else:
                            yield OOV_WORD, tag   # replace this out-of-vocabulary word with OOV
                    yield EOS_WORD, EOS_TAG  # Every line in the file implicitly ends with EOS.

    def get_sentences(self) -> Iterable[Sentence]:
        """Iterable over the sentences in the corpus.  Each is padded to include BOS and EOS.

        (The padding is ugly to have as part of the TaggedCorpus class, because these
        symbols are not really part of the sentence.  We put the padding in just because
        it's convenient for the particular taggers we're writing, and matches the notation
        in the handout.)"""

        sentence = Sentence([(BOS_WORD, BOS_TAG)])
        for word, tag in self.get_tokens():
            sentence.append((word, tag))
            if word == EOS_WORD:
                yield sentence
                # reset for the next sentence (if any)
                sentence = Sentence([(BOS_WORD, BOS_TAG)])


    def draw_sentences_forever(self, randomize: bool = True) -> Iterable[Sentence]:
        """Infinite iterable over sentences drawn from the corpus.  We iterate over
        all the sentences, then do it again, ad infinitum.  This is useful for 
        SGD training.  
        
        If randomize is True, then randomize the order of the sentences each time.  
        This is more in the spirit of SGD, but it forces us to keep all the sentences 
        in memory at once.  (Note: This module seeds the random number generator
        so at least the randomness will be consistent across runs.)
        """
        sentences = peekable(self.get_sentences())
        assert sentences      # there should be at least one sentence.  (This test uses peekability.)
        if not randomize:
            import itertools
            return itertools.cycle(sentences)  # repeat forever
        else:
            pool = tuple(sentences)
            while True:
                for sentence in random.sample(pool, len(pool)):
                    yield sentence


    # Utility methods for integerizing the objects that are returned above.

    def integerize_tag(self, tag: Tag) -> int:
        t = self.tagset.index(tag)
        if t is None:
            raise KeyError(tag, self, "This tag is not in the tagset of this corpus, and we don't support OOV tags")
        return t

    def integerize_word(self, word: Word) -> int:
        w = self.vocab.index(word)
        if w is None: 
            w = self.oov_w
            if w is None:   # OOV_WORD needs to be in our tagset
                raise KeyError(word, self, "This word is not in the vocab of this corpus, and can't be converted to OOV since OOV is also missing from the vocab")
        return w

    def integerize_tword(self, tword: TWord) -> IntegerizedTWord:
        word, tag = tword
        return self.integerize_word(word), (None if tag is None else self.integerize_tag(tag))

    def integerize_sentence(self, sentence: Sentence) -> IntegerizedSentence:
        return [self.integerize_tword(tword) for tword in sentence]
