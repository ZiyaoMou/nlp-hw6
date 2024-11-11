# NLP Homework 6: Structured Prediction

## Setup and Files

As in previous homeworks, you can activate the environment anytime using 

    conda activate nlp-class

We have provided you with starter code.  If you choose to use this code,
**you should study it first**, so that you understand it as if you had written it
yourself.  It will help a lot to experiment with it in a notebook, 
constructing small examples of the classes and calling their methods.
(Also a good way to learn PyTorch!)

After reading the reading handout, you probably want to study the files in this order.
**Boldface** indicates parts of the code that you will write.  

* `integerize.py` -- converts words and tags to ints that can be used to index PyTorch tensors (we've used this before)
* `corpus.py` -- manage access to a corpus (compare `Probs.py` on the lm homework)
* `hmm.py` -- HMM parameters, **Viterbi algorithm**, **forward-backward algorithm**, **M-step** training
* `eval.py` -- measure tagging accuracy
* `test_ic.py` or `test_ic.ipynb` -- uses the above to test Viterbi tagging, supervised learning, unsupervised learning on ice cream data
* `test_en.py` or `test_en.ipynb` -- uses the above to train on larger English data and evaluate on held-out cross-entropy and tagging accuracy
* `tag.py` -- your command-line system (**you might add arguments for extra credit**)
* `crf.py` -- CRF **parameters**, **conditional log-probabilities**, **gradients**, **gradient step** training

You can experiment with these modules at the Python prompt.
For example:

    from pathlib import Path
    from corpus import *
    c = TaggedCorpus(Path("icsup"))
    c.tagset
    list(c.tagset)
    list(c.vocab)
    iter = c.get_sentences()
    next(iter)
    next(iter)

## What To Do Overall

The starter code has left sections for you to implement.  

**The easiest way to approach this assignment** is to open up the
`test_ic.ipynb` notebook and start running it.  There will be pieces that raise
`NotImplementedError` because you haven't implemented them yet.  So, keep
implementing what you need until you can get through the HMM part of that
notebook.  You can check in the notebook that you are successfully reproducing
computations from the [ice cream
spreadsheet](http://cs.jhu.edu/~jason/465/hw-tag/hmm.xls) that we covered in
class.

Then move on to the `test_en.ipynb` notebook.  Finally, go back to
both notebooks and do their CRF parts.

As you work, also answer the questions in the assignment handout.  You can do
this by trying additional commands in the notebook.

(If you don't like Jupyter notebooks, you can also work directly at the Python prompt by copy-pasting short code blocks from `test_ic.py` and typing your own commands, or by running `test_ic.py` or your own scripts.)

Your deliverables are *written answers to the questions*, plus your completed
versions of the *Python scripts* above.  Do not separately hand in notebooks or
printouts of your interpreter session.

## Steps

Below, we'll give you some hints on what you'll need to do as you work
through the notebooks.

*Note:* You should probably execute the following near the top of each
notebook:

   %load_ext autoreload
   %autoreload 2

That ensures that if you edit `hmm.py` (or any other module), the
notebook will notice the update.  Specifically, it will re-import
symbols, such as `HiddenMarkovModel`.  If you *already* executed `m =
HiddenMarkovModel(...)`, then the `m` object doesn't change -- it's
still an element of the old `HiddenMarkovModel` class.  But if you
re-execute that line after editing the class, it will now call the
constructor of the updated `HiddenMarkovModel` class

### Step 0: Create an HMM

  You can hard-code the initial spreadsheet
parameters into your HMM, something like this:

    hmm = HiddenMarkovModel(...)
    hmm.A = torch.tensor(...)   # transition matrix
    hmm.B = torch.tensor(...)   # emission matrix

This is illustrated in `test_ic`. 

If you want to experiment with the parameters later as we did in class, you can set individual elements of a `Tensor` the way you'd expect:

    my_tensor[3, 5] = 8.0

Think about how to get those indices, though. (Where in `hmm.B` is the
parameter for emitting `3` while in state `H`?) You may want to use the
corpus's `integerize_tag` and `integerize_word` functions or access the
integerizers directly.

### Step 1: Viterbi Tagging

Implement the `viterbi_tagging()` method in `hmm.py` as described in the handout.
Structurally, this method is very similar to the forward algorithm.
It has a handful of differences, though:

* You're taking the max instead of the sum over possible predecessor tags.
* You must track backpointers and reconstruct the best path in a backward pass.
* This function returns a sentence tagged with the highest-probability tag sequence,
  instead of returning a (log-)probability.

Remember to handle the BOS and EOS tags appropriately.  Each will
be involved in 1 transition but 0 emissions.

You may benefit from [PyTorch's tutorial](https://pytorch.org/tutorials/beginner/nlp/pytorch_tutorial.html) and from carefully reading the reading handout and the commented starter code.

At this point, you're just going to be running on the small ice cream
example, so you should not yet have to use the tricks in the
"Numerical Issues" section of the reading handout.  The spreadsheet
didn't have to use them either.

Run your implementation on `icraw` (the untagged diary data from the spreadsheet), using the hard-coded parameters
from above.  To do this, look at how `test_ic` calls `viterbi_tagging`.

Check your results against the
[Viterbi version of the spreadsheet](http://cs.jhu.edu/~jason/465/hw-tag/hmm-viterbi.xls).
(_Note:_ Backpointers are hard to implement on spreadsheets, so the spreadsheet uses an alternative technique
to get the Viterbi tagging, namely to symmetrically compute $\hat{\alpha}$ and $\hat{\beta}$
values at each state; these are called $\mu$ and $\nu$ on the spreadsheet.  Their product
is the probability of the *best* path through each state, just as the product $\alpha \cdot \beta$ in
the forward algorithm is the total probability of all *paths* through each state.  That's
just a change of semiring from + to max.)

Do your $\hat{\alpha}$ values match the spreadsheet, for each word, if you print them out along the way?
When you follow the backpointers, do you get either `HHHHHHHHHHHHHCCCCCCCCCCCCCCHHHHHH`
or `HHHHHHHHHHHHHCCCCCCCCCCCCCHHHHHHH` (these paths tie, with the
only difference being on day 27)?
(These are rhetorical questions. The only questions you need to turn in answers to
are in the handout.)

Try out automatic evaluation: compare your Viterbi tagging to the correct answer
in `icdev`, using an appropriate method from `eval.py`.  Again, `test_ic` will
guide you through this.  

### Step 2: Supervised training

The next part of `test_ic` asks you to start from random parameters and do supervised training on the `icsup` dataset, which is a more realistic way to get to the initial parameters on the spreadsheet.

So, you will have to implement at least part of the EM algorithm.  However, because you are working on supervised data, you don't actually have to impute the tags at the E step, so you can replace `E_step()` for now with a simpler version that just runs through the sentence and grabs the counts.  The `M_step()` should be the real thing, but we've written part of it for you to show you the way.

### Step 3: The forward algorithm

To compute the log-probability of `icraw` under these initial parameters (log of
formula (1) on your reading handout), you'll need to implement `forward_pass()`,
which sums over paths using the forward algorithm, instead of maximizing over
paths. This should match the results on [the original
spreadsheet](http://cs.jhu.edu/~jason/465/hw-tag/hmm.xls), and you can compare
the intermediate results to catch any bugs.

### Step 4: Full E step

You now want to run `train()` on the *unsupervised* `icraw` corpus. In general,
this method locally maximizes the (log-)likelihood of the parameters, using the
EM algorithm. The weights are updated after each epoch (iteration across the
dataset).  The `train()` method decides when to stop (see its documentation).

For this, you'll need to fully implement `E_step()`.  You can call the forward
algorithm that you just implemented.  You'll also have to add the backward
algorithm and make it add to the expected counts stored in the
`HiddenMarkovModel` instance.

In particular, you'll need to implement the backward algorithm in `backward_pass()`, which also accumulates expected counts.

You can test your `E_step()` directly in the notebook or the debugger.  Your
implementation should get the same results as before on `icsup` (maybe a little
more slowly), because it restricts to observed tags when available, but now it
should also work on `icraw`.  In the latter case, you can check your backward
probabilities and expected counts against the spreadsheet.

### Step 5: EM training

Once your E step is working, `train()` should exactly mimic the iterations on
the spreadsheet.  Do you eventually get to the same parameters as EM does?

### Step 6: Make it fast

Now let's move on to real data!  Try out the workflow in `test_en.py`
(or its notebook version `test_en.ipynb`).

When training on a supervised corpus `ensup`, our own implementation ran at
roughly 25 training sentences per second on a good 2019 laptop.  (This is for
both supervised and unsupervised sentences, although you could get a speedup on
supervised sentences by handling them specially as you did earlier.)  

This is reported as iterations per secion (`it/s`) on the `tqdm` progress bar
during training.  Note that there are also progress bars for periodic evaluation
on the smaller dev corpus.

If you're notably slower than this, you'll want to speed it up -- probably by
making less use of loops and more use of fast tensor operations.

*Note:* Matrix multiplication is available in PyTorch (and numpy)
using the `matmul` function.  In the simplest case, it can be invoked
with the [infix operator](https://www.python.org/dev/peps/pep-0465/)
`C = A @ B`, which works the way you'd expect from your linear algebra
class.  A different syntax, `D = A * B`, performs _element-wise_
multiplication of two matrices whose entire shapes match.  (It also
works if they're "[broadcastable](https://pytorch.org/docs/stable/notes/broadcasting.html),"
like if `A` is 1x5 and `B` is 3x5.  See also [here](https://numpy.org/doc/stable/user/basics.broadcasting.html).)

### Step 7: Make it stable

You'll probably come across numerical stability problems from working with 
products of small probabilities.  Fix them using one of the methods
in the "Numerical Issues" section of the reading handout.

### Step 8: Check the command-line interface

We've provided a script `tag.py`.  Run it with the `--help` option to see documentation, or look at the code.

It can be run (for example) like this:

    $ python3 tag.py <input_file> --model <model_file> --train <training_files>

This should run an HMM on the `input_file`.
Where does the HMM come from?  It is loaded from the `model_file`
and then trained further on the `training_files` until the error rate metric is
no longer improving on the `input_file`.  The improved model is saved back to the `model_file` at the 
end.

If the `model_file` doesn't exist yet or isn't provided, then the script will 
create a new randomly initialized HMM.  If no `training_files` are provided, 
then the model will not be trained further.

Thus, our autograder will be able to replicate roughly the `test_en.py`
workflow like this:

    $ python3 tag.py endev --model example.pkl --train ensup        # supervised training
    $ python3 tag.py endev --model example.pkl --train ensup enraw  # semi-supervised training

and it then will be able to evaluate the error rate of your saved model on a test file like this:
  
    $ python3 tag.py ensup --model example.pkl --loss viterbi_error  # error rate on TRAINING data
    $ python3 tag.py endev --model example.pkl --loss viterbi_error  # error rate on DEVELOPMENT data

If the test file is untagged, then it has no way to evaluate error
rate, but it can still output a tagging (and evaluate cross-entropy on
the words).  That's how you would apply your tagger to actual new input.

    $ python3 tag.py enraw --model example.pkl

Of course, to get a fair score, the autograder will use blind test data.  The `endev` sentences were already seen during development (used for purposes such as hyperparameter tuning and early stopping).

`tag.py` should also output the Viterbi taggings of all sentences in the `eval_file`
to a text file, in the usual format.  For example, if the `eval_file` is called
`endev`, then it should create a file called `endev_output` with lines like
    
    Papa/N ate/V the/D caviar/N with/P a/D spoon/N ./.

`tag.py` is allowed to print additional text to the standard error stream, e.g.,
by using Python's `logging` library. This can report other information that you
may want to see, including the tags your program picks, its perplexity and
accuracy as it goes along, various probabilities, etc. Anything printed to
standard error will be ignored by the autograder; use it however you'd like.

There are other command-line parameters, such as $\lambda$ for add-$\lambda$ smoothing
You're entirely welcome (and encouraged) to add other command line parameters.
 hyperparameter searching much easier; you can write a script
that loops over different values to automate your search. You may also be able
to parallelize this search. Make sure that your submitted code has default
values set how you want them, so that we run the best version.  (Don't make the
autograder run your hyperparameter search.)

### Step 9: Posterior Decoding (**extra credit**)

Try implementing posterior decoding as described in the reading handout. Since
you've already implemented the forward-backward algorithm, this shouldn't be too
much extra work.  But you will need to add a method for this (or add options to
existing methods).  You should also extend `tag.py` with an option that lets you choose posterior decoding rather than Viterbi decoding.

The posterior marginal probabilities for the `icraw` data are shown on
the spreadsheet.  You can use these to check your code.

On the English data, how much better does posterior decoding do, in terms of
accuracy? Do you notice anything odd about the outputs?

### Step 10: CRF

You can now go back to the ice cream notebook (`test_ic`) and complete the
the `ConditionalRandomField` class in `crf.py`.

Instead of re-estimating the parameters only after each epoch (M step), the
`train` method has been overridden to make an SGD update after each minibatch.
This outer loop has been given to you, but make sure to study it.

You'll mostly focus on setting and adjusting the parameters via SGD updates. You
shouldn't need any new dynamic programming -- you can just inherit those methods
from your `HiddenMarkovModel` class.

Only the model is changing.  So the supporting code like `corpus.py` and
`eval.py` should not have to change.  

Once you've been able to finish `test_ic` successfully, go back to `test_en`.
Also make sure that `tag.py` works properly, using the `--crf` option.

## What to submit [this section subject to change]
You should submit the following files under **Assignment 6 - Programming**:

* Code you changed
    - `hmm.py`
    - `crf.py`

* Supporting code you probably didn't change
    - `tag.py`
    - `eval.py`
    - `corpus.py`
    - `integerize.py`

* Trained models
    - `ensup_hmm.pkl` (english supervised HMM)
    - `entrain_hmm.pkl` (english semi-supervised HMM)
    - `entrain_hmm_awesome.pkl` (english semi-supervised HMM with extra credit improvements)
    - `ensup_crf.pkl` (english supervised CRF)
    - `entrain_crf.pkl` (english semi-supervised CRF)

Try your code out early as it can take a bit of time to run the autograder. Please let us know if anything is broken in the autograder.

*Additional Note:* Please don’t submit the output files that show up in the autograder’s feedback message. Rather, these will be produced by running your code! If you do submit them, the autograder will not grade your assignment.
