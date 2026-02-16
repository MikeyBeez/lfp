MCR2 as Language Model Regularizer: What Works, What Doesn't, and Why the Theory Oversells It

We set out to test whether Maximal Coding Rate Reduction, the information-theoretic objective from Yi Ma's lab, could replace or augment cross-entropy in training a small transformer language model. The theory promises geometrically structured representations: different classes pushed onto incoherent subspaces, same-class representations compressed together. It's elegant. It has clean mathematical foundations. And after a stretch of experiments and seven ablation runs, we can say it does something useful, but not for the reasons the theory claims. The short version: MCR2 works because it forces the model to satisfy two conflicting objectives at once, and that conflict prevents overfitting. Neither the expansion term nor the compression term helps alone — each one alone makes things worse. It's the tension between them that regularizes.

This is open notebook science. We're reporting what worked, what didn't, what surprised us, and what we learned about the gap between theoretical elegance and empirical reality.


The parameter-to-token ratio problem

We spent three days convinced MCR2 wasn't working. Every experiment showed the same thing: cross-entropy dominated, MCR2 contributed nothing, and all three training regimes — CE only, MCR2 only, and the combined "organism" approach — produced nearly identical results. We blamed the loss function. We blamed the implementation. We rewrote the MCR2 computation twice.

The real problem was our model. We were training a 44.6 million parameter transformer on WikiText-2, which has roughly 2.4 million tokens. That's a 19:1 parameter-to-token ratio. The model memorized the training set so completely that no auxiliary loss could influence anything. Training perplexity dropped below 2.0 for all regimes. The model had enough capacity to perfectly fit the data through any loss landscape. We were trying to measure the effect of a regularizer on a model that didn't need to generalize.

Switching to a small model — d_model of 192, 4 layers, 4 attention heads, feed-forward dimension of 768, about 1.77 million non-embedding parameters — changed everything. That's a 0.74:1 parameter-to-token ratio. The model can no longer memorize its way out. For the first time, we saw real training dynamics: different loss functions producing different representation geometries, different generalization behavior, different attention patterns.

The lesson cost us three days and it has nothing to do with MCR2. Experimental infrastructure matters more than the thing you're testing. If your experimental setup can't distinguish between hypotheses, no amount of theoretical sophistication in your loss function will help. Most negative results in deep learning research are probably just misconfigured experiments.


The result

Cross-entropy alone reaches a validation perplexity of 315.2. Adding MCR2 as an auxiliary loss reaches 306.4. That's a 2.8 percent improvement in generalization.

The training perplexity tells the opposite story: CE alone reaches 19.5, while the combined approach only reaches 21.1. The model with MCR2 fits the training data slightly worse but generalizes slightly better. That's exactly what a regularizer should do. It constrains the model, prevents it from fully exploiting the training distribution, and that constraint turns out to be useful when you evaluate on held-out data.

The representation metrics tell a consistent story. Cosine similarity between hidden states — a direct measure of representation collapse — is 0.051 for the combined approach versus 0.054 for CE alone. Lower is better; it means representations are more spread out and using more of the available space. Attention entropy, which measures how distributed attention patterns are across tokens, is 2.457 versus 2.364. Higher means attention is less concentrated on single tokens, fewer attention sinks, healthier information flow. These differences are small but they all point the same direction: MCR2 is doing something real to representation geometry.

Both models overfit dramatically. Training perplexity around 20 with validation perplexity around 300 is a 15x gap. This is a data-starved regime, exactly where you'd expect a regularizer to have visible effect. We don't know if the improvement holds at more favorable parameter-to-data ratios. It might shrink. It might grow. That's a future experiment.


What MCR2 actually does, stated honestly

Strip away the information-theoretic language and MCR2 is a geometrically motivated regularizer built on log-determinants of covariance matrices. It needs a notion of "class" — which representations should be grouped together and which should be pushed apart. In language modeling, the natural choice is the next token. Every position in the sequence belongs to the class of whatever word comes next. So if positions 7, 43, and 128 in a batch all precede the word "the," their hidden states are in the same class. If position 12 precedes "dog," it's in a different class.

Given these classes, MCR2 does two things. It pushes representations of tokens with different next-tokens apart by maximizing the log-determinant of the full batch covariance — this is the expansion term. It pulls representations of tokens with the same next-token together by minimizing the log-determinant of each class's covariance — this is the compression term.

The expansion term prevents representation collapse. If all hidden states converge to the same region of space, the covariance matrix becomes low-rank and its determinant drops. Penalizing this keeps the representation space open. The compression term encourages tokens that predict the same next word to cluster, which sounds like a reasonable inductive bias for language modeling — but it has a problem we'll come back to. The same next token can follow wildly different contexts. "The" follows subjects, verbs, prepositions, and sentence boundaries. Compressing all pre-"the" representations together means asking the model to make unrelated contexts look similar, which fights against exactly the contextual sensitivity that makes transformers work.

Here's the mechanics in plain language, for readers who want to understand what the loss function actually computes. Take all the hidden state vectors in a batch and stack them into a matrix Z. Multiply Z-transpose by Z and you get a square matrix where each entry measures how correlated two dimensions of the representation space are. This is the covariance matrix. Now take its determinant. The determinant measures the volume of the space that the representations occupy — if they're all clustered in a narrow region, the volume is small; if they're spread across many dimensions, the volume is large. Take the log of the determinant to keep the numbers manageable. That's the coding rate: a single number that says how much of the available representation space the model is actually using.

The expansion term maximizes this for all tokens together — spread out, use the whole space. The compression term minimizes it for each class separately — tokens predicting the same next word should cluster into a small volume. The difference between the two is the rate reduction: how much more space the full population uses compared to the individual classes. Maximizing rate reduction means maximizing between-class spread while minimizing within-class spread.

You could describe all of this without ever mentioning coding rates, bits, or information theory. It's a covariance regularizer with a separation objective. That's a legitimate contribution. The MCR2 framework led us to a loss function that works — our disagreement is with why it's claimed to work, not with the result. Yi Ma's group identified a functional form that's genuinely useful for regularizing representations. Covariance regularization with a separation objective, compatible with next-token prediction as a primary loss, is a real result. The information-theoretic framing is not necessary to explain why it works.


The seven ablations

After establishing that full MCR2 beats CE-only, we systematically dismantled it. MCR2 has two terms — expansion and compression. The theory says both are necessary: expansion keeps representations rich, compression groups them into clean subspaces, and the difference between the two is the "rate reduction" that the framework is named after. We tested what happens when you remove each piece.

Expansion-only was our first ablation, and it was the worst run we'd done. Validation perplexity of 335.5, worse than CE-only's 315.2, much worse than full MCR2's 306.4. It had the lowest training perplexity of any run at 19.4 — the model memorized harder, not less. Opening up the representation space without constraining it gave the model more room to overfit.

The early training trajectory was seductive. At step 4000, expansion-only had the best validation perplexity of any run. It was learning fast, spreading representations out, making rapid progress. But that advantage was speed, not generalization. By step 7000 it had crossed over CE-only and kept getting worse. The extra representational capacity was being used to memorize.

We tried selective expansion: compute class centroids — one per unique next-token — and expand those instead of individual tokens. This is frequency-neutral. "The" with 200 occurrences and "algebra" with 2 occurrences contribute equally. Selective expansion ended at 324.2. Better than blind expansion, still worse than CE-only.

Then curriculum approaches. Expansion for the first 2000 steps to build good geometry while the model is plastic, then full MCR2 for the remaining 8000 to constrain against overfitting. It ended at 317.2, slightly worse than CE-only. Alternating phases — cycling between expansion and full MCR2 every 2000 steps — gave 322.1. Same story.

At this point we had a clean narrative: compression is the regularizer. Every variant that removed or reduced compression time did worse. More compression time equals better generalization. The obvious next experiment was compression-only — skip the expansion term entirely and just minimize per-class coding rates.

Our collaborator predicted compression-only would match or beat full MCR2 at lower cost. If true, that would be devastating to the rate reduction story: you'd literally delete half the theory and performance stays. We ran it.


The result that changed the story

Compression-only was the most interesting run we'd done, and the most surprising.

At step 2000, it had the best validation perplexity we'd ever seen: 187.6. Not even close to anything else. The separation ratio was 1.29, the highest we'd measured. The model was learning fast, compression was constraining representations effectively, everything looked perfect.

Then it started overfitting. Validation perplexity at step 3000: 197.2. Step 4000: 224.2. Step 5000: 251.7. Step 7000: 300.7. Step 10000: 317.2. The final number is worse than CE-only and essentially identical to the curriculum run.

Training perplexity told the same story as expansion-only but from the other direction. It kept dropping — 20.8 at convergence, lower than full MCR2's 21.1 — meaning compression-only memorized the training set more than full MCR2 did. Without expansion pushing representations apart, compression collapsed them into tight per-token clusters. Those clusters were too specialized to generalize.

This overturned our earlier narrative. Compression alone doesn't regularize. Expansion alone doesn't regularize. Each term, isolated, adds a different kind of capacity that the model uses to overfit. Expansion opens up representational space — more room to memorize. Compression creates tight clusters — more specific memorization of training patterns. Only when both are active simultaneously does the model face an unsatisfiable conflict: spread out AND cluster. That conflict is the regularization.

Here are the final validation perplexities, all seven runs, same model, same data, same training duration: full MCR2, 306.4. CE-only, 315.2. Curriculum, 317.2. Compression-only, 317.2. Alternating, 322.1. Selective expansion, 324.2. Expansion-only, 335.5.

Full MCR2 wins by a clear margin. Every other variant lost.


Tension is the regularizer

This was the opposite of what we expected at every stage.

First we expected expansion to be the useful part — anti-collapse as the mechanism. Expansion-only was the worst run. Then we expected compression to be the useful part — constraint as the mechanism. Compression-only peaked early and overfitted. The answer is that neither term alone is a regularizer. Each term alone gives the model a way to reduce its training loss without improving generalization. Expansion does this by spreading representations out, making more dimensions available for memorization. Compression does this by collapsing same-class representations into tight clusters, over-specializing to training patterns.

When both terms are active, they create an impossible demand: make all representations occupy a large volume, AND make same-class representations occupy a small volume. The model can't fully do both. The partial failure to satisfy both constraints simultaneously is what prevents overfitting. The model is forced toward representations that partially satisfy both objectives, and those turn out to be more general than representations that fully satisfy either one.

This is how most regularizers work. Dropout destroys information. Label smoothing lies about targets. Weight decay biases toward small norms for no principled reason. None of these are "correct" — they all make the training task slightly harder than it needs to be, and the model finds more robust solutions as a result. MCR2's contribution is a particularly structured form of conflicting pressure: geometric tension in representation space, rather than information destruction or target corruption.

Full MCR2's training perplexity of 21.1 — the worst of any run — is both terms preventing the model from fully fitting the training data. That prevention is the regularization.


The epsilon problem

The coding rate formula at the heart of MCR2 is R(Z) equals d over 2 times log-det of I plus d over n times epsilon squared times Z-transpose-Z, where d is the representation dimension, n is the number of tokens, epsilon is a "precision parameter," and Z is the matrix of representations.

Look at the fraction d over n times epsilon squared. In any given experiment, d and n are constants. The representation dimension is fixed by architecture. The number of tokens per batch is fixed by batch size and sequence length. These are known quantities that you set before training starts. They add no information. The entire fraction reduces to a single scalar: one over epsilon squared, multiplied by a constant. The researcher chooses epsilon. Everything else is determined.

This matters because the theoretical claim is that the coding rate measures something intrinsic about the representations — how many bits you'd need to encode them to a given precision. But if the "precision" is a free parameter chosen by the researcher, the measurement is relative to an arbitrary choice. You can make the coding rate say anything you want by adjusting epsilon.

If the theory truly determined the objective, epsilon would be determined by the data. It would be a function of the noise level, or the signal-to-noise ratio, or some other measurable property of the representations. Instead, it's a hyperparameter. We set it to 0.5 because the original papers use values in that range. We could set it to 0.1 or 5.0 and get qualitatively different behavior.

What you actually have is a log-det regularizer with a tuning parameter. That's fine. Weight decay has a tuning parameter. Dropout has a tuning parameter. But weight decay doesn't claim to be an intrinsic measure of representation geometry. The gap between what MCR2 claims to be and what it operationally is — that gap matters if you're trying to build on the theory, derive architectures from it, or use it to make predictions about what should work. We're not arguing the math is invalid. We're arguing that the information-theoretic interpretation is not necessary to explain the empirical effect, and our ablations suggest a simpler explanation: conflicting geometric pressure regularizes by imposing constraints the model can't fully satisfy.


The frequency problem

Here's a problem the MCR2 literature doesn't discuss. In language modeling, "classes" are defined by the next token. Common words like "the," "of," "and," and "to" appear hundreds of times per batch. Each of these becomes a large class in the MCR2 computation. The compression term — which tries to make same-class representations similar — operates most strongly on these frequent tokens because they have the largest within-class covariance matrices and contribute the most to the loss.

But these are exactly the wrong tokens to compress. The word "the" appears before nouns, verbs, adjectives, proper names, and virtually every other syntactic category. Representations at positions where the next token is "the" should be diverse because the contexts are diverse. A representation that's about to predict "the" before "dog" should be very different from one that's about to predict "the" before "implications." MCR2 spends most of its computational effort trying to collapse exactly the representations that should stay spread out.

Rare tokens are the opposite case. Unusual words benefit most from separation because their representations are the ones most at risk of being crowded out by common-word geometry. But rare words appear once or twice per batch, if at all. Their classes are too small to meaningfully participate in the MCR2 computation. We require a minimum class size of 2 to even include them.

Our data shows this dynamic in action. The separation ratio — the ratio of between-class to within-class distances — starts at 1.23 in the first thousand steps when MCR2 is most active relative to the model's other gradients. By the end of training, it drops to 0.96, actually below the CE-only baseline of 1.01. MCR2 begins by improving separation, then actively degrades it as training progresses and the compression of frequent tokens dominates.

But here's the complication our ablations revealed: despite the frequency problem, the tension between expansion and compression still produces the best generalization. The separation ratio degrades, the theoretical motivation is undermined, and the model still generalizes better with both terms active than with either one alone. The conflicting pressure that MCR2 imposes — even when it's fighting language structure by compressing diverse contexts together — is more useful as regularization than its absence.


What it means for training cost

Full MCR2 adds 3.4 times the training time. Each step requires computing per-class covariance matrices, log-determinants, and their gradients at every layer, on top of the normal forward and backward pass. On our RTX 5070 Ti, CE-only runs at about 26 seconds per hundred steps. The combined approach takes about 88 seconds per hundred steps. The expansion term alone adds almost nothing — about 29 seconds per hundred steps — because the cost is almost entirely in the per-class compression computation.

At inference time, the cost is zero. MCR2 is purely a training objective. The model architecture is unchanged. The forward pass is unchanged. You're paying a one-time cost during training for whatever improvement the regularizer provides.

Whether that trade is worth it depends entirely on context. For a research experiment on WikiText-2 with a small model, the difference is 42 minutes versus 2.4 hours. Irrelevant. For a frontier model training run costing millions of dollars, a 3.4x multiplier is a non-starter unless the improvement is substantial and consistent at scale. We don't have evidence for that yet.


Where this goes next

Three directions seem most promising given what we've learned.

First, frequency-weighted compression. The frequency analysis shows compression is being applied most heavily to the wrong tokens. Inverse-frequency weighting on the compression term would redirect the regularization pressure toward rare tokens and reduce pressure on common tokens whose representations need to stay diverse. This is the most direct fix for the problem we identified.

Second, test at scale on WikiText-103, which has fifty times more tokens. Our current result is on a data-starved model where any regularizer might help. The real question is whether MCR2 provides value when the parameter-to-token ratio is healthier. The gap might widen because there's more data for the geometric structure to work with, or it might vanish because cross-entropy alone has enough signal to prevent collapse.

Third, compare to simple baselines. Our ablations isolated which part of MCR2 works, but we haven't compared it to intentionally boring regularizers like L2 penalties on hidden-state variance, within-class MSE clustering, or label smoothing. If a simple constraint matches MCR2's performance, that would confirm that the log-det geometry is not uniquely important — any source of conflicting pressure would do. If MCR2 meaningfully outperforms simple baselines, that tells us the specific form of geometric tension matters, not just the existence of tension.


The bigger picture

We didn't show "MCR2 works" in the sense that the theory predicts. We showed that representation geometry regularization helps, and MCR2 is one way to get it. Neither the expansion term nor the compression term alone produces the effect. Expansion alone opens up representational capacity — the model memorizes harder. Compression alone collapses representations into over-specialized clusters — the model peaks early then overfits. Only the combination creates an unsatisfiable conflict that forces the model toward more general representations.

This is a common pattern. Theoretically motivated objectives often work for reasons that have little to do with the theory. The theory gets you to a functional form. The functional form has properties — in this case, creating irreconcilable geometric demands on representations — that the theory didn't predict and can't explain. The information-theoretic story about coding rates and precision is elegant, but what's actually happening is simpler: the model is being asked to do two conflicting things, and the conflict prevents it from taking shortcuts.

The frequency problem is real, the epsilon critique stands, and both terms are fighting each other in ways the theory doesn't account for. A better version of this loss — frequency-weighted, tuned for the specific structure of language rather than the idealized geometry of discriminative classification — would probably work better. But the version we tested, warts and all, improves generalization by 2.8 percent while revealing something genuine about how auxiliary objectives interact with language model training. Auxiliary objectives help when they introduce unavoidable tension, not when they produce prettier geometry. That's worth reporting.


Experimental details

All experiments used the same architecture: a decoder-only transformer with d_model of 192, 4 layers, 4 attention heads, feed-forward dimension of 768, context length of 256, and GPT-2's BPE tokenizer with a vocabulary of 50,257. This gives approximately 1.77 million non-embedding parameters. We trained on WikiText-2 for 10,000 steps with AdamW at a learning rate of 3e-4, cosine decay, 1,000 warmup steps, batch size of 64 with 4 gradient accumulation steps for an effective batch of 256 sequences, and bfloat16 mixed precision. The MCR2 weight (beta) was fixed at 0.1, and epsilon was set to 0.5. Hardware was a single NVIDIA RTX 5070 Ti with 16GB VRAM running CUDA 13.0 on Pop!_OS 22.04.

Seven runs were conducted: CE-only baseline, full MCR2 (expansion plus compression), expansion-only, selective expansion (centroid-based, frequency-neutral), curriculum (expansion for 2,000 steps then full MCR2), alternating (cycling between expansion and full MCR2 every 2,000 steps), and compression-only. All used identical model architecture, optimizer settings, and training duration. Only the loss function differed. Code, configs, and training logs are available in our repository.
