---
id: "03-evaluation-methodology"
title: "3. Evaluation Methodology"
language: "en"
---

<!-- cleaned -->
# 3. Evaluation Methodology

The more AI is used, the more opportunity there is for catastrophic failure. We've already seen many failures in the short time that foundation models have been around. A man committed suicide after being encouraged by a chatbot. Lawyers submitted false evidence hallucinated by AI. Air Canada was ordered to pay damages when its AI chatbot gave a passenger false information. Without a way to quality control AI outputs, the risk of AI might outweigh its benefits for many applications.

As teams rush to adopt AI, many quickly realize that the biggest hurdle to bringing AI applications to reality is evaluation. For some applications, figuring out evaluation can take up the majority of the development effort. Due to the importance and complexity of evaluation, this book has two chapters on it. This chapter covers different evaluation methods used to evaluate open-ended models, how these methods work, and their limitations. The next chapter focuses on how to use these methods to select models for your application and build an evaluation pipeline to evaluate your application.

While I discuss evaluation in its own chapters, evaluation has to be considered in the context of a whole system, not in isolation. Evaluation aims to mitigate risks and uncover opportunities. To mitigate risks, you first need to identify the places where your system is likely to fail and design your evaluation around them. Often, this may require redesigning your system to enhance visibility into its failures. Without a clear understanding of where your system fails, no amount of evaluation metrics or tools can make the system robust.

Before diving into evaluation methods, it's important to acknowledge the challenges of evaluating foundation models. Because evaluation is difficult, many people settle for word of mouth (e.g., someone says that the model X is good) or eyeballing the results. This creates even more risk and slows application iteration. Instead, we need to invest in systematic evaluation to make the results more reliable.

Since many foundation models have a language model component, this chapter will provide a quick overview of the metrics used to evaluate language models, including cross entropy and perplexity. These metrics are essential for guiding the training and finetuning of language models and are frequently used in many evaluation methods.

Evaluating foundation models is especially challenging because they are open-ended, and I'll cover best practices for how to tackle these. Using human evaluators remains a necessary option for many applications. However, given how slow and expensive human annotations can be, the goal is to automate the process. This book focuses on automatic evaluation, which includes both exact and subjective evaluation.

The rising star of subjective evaluation is AI as a judge—the approach of using AI to evaluate AI responses. It's subjective because the score depends on what model and prompt the AI judge uses. While this approach is gaining rapid traction in the industry, it also invites intense opposition from those who believe that AI isn't trustworthy enough for this important task. I'm especially excited to go deeper into this discussion, and I hope you will be, too.

## Challenges of Evaluating Foundation Models

Evaluating ML models has always been difficult. With the introduction of foundation models, evaluation has become even more so. There are multiple reasons why evaluating foundation models is more challenging than evaluating traditional ML models.

First, the more intelligent AI models become, the harder it is to evaluate them. Most people can tell if a first grader's math solution is wrong. Few can do the same for a PhD-level math solution. It's easy to tell if a book summary is bad if it's gibberish, but a lot harder if the summary is coherent. To validate the quality of a summary, you might need to read the book first. This brings us to a corollary: evaluation can be so much more time-consuming for sophisticated tasks. You can no longer evaluate a response based on how it sounds. You'll also need to fact-check, reason, and even incorporate domain expertise.

Second, the open-ended nature of foundation models undermines the traditional approach of evaluating a model against ground truths. With traditional ML, most tasks are close-ended. For example, a classification model can only output among the expected categories. To evaluate a classification model, you can evaluate its outputs against the expected outputs. If the expected output is category X but the model's output is category Y, the model is wrong. However, for an open-ended task, for a given input, there are so many possible correct responses. It's impossible to curate a comprehensive list of correct outputs to compare against.

Third, most foundation models are treated as black boxes, either because model providers choose not to expose models' details, or because application developers lack the expertise to understand them. Details such as the model architecture, training data, and the training process can reveal a lot about a model's strengths and weaknesses. Without those details, you can evaluate only a model by observing its outputs.

At the same time, publicly available evaluation benchmarks have proven to be inadequate for evaluating foundation models. Ideally, evaluation benchmarks should capture the full range of model capabilities. As AI progresses, benchmarks need to evolve to catch up. A benchmark becomes saturated for a model once the model achieves the perfect score. With foundation models, benchmarks are becoming saturated fast. The benchmark GLUE (General Language Understanding Evaluation) came out in 2018 and became saturated in just a year, necessitating the introduction of SuperGLUE in 2019. Similarly, NaturalInstructions (2021) was replaced by Super-NaturalInstructions (2022). MMLU (2020), a strong benchmark that many early foundation models relied on, was largely replaced by MMLU-Pro (2024).

Last but not least, the scope of evaluation has expanded for general-purpose models. With task-specific models, evaluation involves measuring a model's performance on its trained task. However, with general-purpose models, evaluation is not only about assessing a model's performance on known tasks but also about discovering new tasks that the model can do, and these might include tasks that extend beyond human capabilities. Evaluation takes on the added responsibility of exploring the potential and limitations of AI.

The good news is that the new challenges of evaluation have prompted many new methods and benchmarks. Figure 3-1 shows that the number of published papers on LLM evaluation grew exponentially every month in the first half of 2023, from 2 papers a month to almost 35 papers a month.

![Figure 3-1. The trend of LLMs evaluation papers over time. Image from Chang et al. (2023).](figure-3-1.png)

In my own analysis of the top 1,000 AI-related repositories on GitHub, as ranked by the number of stars, I found over 50 repositories dedicated to evaluation (as of May 2024). When plotting the number of evaluation repositories by their creation date, the growth curve looks exponential, as shown in Figure 3-2.

The bad news is that despite the increased interest in evaluation, it lags behind in terms of interest in the rest of the AI engineering pipeline. Balduzzi et al. from DeepMind noted in their paper that "developing evaluations has received little systematic attention compared to developing algorithms." According to the paper, experiment results are almost exclusively used to improve algorithms and are rarely used to improve evaluation. Recognizing the lack of investments in evaluation, Anthropic called on policymakers to increase government funding and grants both for developing new evaluation methodologies and analyzing the robustness of existing evaluations.

![Figure 3-2. Number of open source evaluation repositories among the 1,000 most popular AI repositories on GitHub.](figure-3-2.png)

To further demonstrate how the investment in evaluation lags behind other areas in the AI space, the number of tools for evaluation is small compared to the number of tools for modeling and training and AI orchestration, as shown in Figure 3-3.

Inadequate investment leads to inadequate infrastructure, making it hard for people to carry out systematic evaluations. When asked how they are evaluating their AI applications, many people told me that they just eyeballed the results. Many have a small set of go-to prompts that they use to evaluate models. The process of curating these prompts is ad hoc, usually based on the curator's personal experience instead of based on the application's needs. You might be able to get away with this ad hoc approach when getting a project off the ground, but it won't be sufficient for application iteration. This book focuses on a systematic approach to evaluation.

![Figure 3-3. According to data sourced from my list of the 1,000 most popular AI repositories on GitHub, evaluation lags behind other aspects of AI engineering in terms of open source tools.](figure-3-3.png)

## Understanding Language Modeling Metrics

Foundation models evolved out of language models. Many foundation models still have language models as their main components. For these models, the performance of the language model component tends to be well correlated to the foundation model's performance on downstream applications (Liu et al., 2023). Therefore, a rough understanding of language modeling metrics can be quite helpful in understanding downstream performance.

As discussed in Chapter 1, language modeling has been around for decades, popularized by Claude Shannon in his 1951 paper "Prediction and Entropy of Printed English". The metrics used to guide the development of language models haven't changed much since then. Most autoregressive language models are trained using cross entropy or its relative, perplexity. When reading papers and model reports, you might also come across bits-per-character (BPC) and bits-per-byte (BPB); both are variations of cross entropy.

All four metrics—cross entropy, perplexity, BPC, and BPB—are closely related. If you know the value of one, you can compute the other three, given the necessary information. While I refer to them as language modeling metrics, they can be used for any model that generates sequences of tokens, including non-text tokens.

Recall that a language model encodes statistical information (how likely a token is to appear in a given context) about languages. Statistically, given the context "I like drinking __", the next word is more likely to be "tea" than "charcoal". The more statistical information that a model can capture, the better it is at predicting the next token.

In ML lingo, a language model learns the distribution of its training data. The better this model learns, the better it is at predicting what comes next in the training data, and the lower its training cross entropy. As with any ML model, you care about its performance not just on the training data but also on your production data. In general, the closer your data is to a model's training data, the better the model can perform on your data.

Compared to the rest of the book, this section is math-heavy. If you find it confusing, feel free to skip the math part and focus on the discussion of how to interpret these metrics. Even if you're not training or finetuning language models, understanding these metrics can help with evaluating which models to use for your application. These metrics can occasionally be used for certain evaluation and data deduplication techniques, as discussed throughout this book.

### Entropy

Entropy measures how much information, on average, a token carries. The higher the entropy, the more information each token carries, and the more bits are needed to represent a token.

Let's use a simple example to illustrate this. Imagine you want to create a language to describe positions within a square, as shown in Figure 3-4. If your language has only two tokens, shown as (a) in Figure 3-4, each token can tell you whether the position is upper or lower. Since there are only two tokens, one bit is sufficient to represent them. The entropy of this language is, therefore, 1.

![Figure 3-4. Two languages describe positions within a square. Compared to the language on the left (a), the tokens on the right (b) carry more information, but they need more bits to represent them.](figure-3-4.png)

If your language has four tokens, shown as (b) in Figure 3-4, each token can give you a more specific position: upper-left, upper-right, lower-left, or lower-right. However, since there are now four tokens, you need two bits to represent them. The entropy of this language is 2. This language has higher entropy, since each token carries more information, but each token requires more bits to represent.

Intuitively, entropy measures how difficult it is to predict what comes next in a language. The lower a language's entropy (the less information a token of a language carries), the more predictable that language. In our previous example, the language with only two tokens is easier to predict than the language with four (you have to predict among only two possible tokens compared to four). This is similar to how, if you can perfectly predict what I will say next, what I say carries no new information.

### Cross Entropy

When you train a language model on a dataset, your goal is to get the model to learn the distribution of this training data. In other words, your goal is to get the model to predict what comes next in the training data. A language model's cross entropy on a dataset measures how difficult it is for the language model to predict what comes next in this dataset.

A model's cross entropy on the training data depends on two qualities:
1. The training data's predictability, measured by the training data's entropy
2. How the distribution captured by the language model diverges from the true distribution of the training data

Entropy and cross entropy share the same mathematical notation, H. Let P be the true distribution of the training data, and Q be the distribution learned by the language model. Accordingly, the following is true:

- The training data's entropy is, therefore, H(P).
- The divergence of Q with respect to P can be measured using the Kullback–Leibler (KL) divergence, which is mathematically represented as $D_{KL}(P||Q)$.
- The model's cross entropy with respect to the training data is therefore: $H(P, Q) = H(P) + D_{KL}(P||Q)$.

Cross entropy isn't symmetric. The cross entropy of Q with respect to P—H(P, Q)—is different from the cross entropy of P with respect to Q—H(Q, P).

A language model is trained to minimize its cross entropy with respect to the training data. If the language model learns perfectly from its training data, the model's cross entropy will be exactly the same as the entropy of the training data. The KL divergence of Q with respect to P will then be 0. You can think of a model's cross entropy as its approximation of the entropy of its training data.

### Bits-per-Character and Bits-per-Byte

One unit of entropy and cross entropy is bits. If the cross entropy of a language model is 6 bits, this language model needs 6 bits to represent each token.

Since different models have different tokenization methods—for example, one model uses words as tokens and another uses characters as tokens—the number of bits per token isn't comparable across models. Some use the number of bits-per-character (BPC) instead. If the number of bits per token is 6 and on average, each token consists of 2 characters, the BPC is 6/2 = 3.

One complication with BPC arises from different character encoding schemes. For example, with ASCII, each character is encoded using 7 bits, but with UTF-8, a character can be encoded using anywhere between 8 and 32 bits. A more standardized metric would be bits-per-byte (BPB), the number of bits a language model needs to represent one byte of the original training data. If the BPC is 3 and each character is 7 bits, or ⅞ of a byte, then the BPB is 3 / (⅞) = 3.43.

Cross entropy tells us how efficient a language model will be at compressing text. If the BPB of a language model is 3.43, meaning it can represent each original byte (8 bits) using 3.43 bits, this language model can compress the original training text to less than half the text's original size.

### Perplexity

Perplexity is the exponential of entropy and cross entropy. Perplexity is often shortened to PPL. Given a dataset with the true distribution P, its perplexity is defined as:

$$PPL(P) = 2^{H(P)}$$

The perplexity of a language model (with the learned distribution Q) on this dataset is defined as:

$$PPL(P, Q) = 2^{H(P,Q)}$$

If cross entropy measures how difficult it is for a model to predict the next token, perplexity measures the amount of uncertainty it has when predicting the next token. Higher uncertainty means there are more possible options for the next token.

Consider a language model trained to encode the 4 position tokens, as in Figure 3-4 (b), perfectly. The cross entropy of this language model is 2 bits. If this language model tries to predict a position in the square, it has to choose among $2^2 = 4$ possible options. Thus, this language model has a perplexity of 4.

So far, I've been using bit as the unit for entropy and cross entropy. Each bit can represent 2 unique values, hence the base of 2 in the preceding perplexity equation.

Popular ML frameworks, including TensorFlow and PyTorch, use nat (natural log) as the unit for entropy and cross entropy. Nat uses the base of e, the base of natural logarithm. If you use nat as the unit, perplexity is the exponential of e:

$$PPL(P, Q) = e^{H(P,Q)}$$

Due to the confusion around bit and nat, many people report perplexity, instead of cross entropy, when reporting their language models' performance.

### Perplexity Interpretation and Use Cases

As discussed, cross entropy, perplexity, BPC, and BPB are variations of language models' predictive accuracy measurements. The more accurately a model can predict a text, the lower these metrics are. In this book, I'll use perplexity as the default language modeling metric. Remember that the more uncertainty the model has in predicting what comes next in a given dataset, the higher the perplexity.

What's considered a good value for perplexity depends on the data itself and how exactly perplexity is computed, such as how many previous tokens a model has access to. Here are some general rules:

- **More structured data gives lower expected perplexity**: More structured data is more predictable. For example, HTML code is more predictable than everyday text. If you see an opening HTML tag like `<head>`, you can predict that there should be a closing tag, `</head>`, nearby. Therefore, the expected perplexity of a model on HTML code should be lower than the expected perplexity of a model on everyday text.
- **The bigger the vocabulary, the higher the perplexity**: Intuitively, the more possible tokens there are, the harder it is for the model to predict the next token. For example, a model's perplexity on a children's book will likely be lower than the same model's perplexity on War and Peace. For the same dataset, say in English, character-based perplexity (predicting the next character) will be lower than word-based perplexity (predicting the next word), because the number of possible characters is smaller than the number of possible words.
- **The longer the context length, the lower the perplexity**: The more context a model has, the less uncertainty it will have in predicting the next token. In 1951, Claude Shannon evaluated his model's cross entropy by using it to predict the next token conditioned on up to 10 previous tokens. As of this writing, a model's perplexity can typically be computed and conditioned on between 500 and 10,000 previous tokens, and possibly more, upperbounded by the model's maximum context length.

For reference, it's not uncommon to see perplexity values as low as 3 or even lower. If all tokens in a hypothetical language have an equal chance of happening, a perplexity of 3 means that this model has a 1 in 3 chance of predicting the next token correctly. Given that a model's vocabulary is in the order of 10,000s and 100,000s, these odds are incredible.

Other than guiding the training of language models, perplexity is useful in many parts of an AI engineering workflow. First, perplexity is a good proxy for a model's capabilities. If a model's bad at predicting the next token, its performance on downstream tasks will also likely be bad. OpenAI's GPT-2 report shows that larger models, which are also more powerful models, consistently give lower perplexity on a range of datasets, as shown in Table 3-1. Sadly, following the trend of companies being increasingly more secretive about their models, many have stopped reporting their models' perplexity.

**Table 3-1. Larger GPT-2 models consistently give lower perplexity on different datasets. Source: OpenAI**

| Model | LAMBADA (PPL) | LAMBADA (ACC) | CBT-CN (ACC) | CBT-NE (ACC) |
|-------|---------------|---------------|--------------|--------------|
| SOTA  | 99.8          | 59.23         | 85.7         | 82.3         |
| 117M  | 35.13         | 45.99         | 87.65        | 83.4         |
| 345M  | 15.60         | 55.48         | 92.35        | 87.1         |
| 762M  | 10.87         | 60.12         | 93.45        | 88.0         |
| 1542M | 8.63          | 63.24         | 93.30        | 89.05        |

> **WARNING**
> Perplexity might not be a great proxy to evaluate models that have been post-trained using techniques like SFT and RLHF. Post-training is about teaching models how to complete tasks. As a model gets better at completing tasks, it might get worse at predicting the next tokens. A language model's perplexity typically increases after post-training. Some people say that post-training collapses entropy. Similarly, quantization—a technique that reduces a model's numerical precision and, with it, its memory footprint—can also change a model's perplexity in unexpected ways.

Recall that the perplexity of a model with respect to a text measures how difficult it is for this model to predict this text. For a given model, perplexity is the lowest for texts that the model has seen and memorized during training. Therefore, perplexity can be used to detect whether a text was in a model's training data. This is useful for detecting data contamination—if a model's perplexity on a benchmark's data is low, this benchmark was likely included in the model's training data, making the model's performance on this benchmark less trustworthy. This can also be used for deduplication of training data: e.g., add new data to the existing training dataset only if the perplexity of the new data is high.

Perplexity is the highest for unpredictable texts, such as texts expressing unusual ideas (like "my dog teaches quantum physics in his free time") or gibberish (like "home cat go eye"). Therefore, perplexity can be used to detect abnormal texts.

Perplexity and its related metrics help us understand the performance of the underlying language model, which is a proxy for understanding the model's performance on downstream tasks. The rest of the chapter discusses how to measure a model's performance on downstream tasks directly.

**HOW TO USE A LANGUAGE MODEL TO COMPUTE A TEXT'S PERPLEXITY**

A model's perplexity with respect to a text measures how difficult it is for the model to predict that text. Given a language model X, and a sequence of tokens $[x_1, x_2, \ldots, x_n]$, X's perplexity for this sequence is:

$$PPL(X, [x_1, x_2, \ldots, x_n]) = \left( P(x_1, x_2, \ldots, x_n) \right)^{-\frac{1}{n}} = \left( \prod_{i=1}^{n} P(x_i | x_1, \ldots, x_{i-1}) \right)^{-\frac{1}{n}}$$

where $P(x_i | x_1, \ldots, x_{i-1})$ denotes the probability that X assigns to the token $x_i$ given the previous tokens $x_1, \ldots, x_{i-1}$.

To compute perplexity, you need access to the probabilities (or logprobs) the language model assigns to each next token. Unfortunately, not all commercial models expose their models' logprobs, as discussed in Chapter 2.

## Exact Evaluation

When evaluating models' performance, it's important to differentiate between exact and subjective evaluation. Exact evaluation produces judgment without ambiguity. For example, if the answer to a multiple-choice question is A and you pick B, your answer is wrong. There's no ambiguity around that. On the other hand, essay grading is subjective. An essay's score depends on who grades the essay. The same person, if asked twice some time apart, can give the same essay different scores. Essay grading can become more exact with clear grading guidelines. As you'll see in the next section, AI as a judge is subjective. The evaluation result can change based on the judge model and the prompt.

I'll cover two evaluation approaches that produce exact scores: functional correctness and similarity measurements against reference data. Note that this section focuses on evaluating open-ended responses (arbitrary text generation) as opposed to close-ended responses (such as classification). This is not because foundation models aren't being used for close-ended tasks. In fact, many foundation model systems have at least a classification component, typically for intent classification or scoring. This section focuses on open-ended evaluation because close-ended evaluation is already well understood.

### Functional Correctness

Functional correctness evaluation means evaluating a system based on whether it performs the intended functionality. For example, if you ask a model to create a website, does the generated website meet your requirements? If you ask a model to make a reservation at a certain restaurant, does the model succeed?

Functional correctness is the ultimate metric for evaluating the performance of any application, as it measures whether your application does what it's intended to do. However, functional correctness isn't always straightforward to measure, and its measurement can't be easily automated.

Code generation is an example of a task where functional correctness measurement can be automated. Functional correctness in coding is sometimes execution accuracy. Say you ask the model to write a Python function, `gcd(num1, num2)`, to find the greatest common denominator (gcd) of two numbers, `num1` and `num2`. The generated code can then be input into a Python interpreter to check whether the code is valid and if it is, whether it outputs the correct result of a given pair `(num1, num2)`. For example, given the pair `(num1=15, num2=20)`, if the function `gcd(15, 20)` doesn't return 5, the correct answer, you know that the function is wrong.

Long before AI was used for writing code, automatically verifying code's functional correctness was standard practice in software engineering. Code is typically validated with unit tests where code is executed in different scenarios to ensure that it generates the expected outputs. Functional correctness evaluation is how coding platforms like LeetCode and HackerRank validate the submitted solutions.

Popular benchmarks for evaluating AI's code generation capabilities, such as OpenAI's HumanEval and Google's MBPP (Mostly Basic Python Problems Dataset) use functional correctness as their metrics. Benchmarks for text-to-SQL (generating SQL queries from natural languages) like Spider (Yu et al., 2018), BIRD-SQL (Big Bench for Large-scale Database Grounded Text-to-SQL Evaluation) (Li et al., 2023), and WikiSQL (Zhong, et al., 2017) also rely on functional correctness.

A benchmark problem comes with a set of test cases. Each test case consists of a scenario the code should run and the expected output for that scenario. Here's an example of a problem and its test cases in HumanEval:

```python
# Problem
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
```

```python
# Test cases (each assert statement represents a test case)
def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False
```

When evaluating a model, for each problem a number of code samples, denoted as k, are generated. A model solves a problem if any of the k code samples it generated pass all of that problem's test cases. The final score, called pass@k, is the fraction of the solved problems out of all problems. If there are 10 problems and a model solves 5 with k = 3, then that model's pass@3 score is 50%. The more code samples a model generates, the more chance the model has at solving each problem, hence the greater the final score. This means that in expectation, pass@1 score should be lower than pass@3, which, in turn, should be lower than pass@10.

Another category of tasks whose functional correctness can be automatically evaluated is game bots. If you create a bot to play Tetris, you can tell how good the bot is by the score it gets. Tasks with measurable objectives can typically be evaluated using functional correctness. For example, if you ask AI to schedule your workloads to optimize energy consumption, the AI's performance can be measured by how much energy it saves.

### Similarity Measurements Against Reference Data

If the task you care about can't be automatically evaluated using functional correctness, one common approach is to evaluate AI's outputs against reference data. For example, if you ask a model to translate a sentence from French to English, you can evaluate the generated English translation against the correct English translation.

Each example in the reference data follows the format (input, reference responses). An input can have multiple reference responses, such as multiple possible English translations of a French sentence. Reference responses are also called ground truths or canonical responses. Metrics that require references are reference-based, and metrics that don't are reference-free.

Since this evaluation approach requires reference data, it's bottlenecked by how much and how fast reference data can be generated. Reference data is generated typically by humans and increasingly by AIs. Using human-generated data as the reference means that we treat human performance as the gold standard, and AI's performance is measured against human performance. Human-generated data can be expensive and time-consuming to generate, leading many to use AI to generate reference data instead. AI-generated data might still need human reviews, but the labor needed to review it is much less than the labor needed to generate reference data from scratch.

Generated responses that are more similar to the reference responses are considered better. There are four ways to measure the similarity between two open-ended texts:

1. Asking an evaluator to make the judgment whether two texts are the same
2. Exact match: whether the generated response matches one of the reference responses exactly
3. Lexical similarity: how similar the generated response looks to the reference responses
4. Semantic similarity: how close the generated response is to the reference responses in meaning (semantics)

Two responses can be compared by human evaluators or AI evaluators. AI evaluators are increasingly common and will be the focus of the next section.

This section focuses on hand-designed metrics: exact match, lexical similarity, and semantic similarity. Scores by exact matching are binary (match or not), whereas the other two scores are on a sliding scale (such as between 0 and 1 or between –1 and 1). Despite the ease of use and flexibility of the AI as a judge approach, hand-designed similarity measurements are still widely used in the industry for their exact nature.

> **NOTE**
> This section discusses how you can use similarity measurements to evaluate the quality of a generated output. However, you can also use similarity measurements for many other use cases, including but not limited to the following:
> - **Retrieval and search**: find items similar to a query
> - **Ranking**: rank items based on how similar they are to a query
> - **Clustering**: cluster items based on how similar they are to each other
> - **Anomaly detection**: detect items that are the least similar to the rest
> - **Data deduplication**: remove items that are too similar to other items
>
> Techniques discussed in this section will come up again throughout the book.

#### Exact Match

It's considered an exact match if the generated response matches one of the reference responses exactly. Exact matching works for tasks that expect short, exact responses such as simple math problems, common knowledge queries, and trivia-style questions. Here are examples of inputs that have short, exact responses:

- "What's 2 + 3?"
- "Who was the first woman to win a Nobel Prize?"
- "What's my current account balance?"
- "Fill in the blank: Paris to France is like ___ to England."

There are variations to matching that take into account formatting issues. One variation is to accept any output that contains the reference response as a match. Consider the question "What's 2 + 3?" The reference response is "5". This variation accepts all outputs that contain "5", including "The answer is 5" and "2 + 3 is 5".

However, this variation can sometimes lead to the wrong solution being accepted. Consider the question "What year was Anne Frank born?" Anne Frank was born on June 12, 1929, so the correct response is 1929. If the model outputs "September 12, 1929", the correct year is included in the output, but the output is factually wrong.

Beyond simple tasks, exact match rarely works. Given the original French sentence "Comment ça va?", there are multiple possible English translations, such as "How are you?", "How is everything?", and "How are you doing?" If the reference data contains only these three translations and a model generates "How is it going?", the model's response will be marked as wrong. The longer and more complex the original text, the more possible translations there are. It's impossible to create an exhaustive set of possible responses for an input. For complex tasks, lexical similarity and semantic similarity work better.

#### Lexical Similarity

Lexical similarity measures how much two texts overlap. You can do this by first breaking each text into smaller tokens.

In its simplest form, lexical similarity can be measured by counting how many tokens two texts have in common. As an example, consider the reference response "My cats scare the mice" and two generated responses:

- A: "My cats eat the mice"
- B: "Cats and mice fight all the time"

Assume that each token is a word. If you count overlapping of individual words only, response A contains 4 out of 5 words in the reference response (the similarity score is 80%), whereas response B contains only 3 out of 5 (the similarity score is 60%). Response A is, therefore, considered more similar to the reference response.

One way to measure lexical similarity is approximate string matching, known colloquially as fuzzy matching. It measures the similarity between two texts by counting how many edits it'd need to convert from one text to another, a number called edit distance. The usual three edit operations are:

1. Deletion: "brad" -> "bad"
2. Insertion: "bad" -> "bard"
3. Substitution: "bad" -> "bed"

Some fuzzy matchers also treat transposition, swapping two letters (e.g., "mats" -> "mast"), to be an edit. However, some fuzzy matchers treat each transposition as two edit operations: one deletion and one insertion.

For example, "bad" is one edit to "bard" and three edits to "cash", so "bad" is considered more similar to "bard" than to "cash".

Another way to measure lexical similarity is n-gram similarity, measured based on the overlapping of sequences of tokens, n-grams, instead of single tokens. A 1-gram (unigram) is a token. A 2-gram (bigram) is a set of two tokens. "My cats scare the mice" consists of four bigrams: "my cats", "cats scare", "scare the", and "the mice". You measure what percentage of n-grams in reference responses is also in the generated response.

Common metrics for lexical similarity are BLEU, ROUGE, METEOR++, TER, and CIDEr. They differ in exactly how the overlapping is calculated. Before foundation models, BLEU, ROUGE, and their relatives were common, especially for translation tasks. Since the rise of foundation models, fewer benchmarks use lexical similarity. Examples of benchmarks that use these metrics are WMT, COCO Captions, and GEMv2.

A drawback of this method is that it requires curating a comprehensive set of reference responses. A good response can get a low similarity score if the reference set doesn't contain any response that looks like it. On some benchmark examples, Adept found that its model Fuyu performed poorly not because the model's outputs were wrong, but because some correct answers were missing in the reference data. Figure 3-5 shows an example of an image-captioning task in which Fuyu generated a correct caption but was given a low score.

Not only that, but references can be wrong. For example, the organizers of the WMT 2023 Metrics shared task, which focuses on examining evaluation metrics for machine translation, reported that they found many bad reference translations in their data. Low-quality reference data is one of the reasons that reference-free metrics were strong contenders for reference-based metrics in terms of correlation to human judgment (Freitag et al., 2023).

Another drawback of this measurement is that higher lexical similarity scores don't always mean better responses. For example, on HumanEval, a code generation benchmark, OpenAI found that BLEU scores for incorrect and correct solutions were similar. This indicates that optimizing for BLEU scores isn't the same as optimizing for functional correctness (Chen et al., 2021).

![Figure 3-5. An example where Fuyu generated a correct option but was given a low score because of the limitation of reference captions.](figure-3-5.png)

#### Semantic Similarity

Lexical similarity measures whether two texts look similar, not whether they have the same meaning. Consider the two sentences "What's up?" and "How are you?" Lexically, they are different—there's little overlapping in the words and letters they use. However, semantically, they are close. Conversely, similar-looking texts can mean very different things. "Let's eat, grandma" and "Let's eat grandma" mean two completely different things.

Semantic similarity aims to compute the similarity in semantics. This first requires transforming a text into a numerical representation, which is called an embedding. For example, the sentence "the cat sits on a mat" might be represented using an embedding that looks like this: `[0.11, 0.02, 0.54]`. Semantic similarity is, therefore, also called embedding similarity. "Introduction to Embedding" discusses how embeddings work. For now, let's assume that you have a way to transform texts into embeddings. The similarity between two embeddings can be computed using metrics such as cosine similarity. Two embeddings that are exactly the same have a similarity score of 1. Two opposite embeddings have a similarity score of –1.

I'm using text examples, but semantic similarity can be computed for embeddings of any data modality, including images and audio. Semantic similarity for text is sometimes called semantic textual similarity.

> **WARNING**
> While I put semantic similarity in the exact evaluation category, it can be considered subjective, as different embedding algorithms can produce different embeddings. However, given two embeddings, the similarity score between them is computed exactly.

Mathematically, let A be an embedding of the generated response, and B be an embedding of a reference response. The cosine similarity between A and B is computed as $\frac{A \cdot B}{||A|| \cdot ||B||}$, with:

- $A \cdot B$ being the dot product of A and B
- $||A||$ being the Euclidean norm (also known as L2 norm) of A. If A is `[0.11, 0.02, 0.54]`, $||A|| = \sqrt{0.11^2 + 0.02^2 + 0.54^2}$

Metrics for semantic textual similarity include BERTScore (embeddings are generated by BERT) and MoverScore (embeddings are generated by a mixture of algorithms).

Semantic textual similarity doesn't require a set of reference responses as comprehensive as lexical similarity does. However, the reliability of semantic similarity depends on the quality of the underlying embedding algorithm. Two texts with the same meaning can still have a low semantic similarity score if their embeddings are bad. Another drawback of this measurement is that the underlying embedding algorithm might require nontrivial compute and time to run.

Before we move on to discuss AI as a judge, let's go over a quick introduction to embedding. The concept of embedding lies at the heart of semantic similarity, and is the backbone of many topics we explore throughout the book, including vector search in Chapter 6 and data deduplication in Chapter 8.

### Introduction to Embedding

Since computers work with numbers, a model needs to convert its input into numerical representations that computers can process. An embedding is a numerical representation that aims to capture the meaning of the original data.

An embedding is a vector. For example, the sentence "the cat sits on a mat" might be represented using an embedding vector that looks like this: `[0.11, 0.02, 0.54]`. Here, I use a small vector as an example. In reality, the size of an embedding vector (the number of elements in the embedding vector) is typically between 100 and 10,000.

Models trained especially to produce embeddings include the open source models BERT, CLIP (Contrastive Language–Image Pre-training), and Sentence Transformers. There are also proprietary embedding models provided as APIs. Table 3-2 shows the embedding sizes of some popular models.

**Table 3-2. Embedding sizes used by common models.**

| Model | Embedding size |
|-------|----------------|
| Google's BERT | BERT base: 768, BERT large: 1024 |
| OpenAI's CLIP | Image: 512, Text: 512 |
| OpenAI Embeddings API | text-embedding-3-small: 1536, text-embedding-3-large: 3072 |
| Cohere's Embed v3 | embed-english-v3.0: 1024, embed-english-light-3.0: 384 |

Because models typically require their inputs to first be transformed into vector representations, many ML models, including GPTs and Llamas, also involve a step to generate embeddings. "Transformer architecture" visualizes the embedding layer in a transformer model. If you have access to the intermediate layers of these models, you can use them to extract embeddings. However, the quality of these embeddings might not be as good as the embeddings generated by specialized embedding models.

The goal of the embedding algorithm is to produce embeddings that capture the essence of the original data. How do we verify that? The embedding vector `[0.11, 0.02, 0.54]` looks nothing like the original text "the cat sits on a mat".

At a high level, an embedding algorithm is considered good if more-similar texts have closer embeddings, measured by cosine similarity or related metrics. The embedding of the sentence "the cat sits on a mat" should be closer to the embedding of "the dog plays on the grass" than the embedding of "AI research is super fun".

You can also evaluate the quality of embeddings based on their utility for your task. Embeddings are used in many tasks, including classification, topic modeling, recommender systems, and RAG. An example of benchmarks that measure embedding quality on multiple tasks is MTEB, Massive Text Embedding Benchmark (Muennighoff et al., 2023).

I use texts as examples, but any data can have embedding representations. For example, ecommerce solutions like Criteo and Coveo have embeddings for products. Pinterest has embeddings for images, graphs, queries, and even users.

A new frontier is to create joint embeddings for data of different modalities. CLIP (Radford et al., 2021) was one of the first major models that could map data of different modalities, text and images, into a joint embedding space. ULIP (unified representation of language, images, and point clouds) (Xue et al., 2022) aims to create unified representations of text, images, and 3D point clouds. ImageBind (Girdhar et al., 2023) learns a joint embedding across six different modalities, including text, images, and audio.

Figure 3-6 visualizes CLIP's architecture. CLIP is trained using (image, text) pairs. The text corresponding to an image can be the caption or a comment associated with this image. For each (image, text) pair, CLIP uses a text encoder to convert the text to a text embedding, and an image encoder to convert the image to an image embedding. It then projects both these embeddings into a joint embedding space. The training goal is to get the embedding of an image close to the embedding of the corresponding text in this joint space.

![Figure 3-6. CLIP's architecture (Radford et al., 2021).](figure-3-6.png)

A joint embedding space that can represent data of different modalities is a multimodal embedding space. In a text–image joint embedding space, the embedding of an image of a man fishing should be closer to the embedding of the text "a fisherman" than the embedding of the text "fashion show". This joint embedding space allows embeddings of different modalities to be compared and combined. For example, this enables text-based image search. Given a text, it helps you find images closest to this text.

## AI as a Judge

The challenges of evaluating open-ended responses have led many teams to fall back on human evaluation. As AI has successfully been used to automate many challenging tasks, can AI automate evaluation as well? The approach of using AI to evaluate AI is called AI as a judge or LLM as a judge. An AI model that is used to evaluate other AI models is called an AI judge.

While the idea of using AI to automate evaluation has been around for a long time, it only became practical when AI models became capable of doing so, which was around 2020 with the release of GPT-3. As of this writing, AI as a judge has become one of the most, if not the most, common methods for evaluating AI models in production. Most demos of AI evaluation startups I saw in 2023 and 2024 leveraged AI as a judge in one way or another. LangChain's State of AI report in 2023 noted that 58% of evaluations on their platform were done by AI judges. AI as a judge is also an active area of research.

### Why AI as a Judge?

AI judges are fast, easy to use, and relatively cheap compared to human evaluators. They can also work without reference data, which means they can be used in production environments where there is no reference data. You can ask AI models to judge an output based on any criteria: correctness, repetitiveness, toxicity, wholesomeness, hallucinations, and more. This is similar to how you can ask a person to give their opinion about anything. You might think, "But you can't always trust people's opinions." That's true, and you can't always trust AI's judgments, either. However, as each AI model is an aggregation of the masses, it's possible for AI models to make judgments representative of the masses. With the right prompt for the right model, you can get reasonably good judgments on a wide range of topics.

Studies have shown that certain AI judges are strongly correlated to human evaluators. In 2023, Zheng et al. found that on their evaluation benchmark, MT-Bench, the agreement between GPT-4 and humans reached 85%, which is even higher than the agreement among humans (81%). AlpacaEval authors (Dubois et al., 2023) also found that their AI judges have a near perfect (0.98) correlation with LMSYS's Chat Arena leaderboard, which is evaluated by humans.

Not only can AI evaluate a response, but it can also explain its decision, which can be especially useful when you want to audit your evaluation results. Figure 3-7 shows an example of GPT-4 explaining its judgment.

Its flexibility makes AI as a judge useful for a wide range of applications, and for some applications, it's the only automatic evaluation option. Even when AI judgments aren't as good as human judgments, they might still be good enough to guide an application's development and provide sufficient confidence to get a project off the ground.

![Figure 3-7. Not only can AI judges score, they also can explain their decisions.](figure-3-7.png)

### How to Use AI as a Judge

There are many ways you can use AI to make judgments. For example, you can use AI to evaluate the quality of a response by itself, compare that response to reference data, or compare that response to another response. Here are naive example prompts for these three approaches:

1. **Evaluate the quality of a response by itself, given the original question:**

```text
"Given the following question and answer, evaluate the quality of the answer for the question. Use the score from 1 to 5.
- 1 means very bad.
- 5 means very good.

Question: [QUESTION]
Answer: [ANSWER]
Score:"
```

2. **Compare a generated response to a reference response to evaluate whether the generated response is the same as the reference response.** This can be an alternative approach to human-designed similarity measurements:

```text
"Given the following question, reference answer, and generated answer, evaluate whether this generated answer is the same as the reference answer. Output True or False.

Question: [QUESTION]
Reference answer: [REFERENCE ANSWER]
Generated answer: [GENERATED ANSWER]"
```

3. **Compare two generated responses and determine which one is better or predict which one users will likely prefer.** This is helpful for generating preference data for post-training alignment (discussed in Chapter 2), test-time compute (discussed in Chapter 2), and ranking models using comparative evaluation (discussed in the next section):

```text
"Given the following question and two answers, determine which answer is better. Output A or B.

Question: [QUESTION]
A: [FIRST ANSWER]
B: [SECOND ANSWER]
The better answer is:"
```

A general-purpose AI judge can be asked to evaluate a response based on any criteria. If you're building a roleplaying chatbot, you might want to evaluate if a chatbot's response is consistent with the role users want it to play,

## Figures

![Figure](/assets/en/03-evaluation-methodology/p237_01.png)

![Figure](/assets/en/03-evaluation-methodology/p238_01.png)

![Figure](/assets/en/03-evaluation-methodology/p240_01.png)

![Figure](/assets/en/03-evaluation-methodology/p243_01.png)

![Figure](/assets/en/03-evaluation-methodology/p264_01.png)

![Figure](/assets/en/03-evaluation-methodology/p270_01.png)

![Figure](/assets/en/03-evaluation-methodology/p274_01.png)

![Figure](/assets/en/03-evaluation-methodology/p281_01.png)

![Figure](/assets/en/03-evaluation-methodology/p295_01.png)

![Figure](/assets/en/03-evaluation-methodology/p297_01.png)
