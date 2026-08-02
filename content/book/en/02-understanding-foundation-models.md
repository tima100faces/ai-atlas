---
id: "02-understanding-foundation-models"
title: "2. Understanding Foundation Models"
language: "en"
---

<!-- cleaned -->
# 2. Understanding Foundation Models

To build applications with foundation models, you first need foundation models. While you don't need to know how to develop a model to use it, a high-level understanding will help you decide what model to use and how to adapt it to your needs.

Training a foundation model is an incredibly complex and costly process. Those who know how to do this well are likely prevented by confidentiality agreements from disclosing the secret sauce. This chapter won't be able to tell you how to build a model to compete with ChatGPT. Instead, I'll focus on design decisions with consequential impact on downstream applications.

With the growing lack of transparency in the training process of foundation models, it's difficult to know all the design decisions that go into making a model. In general, however, differences in foundation models can be traced back to decisions about training data, model architecture and size, and how they are post-trained to align with human preferences.

Since models learn from data, their training data reveals a great deal about their capabilities and limitations. This chapter begins with how model developers curate training data, focusing on the distribution of training data. Chapter 8 explores dataset engineering techniques in detail, including data quality evaluation and data synthesis.

Given the dominance of the transformer architecture, it might seem that model architecture is less of a choice. You might be wondering, what makes the transformer architecture so special that it continues to dominate? How long until another architecture takes over, and what might this new architecture look like? This chapter will address all of these questions.

Whenever a new model is released, one of the first things people want to know is its size. This chapter will also explore how a model developer might determine the appropriate size for their model.

As mentioned in Chapter 1, a model's training process is often divided into pre-training and post-training. Pre-training makes a model capable, but not necessarily safe or easy to use. This is where post-training comes in. The goal of post-training is to align the model with human preferences. But what exactly is human preference? How can it be represented in a way that a model can learn? The way a model developer aligns their model has a significant impact on the model's usability, and will be discussed in this chapter.

While most people understand the impact of training on a model's performance, the impact of sampling is often overlooked. Sampling is how a model chooses an output from all possible options. It is perhaps one of the most underrated concepts in AI. Not only does sampling explain many seemingly baffling AI behaviors, including hallucinations and inconsistencies, but choosing the right sampling strategy can also significantly boost a model's performance with relatively little effort. For this reason, sampling is the section that I was the most excited to write about in this chapter.

Concepts covered in this chapter are fundamental for understanding the rest of the book. However, because these concepts are fundamental, you might already be familiar with them. Feel free to skip any concept that you're confident about. If you encounter a confusing concept later on, you can revisit this chapter.

## Training Data

An AI model is only as good as the data it was trained on. If there's no Vietnamese in the training data, the model won't be able to translate from English into Vietnamese. Similarly, if an image classification model sees only animals in its training set, it won't perform well on photos of plants.

If you want a model to improve on a certain task, you might want to include more data for that task in the training data. However, collecting sufficient data for training a large model isn't easy, and it can be expensive. Model developers often have to rely on available data, even if this data doesn't exactly meet their needs.

For example, a common source for training data is Common Crawl, created by a nonprofit organization that sporadically crawls websites on the internet. In 2022 and 2023, this organization crawled approximately 2–3 billion web pages each month. Google provides a clean subset of Common Crawl called the Colossal Clean Crawled Corpus, or C4 for short.

The data quality of Common Crawl, and C4 to a certain extent, is questionable—think clickbait, misinformation, propaganda, conspiracy theories, racism, misogyny, and every sketchy website you've ever seen or avoided on the internet. A study by the Washington Post shows that the 1,000 most common websites in the dataset include several media outlets that rank low on NewsGuard's scale for trustworthiness. In lay terms, Common Crawl contains plenty of fake news.

Yet, simply because Common Crawl is available, variations of it are used in most foundation models that disclose their training data sources, including OpenAI's GPT-3 and Google's Gemini. I suspect that Common Crawl is also used in models that don't disclose their training data. To avoid scrutiny from both the public and competitors, many companies have stopped disclosing this information.

Some teams use heuristics to filter out low-quality data from the internet. For example, OpenAI used only the Reddit links that received at least three upvotes to train GPT-2. While this does help screen out links that nobody cares about, Reddit isn't exactly the pinnacle of propriety and good taste.

The "use what we have, not what we want" approach may lead to models that perform well on tasks present in the training data but not necessarily on the tasks you care about. To address this issue, it's crucial to curate datasets that align with your specific needs. This section focuses on curating data for specific languages and domains, providing a broad yet specialized foundation for applications within those areas. Chapter 8 explores data strategies for models tailored to highly specific tasks.

While language- and domain-specific foundation models can be trained from scratch, it's also common to finetune them on top of general-purpose models.

Some might wonder, why not just train a model on all data available, both general data and specialized data, so that the model can do everything? This is what many people do. However, training on more data often requires more compute resources and doesn't always lead to better performance. For example, a model trained with a smaller amount of high-quality data might outperform a model trained with a large amount of low-quality data. Using 7B tokens of high-quality coding data, Gunasekar et al. (2023) were able to train a 1.3B-parameter model that outperforms much larger models on several important coding benchmarks. The impact of data quality is discussed more in Chapter 8.

### Multilingual Models

English dominates the internet. An analysis of the Common Crawl dataset shows that English accounts for almost half of the data (45.88%), making it eight times more prevalent than the second-most common language, Russian (5.97%) (Lai et al., 2023). See Table 2-1 for a list of languages with at least 1% in Common Crawl. Languages with limited availability as training data—typically languages not included in this list—are considered low-resource.

**Table 2-1. The most common languages in Common Crawl, a popular dataset for training LLMs. Source: Lai et al. (2023).**

| Language | Code | Pop. (M) | CC size (%) | Cat. |
|---|---|---|---|---|
| English | en | 1,452 | 45.8786 | H |
| Russian | ru | — | 5.9692 | H |
| German | de | — | 5.8811 | H |
| Chinese | zh | 1,118 | 4.8747 | H |
| Japanese | jp | — | 4.7884 | H |
| French | fr | — | 4.7254 | H |
| Spanish | es | — | 4.4690 | H |
| Italian | it | — | 2.5712 | H |
| Dutch | nl | — | 2.0585 | H |
| Polish | pl | — | 1.6636 | H |
| Portuguese | pt | — | 1.1505 | H |
| Vietnamese | vi | — | 1.0299 | H |

Many other languages, despite having a lot of speakers today, are severely under-represented in Common Crawl. Table 2-2 shows some of these languages. Ideally, the ratio between world population representation and Common Crawl representation should be 1. The higher this ratio, the more under-represented this language is in Common Crawl.

**Table 2-2. Examples of under-represented languages in Common Crawl. The last row, English, is for comparison. The numbers for % in Common Crawl are taken from Lai et al. (2023).**

| Language | Speakers (million) | % world population | % in Common Crawl | World/Common Crawl ratio |
|---|---|---|---|---|
| Punjabi | — | 1.41% | 0.0061% | 231.5 |
| Swahili | — | 0.89% | 0.0077% | 115.2 |
| Urdu | — | 2.89% | 0.0274% | 105.3 |
| Kannada | — | 0.80% | 0.0122% | 65.57 |
| Telugu | — | 1.19% | 0.0183% | 64.89 |
| Gujarati | — | 0.78% | 0.0126% | 61.51 |
| Marathi | — | 1.24% | 0.0213% | 58.10 |
| Bengali | — | 3.40% | 0.0930% | 36.56 |
| English | — | 18.15% | 45.88% | 0.40 |

*A world population of eight billion was used for this calculation.*

Given the dominance of English in the internet data, it's not surprising that general-purpose models work much better for English than other languages, according to multiple studies. For example, on the MMLU benchmark, a suite of 14,000 multiple-choice problems spanning 57 subjects, GPT-4 performed much better in English than under-represented languages like Telugu, as shown in Figure 2-1 (OpenAI, 2023).

[Image: Figure 2-1. On the MMLU benchmark, GPT-4 performs better in English than in any other language. To obtain MMLU in other languages, OpenAI translated the questions using Azure AI Translator.]

Similarly, when tested on six math problems on Project Euler, Yennie Jun found that GPT-4 was able to solve problems in English more than three times as often compared to Armenian or Farsi. GPT-4 failed in all six questions for Burmese and Amharic, as shown in Figure 2-2.

[Image: Figure 2-2. GPT-4 is much better at math in English than in other languages.]

Under-representation is a big reason for this underperformance. The three languages that have the worst performance on GPT-4's MMLU benchmarks—Telugu, Marathi, and Punjabi—are also among the languages that are most under-represented in Common Crawl. However, under-representation isn't the only reason. A language's structure and the culture it embodies can also make a language harder for a model to learn.

Given that LLMs are generally good at translation, can we just translate all queries from other languages into English, obtain the responses, and translate them back into the original language? Many people indeed follow this approach, but it's not ideal. First, this requires a model that can sufficiently understand under-represented languages to translate. Second, translation can cause information loss. For example, some languages, like Vietnamese, have pronouns to denote the relationship between the two speakers. When translating into English, all these pronouns are translated into *I* and *you*, causing the loss of the relationship information.

Models can also have unexpected performance challenges in non-English languages. For example, NewsGuard found that ChatGPT is more willing to produce misinformation in Chinese than in English. In April 2023, NewsGuard asked ChatGPT-3.5 to produce misinformation articles about China in English, simplified Chinese, and traditional Chinese. For English, ChatGPT declined to produce false claims for six out of seven prompts. However, it produced false claims in simplified Chinese and traditional Chinese all seven times. It's unclear what causes this difference in behavior.

Other than quality issues, models can also be slower and more expensive for non-English languages. A model's inference latency and cost is proportional to the number of tokens in the input and response. It turns out that tokenization can be much more efficient for some languages than others. Benchmarking GPT-4 on MASSIVE, a dataset of one million short texts translated across 52 languages, Yennie Jun found that, to convey the same meaning, languages like Burmese and Hindi require a lot more tokens than English or Spanish. For the MASSIVE dataset, the median token length in English is 7, but the median length in Hindi is 32, and in Burmese, it's a whopping 72, which is ten times longer than in English.

Assuming that the time it takes to generate a token is the same in all languages, GPT-4 takes approximately ten times longer in Burmese than in English for the same content. For APIs that charge by token usage, Burmese costs ten times more than English.

To address this, many models have been trained to focus on non-English languages. The most active language, other than English, is undoubtedly Chinese, with ChatGLM, YAYI, Llama-Chinese, and others. There are also models in French (CroissantLLM), Vietnamese (PhoGPT), Arabic (Jais), and many more languages.

### Domain-Specific Models

General-purpose models like Gemini, GPTs, and Llamas can perform incredibly well on a wide range of domains, including but not limited to coding, law, science, business, sports, and environmental science. This is largely thanks to the inclusion of these domains in their training data. Figure 2-3 shows the distribution of domains present in Common Crawl according to the Washington Post's 2023 analysis.

[Image: Figure 2-3. Distribution of domains in the C4 dataset. Reproduced from the statistics from the Washington Post. One caveat of this analysis is that it only shows the categories that are included, not the categories missing.]

As of this writing, there haven't been many analyses of domain distribution in vision data. This might be because images are harder to categorize than texts. However, you can infer a model's domains from its benchmark performance. Table 2-3 shows how two models, CLIP and Open CLIP, perform on different benchmarks. These benchmarks show how well these two models do on birds, flowers, cars, and a few more categories, but the world is so much bigger and more complex than these few categories.

**Table 2-3. Open CLIP and CLIP's performance on different image datasets.**

| Dataset | CLIP Accuracy of ViT-B/32 (OpenAI) | Open CLIP Accuracy of ViT-B/32 (Cade) |
|---|---|---|
| ImageNet | 63.2 | 62.9 |
| ImageNet v2 | – | 62.6 |
| Birdsnap | 37.8 | 46.0 |
| Country211 | 17.8 | 14.8 |
| Oxford 102 Category Flower | 66.7 | 66.0 |
| German Traffic Sign Recognition Benchmark | 32.2 | 42.0 |
| Stanford Cars | 59.4 | 79.3 |
| UCF101 | 64.5 | 63.1 |

Even though general-purpose foundation models can answer everyday questions about different domains, they are unlikely to perform well on domain-specific tasks, especially if they never saw these tasks during training. Two examples of domain-specific tasks are drug discovery and cancer screening. Drug discovery involves protein, DNA, and RNA data, which follow specific formats and are expensive to acquire. This data is unlikely to be found in publicly available internet data. Similarly, cancer screening typically involves X-ray and fMRI (functional magnetic resonance imaging) scans, which are hard to obtain due to privacy.

To train a model to perform well on these domain-specific tasks, you might need to curate very specific datasets. One of the most famous domain-specific models is perhaps DeepMind's AlphaFold, trained on the sequences and 3D structures of around 100,000 known proteins. NVIDIA's BioNeMo is another model that focuses on biomolecular data for drug discovery. Google's Med-PaLM2 combined the power of an LLM with medical data to answer medical queries with higher accuracy.

> **TIP**
> Domain-specific models are especially common for biomedicine, but other fields can benefit from domain-specific models too. It's possible that a model trained on architectural sketches can help architects much better than Stable Diffusion, or a model trained on factory plans can be optimized for manufacturing processes much better than a generic model like ChatGPT.

This section gave a high-level overview of how training data impacts a model's performance. Next, let's explore the impact of how a model is designed on its performance.

## Modeling

Before training a model, developers need to decide what the model should look like. What architecture should it follow? How many parameters should it have? These decisions impact not only the model's capabilities but also its usability for downstream applications. For example, a 7B-parameter model will be vastly easier to deploy than a 175B-parameter model. Similarly, optimizing a transformer model for latency is very different from optimizing another architecture. Let's explore the factors behind these decisions.

### Model Architecture

As of this writing, the most dominant architecture for language-based foundation models is the transformer architecture (Vaswani et al., 2017), which is based on the attention mechanism. It addresses many limitations of the previous architectures, which contributed to its popularity. However, the transformer architecture has its own limitations. This section analyzes the transformer architecture and its alternatives. Because it goes into the technical details of different architectures, it can be technically dense. If you find any part too deep in the weeds, feel free to skip it.

#### Transformer architecture

To understand the transformer, let's look at the problem it was created to solve. The transformer architecture was popularized on the heels of the success of the seq2seq (sequence-to-sequence) architecture. At the time of its introduction in 2014, seq2seq provided significant improvement on then-challenging tasks: machine translation and summarization. In 2016, Google incorporated seq2seq into Google Translate, an update that they claimed to have given them the "largest improvements to date for machine translation quality". This generated a lot of interest in seq2seq, making it the go-to architecture for tasks involving sequences of text.

At a high level, seq2seq contains an encoder that processes inputs and a decoder that generates outputs. Both inputs and outputs are sequences of tokens, hence the name. Seq2seq uses RNNs (recurrent neural networks) as its encoder and decoder. In its most basic form, the encoder processes the input tokens sequentially, outputting the final hidden state that represents the input. The decoder then generates output tokens sequentially, conditioned on both the final hidden state of the input and the previously generated token. A visualization of the seq2seq architecture is shown in the top half of Figure 2-4.

[Image: Figure 2-4. Seq2seq architecture versus transformer architecture. For the transformer architecture, the arrows show the tokens that the decoder attends to when generating each output token.]

There are two problems with seq2seq that Vaswani et al. (2017) addresses. First, the vanilla seq2seq decoder generates output tokens using only the final hidden state of the input. Intuitively, this is like generating answers about a book using the book summary. This limits the quality of the generated outputs. Second, the RNN encoder and decoder mean that both input processing and output generation are done sequentially, making it slow for long sequences. If an input is 200 tokens long, seq2seq has to wait for each input token to finish processing before moving on to the next.

The transformer architecture addresses both problems with the attention mechanism. The attention mechanism allows the model to weigh the importance of different input tokens when generating each output token. This is like generating answers by referencing any page in the book. A simplified visualization of the transformer architecture is shown in the bottom half of Figure 2-4.

> **NOTE**
> While the attention mechanism is often associated with the transformer model, it was introduced three years before the transformer paper. The attention mechanism can also be used with other architectures. Google used the attention mechanism with their seq2seq architecture in 2016 for their GNMT (Google Neural Machine Translation) model. However, it wasn't until the transformer paper showed that the attention mechanism could be used without RNNs that it took off.

The transformer architecture dispenses with RNNs entirely. With transformers, the input tokens can be processed in parallel, significantly speeding up input processing. While the transformer removes the sequential input bottleneck, transformer-based autoregressive language models still have the sequential output bottleneck.

Inference for transformer-based language models, therefore, consists of two steps:

**Prefill**
The model processes the input tokens in parallel. This step creates the intermediate state necessary to generate the first output token. This intermediate state includes the key and value vectors for all input tokens.

**Decode**
The model generates one output token at a time.

As explored later in Chapter 9, the parallelizable nature of prefilling and the sequential aspect of decoding both motivate many optimization techniques to make language model inference cheaper and faster.

#### Attention mechanism

At the heart of the transformer architecture is the attention mechanism. Understanding this mechanism is necessary to understand how transformer models work. Under the hood, the attention mechanism leverages key, value, and query vectors:

- The **query vector (Q)** represents the current state of the decoder at each decoding step. Using the same book summary example, this query vector can be thought of as the person looking for information to create a summary.
- Each **key vector (K)** represents a previous token. If each previous token is a page in the book, each key vector is like the page number. Note that at a given decoding step, previous tokens include both input tokens and previously generated tokens.
- Each **value vector (V)** represents the actual value of a previous token, as learned by the model. Each value vector is like the page's content.

The attention mechanism computes how much attention to give an input token by performing a dot product between the query vector and its key vector. A high score means that the model will use more of that page's content (its value vector) when generating the book's summary. A visualization of the attention mechanism with the key, value, and query vectors is shown in Figure 2-5. In this visualization, the query vector is seeking information from the previous tokens *How, are, you, ?, ¿* to generate the next token.

[Image: Figure 2-5. An example of the attention mechanism in action next to its high-level visualization from the famous transformer paper, "Attention Is All You Need" (Vaswani et al., 2017).]

Because each previous token has a corresponding key and value vector, the longer the sequence, the more key and value vectors need to be computed and stored. This is one reason why it's so hard to extend context length for transformer models. How to efficiently compute and store key and value vectors comes up again in Chapters 7 and 9.

Let's look into how the attention function works. Given an input $x$, the key, value, and query vectors are computed by applying key, value, and query matrices to the input. Let $W_K$, $W_V$, and $W_Q$ be the key, value, and query matrices. The key, value, and query vectors are computed as follows:

$$K = xW_K$$
$$V = xW_V$$
$$Q = xW_Q$$

The query, key, and value matrices have dimensions corresponding to the model's hidden dimension. For example, in Llama 2-7B (Touvron et al., 2023), the model's hidden dimension size is 4096, meaning that each of these matrices has a 4096 × 4096 dimension. Each resulting $K$, $V$, $Q$ vector has the dimension of 4096.

The attention mechanism is almost always multi-headed. Multiple heads allow the model to attend to different groups of previous tokens simultaneously. With multi-headed attention, the query, key, and value vectors are split into smaller vectors, each corresponding to an attention head. In the case of Llama 2-7B, because it has 32 attention heads, each $K$, $V$, and $Q$ vector will be split into 32 vectors of the dimension 128. This is because $4096 / 32 = 128$.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

The outputs of all attention heads are then concatenated. An output projection matrix is used to apply another transformation to this concatenated output before it's fed to the model's next computation step. The output projection matrix has the same dimension as the model's hidden dimension.

#### Transformer block

Now that we've discussed how attention works, let's see how it's used in a model. A transformer architecture is composed of multiple transformer blocks. The exact content of the block varies between models, but, in general, each transformer block contains the attention module and the MLP (multi-layer perceptron) module:

- **Attention module**: Each attention module consists of four weight matrices: query, key, value, and output projection.
- **MLP module**: An MLP module consists of linear layers separated by nonlinear activation functions. Each linear layer is a weight matrix that is used for linear transformations, whereas an activation function allows the linear layers to learn nonlinear patterns. A linear layer is also called a feedforward layer.

Common nonlinear functions are ReLU, Rectified Linear Unit (Agarap, 2018), and GELU (Hendrycks and Gimpel, 2016), which was used by GPT-2 and GPT-3, respectively. Activation functions are very simple. For example, all ReLU does is convert negative values to 0. Mathematically, it's written as:

$$\text{ReLU}(x) = \max(0, x)$$

The number of transformer blocks in a transformer model is often referred to as that model's number of layers. A transformer-based language model is also outfitted with a module before and after all the transformer blocks:

- **An embedding module before the transformer blocks**: This module consists of the embedding matrix and the positional embedding matrix, which convert tokens and their positions into embedding vectors, respectively. Naively, the number of position indices determines the model's maximum context length. For example, if a model keeps track of 2,048 positions, its maximum context length is 2,048. However, there are techniques that increase a model's context length without increasing the number of position indices.
- **An output layer after the transformer blocks**: This module maps the model's output vectors into token probabilities used to sample model outputs (discussed in "Sampling"). This module typically consists of one matrix, which is also called the unembedding layer. Some people refer to the output layer as the model head, as it's the model's last layer before output generation.

Figure 2-6 visualizes a transformer model architecture. The size of a transformer model is determined by the dimensions of its building blocks. Some of the key values are:

- The model's dimension determines the sizes of the key, query, value, and output projection matrices in the transformer block.
- The number of transformer blocks.
- The dimension of the feedforward layer.
- The vocabulary size.

[Image: Figure 2-6. A visualization of the weight composition of a transformer model.]

Larger dimension values result in larger model sizes. Table 2-4 shows these dimension values for different Llama 2 (Touvron et al., 2023) and Llama 3 (Dubey et al., 2024) models. Note that while the increased context length impacts the model's memory footprint, it doesn't impact the model's total number of parameters.

**Table 2-4. The dimension values of different Llama models.**

| Model | # transformer blocks | Model dim | Feedforward dim | Vocab size |
|---|---|---|---|---|
| Llama 2-7B | 32 | 4,096 | 11,008 | 32K |
| Llama 2-13B | 40 | 5,120 | 13,824 | 32K |
| Llama 2-70B | 80 | 8,192 | 22,016 | 32K |
| Llama 3-7B | 32 | 4,096 | 14,336 | 128K |
| Llama 3-70B | 80 | 8,192 | 28,672 | 128K |
| Llama 3-405B | 126 | 16,384 | 53,248 | 128K |

#### Other model architectures

While the transformer model dominates the landscape, it's not the only architecture. Since AlexNet revived the interest in deep learning in 2012, many architectures have gone in and out of fashion. Seq2seq was in the limelight for four years (2014–2018). GANs (generative adversarial networks) captured the collective imagination a bit longer (2014–2019). Compared to architectures that came before it, the transformer is sticky. It's been around since 2017. How long until something better comes along?

Developing a new architecture to outperform transformers isn't easy. The transformer has been heavily optimized since 2017. A new architecture that aims to replace the transformer will have to perform at the scale that people care about, on the hardware that people care about.

However, there's hope. While transformer-based models are dominating, as of this writing, several alternative architectures are gaining traction.

One popular model is RWKV (Peng et al., 2023), an RNN-based model that can be parallelized for training. Due to its RNN nature, in theory, it doesn't have the same context length limitation that transformer-based models have. However, in practice, having no context length limitation doesn't guarantee good performance with long context.

Modeling long sequences remains a core challenge in developing LLMs. An architecture that has shown a lot of promise in long-range memory is SSMs (state space models) (Gu et al., 2021a). Since the architecture's introduction in 2021, multiple techniques have been introduced to make the architecture more efficient, better at long sequence processing, and scalable to larger model sizes. Here are a few of these techniques, to illustrate the evolution of a new architecture:

- **S4**, introduced in "Efficiently Modeling Long Sequences with Structured State Spaces" (Gu et al., 2021b), was developed to make SSMs more efficient.
- **H3**, introduced in "Hungry Hungry Hippos: Towards Language Modeling with State Space Models" (Fu et al., 2022), incorporates a mechanism that allows the model to recall early tokens and compare tokens across sequences. This mechanism's purpose is akin to that of the attention mechanism in the transformer architecture, but it is more efficient.
- **Mamba**, introduced in "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu and Dao, 2023), scales SSMs to three billion parameters. On language modeling, Mamba-3B outperforms transformers of the same size and matches transformers twice its size. The authors also show that Mamba's inference computation scales linearly with sequence length (compared to quadratic scaling for transformers). Its performance shows improvement on real data up to million-length sequences.
- **Jamba**, introduced in "Jamba: A Hybrid Transformer–Mamba Language Model" (Lieber et al., 2024), interleaves blocks of transformer and Mamba layers to scale up SSMs even further. The authors released a mixture-of-experts model with 52B total available parameters (12B active parameters) designed to fit in a single 80 GB GPU. Jamba shows strong performance on standard language model benchmarks and long-context evaluations for up to a context length of 256K tokens. It also has a small memory footprint compared to vanilla transformers.

Figure 2-7 visualizes the transformer, Mamba, and Jamba blocks.

While it's challenging to develop an architecture that outperforms the transformer, given its many limitations, there are a lot of incentives to do so. If another architecture does indeed overtake the transformer, some of the model adaptation techniques discussed in this book might change. However, just as the shift from ML engineering to AI engineering has kept many things unchanged, changing the underlying model architecture won't alter the fundamental approaches.

[Image: Figure 2-7. A visualization of the transformer, Mamba, and Jamba layers. Image adapted from "Jamba: A Hybrid Transformer–Mamba Language Model" (Lieber et al., 2024).]

### Model Size

Much of AI progress in recent years can be attributed to increased model size. It's hard to talk about foundation models without talking about their number of parameters. The number of parameters is usually appended at the end of a model name. For example, Llama-13B refers to the version of Llama, a model family developed by Meta, with 13 billion parameters.

In general, increasing a model's parameters increases its capacity to learn, resulting in better models. Given two models of the same model family, the one with 13 billion parameters is likely to perform much better than the one with 7 billion parameters.

> **NOTE**
> As the community better understands how to train large models, newer-generation models tend to outperform older-generation models of the same size. For example, Llama 3-8B (2024) outperforms even Llama 2-70B (2023) on the MMLU benchmark.

The number of parameters helps us estimate the compute resources needed to train and run this model. For example, if a model has 7 billion parameters, and each parameter is stored using 2 bytes (16 bits), then we can calculate that the GPU memory needed to do inference using this model will be at least 14 billion bytes (14 GB).

The number of parameters can be misleading if the model is sparse. A sparse model has a large percentage of zero-value parameters. A 7B-parameter model that is 90% sparse only has 700 million non-zero parameters. Sparsity allows for more efficient data storage and computation. This means that a large sparse model can require less compute than a small dense model.

A type of sparse model that has gained popularity in recent years is mixture-of-experts (MoE) (Shazeer et al., 2017). An MoE model is divided into different groups of parameters, and each group is an expert. Only a subset of the experts is active for (used to) process each token.

For example, Mixtral 8x7B is a mixture of eight experts, each expert with seven billion parameters. If no two experts share any parameter, it should have $8 \times 7$ billion = 56 billion parameters. However, due to some parameters being shared, it has only 46.7 billion parameters.

At each layer, for each token, only two experts are active. This means that only 12.9 billion parameters are active for each token. While this model has 46.7 billion parameters, its cost and speed are the same as a 12.9-billion-parameter model.

A larger model can also underperform a smaller model if it's not trained on enough data. Imagine a 13B-param model trained on a dataset consisting of a single sentence: "I like pineapples." This model will perform much worse than a much smaller model trained on more data.

When discussing model size, it's important to consider the size of the data it was trained on. For most models, dataset sizes are measured by the number of training samples. For example, Google's Flamingo (Alayrac et al., 2022) was trained using four datasets—one of them has 1.8 billion (image, text) pairs and one has 312 million (image, text) pairs.

For language models, a training sample can be a sentence, a Wikipedia page, a chat conversation, or a book. A book is worth a lot more than a sentence, so the number of training samples is no longer a good metric to measure dataset sizes. A better measurement is the number of tokens in the dataset.

The number of tokens isn't a perfect measurement either, as different models can have different tokenization processes, resulting in the same dataset having different numbers of tokens for different models. Why not just use the number of words or the number of letters? Because a token is the unit that a model operates on, knowing the number of tokens in a dataset helps us measure how much a model can potentially learn from that data.

As of this writing, LLMs are trained using datasets in the order of trillions of tokens. Meta used increasingly larger datasets to train their Llama models:

- 1.4 trillion tokens for Llama 1
- 2 trillion tokens for Llama 2
- 15 trillion tokens for Llama 3

Together's open source dataset RedPajama-v2 has 30 trillion tokens. This is equivalent to 450 million books or 5,400 times the size of Wikipedia. However, since RedPajama-v2 consists of indiscriminate content, the amount of high-quality data is much lower.

The number of tokens in a model's dataset isn't the same as its number of training tokens. The number of training tokens measures the tokens that the model is trained on. If a dataset contains 1 trillion tokens and a model is trained on that dataset for two epochs—an epoch is a pass through the dataset—the number of training tokens is 2 trillion. See Table 2-5 for examples of the number of training tokens for models with different numbers of parameters.

**Table 2-5. Examples of the number of training tokens for models with different numbers of parameters. Source: "Training Compute-Optimal Large Language Models" (DeepMind, 2022).**

| Model | Size (# parameters) | Training tokens |
|---|---|---|
| LaMDA (Thoppilan et al., 2022) | 137 billion | 168 billion |
| GPT-3 (Brown et al., 2020) | 175 billion | 300 billion |
| Jurassic (Lieber et al., 2021) | 178 billion | 300 billion |
| Gopher (Rae et al., 2021) | 280 billion | 300 billion |
| MT-NLG 530B (Smith et al., 2022) | 530 billion | 270 billion |
| Chinchilla | 70 billion | 1.4 trillion |

> **NOTE**
> While this section focuses on the scale of data, quantity isn't the only thing that matters. Data quality and data diversity matter, too. Quantity, quality, and diversity are the three golden goals for training data. They are discussed further in Chapter 8.

Pre-training large models requires compute. One way to measure the amount of compute needed is by considering the number of machines, e.g., GPUs, CPUs, and TPUs. However, different machines have very different capacities and costs. An NVIDIA A10 GPU is different from an NVIDIA H100 GPU and an Intel Core Ultra Processor.

A more standardized unit for a model's compute requirement is FLOP, or floating point operation. FLOP measures the number of floating point operations performed for a certain task. Google's largest PaLM-2 model, for example, was trained using $10^{23}$ FLOPs (Chowdhery et al., 2022). GPT-3-175B was trained using $3.14 \times 10^{18}$ FLOPs (Brown et al., 2020).

The plural form of FLOP, FLOPs, is often confused with FLOP/s, floating point operations per Second. FLOPs measure the compute requirement for a task, whereas FLOP/s measures a machine's peak performance. For example, an NVIDIA H100 NVL GPU can deliver a maximum of 60 TeraFLOP/s: $6 \times 10^{13}$ FLOPs a second or $5.2 \times 10^{18}$ FLOPs a day.

> **WARNING**
> Be alert for confusing notations. FLOP/s is often written as FLOPS, which looks similar to FLOPs. To avoid this confusion, some companies, including OpenAI, use FLOP/s-day in place of FLOPs to measure compute requirements:
> 1 FLOP/s-day = $60 \times 60 \times 24$ = 86,400 FLOPs
> This book uses FLOPs for counting floating point operations and FLOP/s for FLOPs per second.

Assume that you have 256 H100s. If you can use them at their maximum capacity and make no training mistakes, it'd take you $(3.14 \times 10^{18}) / (256 \times 5.2 \times 10^{18}) = \sim 236$ days, or approximately 7.8 months, to train GPT-3-175B.

However, it's unlikely you can use your machines at their peak capacity all the time. Utilization measures how much of the maximum compute capacity you can use. What's considered good utilization depends on the model, the workload, and the hardware. Generally, if you can get half the advertised performance, 50% utilization, you're doing okay. Anything above 70% utilization is considered great. Don't let this rule stop you from getting even higher utilization. Chapter 9 discusses hardware metrics and utilization in more detail.

At 70% utilization and $2/h for one H100, training GPT-3-175B would cost over $4 million:

$$\$2/\text{H100/hour} \times 256 \text{ H100} \times 24 \text{ hours} \times 256 \text{ days} / 0.7$$

> **TIP**
> In summary, three numbers signal a model's scale:
> - Number of parameters, which is a proxy for the model's learning capacity.
> - Number of tokens a model was trained on, which is a proxy for how much a model learned.
> - Number of FLOPs, which is a proxy for the training cost.

#### Inverse Scaling

We've assumed that bigger models are better. Are there scenarios for which bigger models perform worse? In 2022, Anthropic discovered that, counterintuitively, more alignment training (discussed in "Post-Training") leads to models that align less with human preference (Perez et al., 2022). According to their paper, models trained to be more aligned "are much more likely to express specific political views (pro-gun rights and immigration) and religious views (Buddhist), self-reported conscious experience and moral self-worth, and a desire to not be shut down."

In 2023, a group of researchers, mostly from New York University, launched the Inverse Scaling Prize to find tasks where larger language models perform worse. They offered $5,000 for each third prize, $20,000 for each second prize, and $100,000 for one first prize. They received a total of 99 submissions, of which 11 were awarded third prizes. They found that larger language models are sometimes (only sometimes) worse on tasks that require memorization and tasks with strong priors. However, they didn't award any second or first prizes because even though the submitted tasks show failures for a small test set, none demonstrated failures in the real world.

#### Scaling law: Building compute-optimal models

I hope that the last section has convinced you of three things:

1. Model performance depends on the model size and the dataset size.
2. Bigger models and bigger datasets require more compute.
3. Compute costs money.

Unless you have unlimited money, budgeting is essential. You don't want to start with an arbitrarily large model size and see how much it would cost. You start with a budget—how much money you want to spend—and work out the best model performance you can afford. As compute is often the limiting factor—compute infrastructure is not only expensive but also hard to set up—teams often start with a compute budget. Given a fixed amount of FLOPs, what model size and dataset size would give the best performance? A model that can achieve the best performance given a fixed compute budget is compute-optimal.

Given a compute budget, the rule that helps calculate the optimal model size and dataset size is called the Chinchilla scaling law, proposed in the Chinchilla paper "Training Compute-Optimal Large Language Models" (DeepMind, 2022). To study the relationship between model size, dataset size, compute budget, and model performance, the authors trained 400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens. They found that for compute-optimal training, you need the number of training tokens to be approximately 20 times the model size. This means that a 3B-parameter model needs approximately 60B training tokens. The model size and the number of training tokens should be scaled equally: for every doubling of the model size, the number of training tokens should also be doubled.

We've come a long way from when the training process was treated like alchemy. Figure 2-8 shows that we can predict not only the optimal number of parameters and tokens for each FLOP budget but also the expected training loss from these settings (assuming we do things right).

This compute-optimal calculation assumes that the cost of acquiring data is much cheaper than the cost of compute. The same Chinchilla paper proposes another calculation for when the cost of training data is nontrivial.

[Image: Figure 2-8. Graphs that depict the relationships between training loss, a model's number of parameters, FLOPs, and number of training tokens. Source: "Training Compute-Optimal Large Language Models" (DeepMind, 2022).]

The scaling law was developed for dense models trained on predominantly human-generated data. Adapting this calculation for sparse models, such as mixture-of-expert models, and synthetic data is an active research area.

The scaling law optimizes model quality given a compute budget. However, it's important to remember that for production, model quality isn't everything. Some models, most notably Llama, have suboptimal performance but better usability. Given their compute budget, Llama authors could've chosen bigger models that would perform better, but they opted for smaller models. Smaller models are easier to work with and cheaper to run inference on, which helped their models gain wider adoption. Sardana et al. (2023) modified the Chinchilla scaling law to calculate the optimal LLM parameter count and pre-training data size to account for this inference demand.

On the topic of model performance given a compute budget, it's worth noting that the cost of achieving a given model performance is decreasing. For example, on the ImageNet dataset, the cost to achieve 93% accuracy halved from 2019 to 2021, according to the Artificial Intelligence Index Report 2022 (Stanford University HAI).

While the cost for the same model performance is decreasing, the cost for model performance improvement remains high. Similar to the last mile challenge discussed in Chapter 1, improving a model's accuracy from 90 to 95% is more expensive than improving it from 85 to 90%. As Meta's paper "Beyond Neural Scaling Laws: Beating Power Law Scaling via Data Pruning" pointed out, this means a model with a 2% error rate might require an order of magnitude more data, compute, or energy than a model with a 3% error rate.

In language modeling, a drop in cross entropy loss from about 3.4 to 2.8 nats requires 10 times more training data. Cross entropy and its units, including nats, are discussed in Chapter 3. For large vision models, increasing the number of training samples from 1 billion to 2 billion leads to an accuracy gain on ImageNet of only a few percentage points.

However, small performance changes in language modeling loss or ImageNet accuracy can lead to big differences in the quality of downstream applications. If you switch from a model with a cross-entropy loss of 3.4 to one with a loss of 2.8, you'll notice a difference.

#### Scaling extrapolation

The performance of a model depends heavily on the values of its hyperparameters. When working with small models, it's a common practice to train a model multiple times with different sets of hyperparameters and pick the best-performing one. This is, however, rarely possible for large models as training them once is resource-draining enough.

> **PARAMETER VERSUS HYPERPARAMETER**
> A parameter can be learned by the model during the training process. A hyperparameter is set by users to configure the model and control how the model learns. Hyperparameters to configure the model include the number of layers, the model dimension, and vocabulary size. Hyperparameters to control how a model learns include batch size, number of epochs, learning rate, per-layer initial variance, and more.

This means that for many models, you might have only one shot of getting the right set of hyperparameters. As a result, scaling extrapolation (also called hyperparameter transferring) has emerged as a research subfield that tries to predict, for large models, what hyperparameters will give the best performance. The current approach is to study the impact of hyperparameters on models of different sizes, usually much smaller than the target model size, and then extrapolate how these hyperparameters would work on the target model size. A 2022 paper by Microsoft and OpenAI shows that it was possible to transfer hyperparameters from a 40M model to a 6.7B model.

Scaling extrapolation is still a niche topic, as few people have the experience and resources to study the training of large models. It's also difficult to do due to the sheer number of hyperparameters and how they interact with each other. If you have ten hyperparameters, you'd have to study 1,024 hyperparameter combinations. You would have to study each hyperparameter individually, then two of them together, and three of them together, and so on.

In addition, emergent abilities (Wei et al., 2022) make the extrapolation less accurate. Emergent abilities refer to those that are only present at scale might not be observable on smaller models trained on smaller datasets. To learn more about scaling extrapolation, check out this excellent blog post: "On the Difficulty of Extrapolation with NN Scaling" (Luke Metz, 2022).

#### Scaling bottlenecks

Until now, every order of magnitude increase in model size has led to an increase in model performance. GPT-2 has an order of magnitude more parameters than GPT-1 (1.5 billion versus 117 million). GPT-3 has two orders of magnitude more than GPT-2 (175 billion versus 1.5 billion). This means a three-orders-of-magnitude increase in model sizes between 2018 and 2021. Three more orders of magnitude growth would result in 100-trillion-parameter models.

How many more orders of magnitude can model sizes grow? Would there be a point where the model performance plateaus regardless of its size? While it's hard to answer these questions, there are already two visible bottlenecks for scaling: training data and electricity.

Foundation models use so much data that there's a realistic concern we'll run out of internet data in the next few years. The rate of training dataset size growth is much faster than the rate of new data being generated (Villalobos et al., 2022), as illustrated in Figure 2-9. If you've ever put anything on the internet, you should assume that it already is or will be included in the training data for some language models, whether you consent or not. This is similar to how, if you post something on the internet, you should expect it to be indexed by Google.

[Image: Figure 2-9. Projection of historical trend of training dataset sizes and available data stock. Source: Villalobos et al., 2024.]

Some people are leveraging this fact to inject data they want into the training data of future models. They do this simply by publishing the text they want on the internet, hoping it will influence future models to generate the responses they desire. Bad actors can also leverage this approach for prompt injection attacks, as discussed in Chapter 5.

> **NOTE**
> An open research question is how to make a model forget specific information it has learned during training. Imagine you published a blog post that you eventually deleted. If that blog post was included in a model's training data, the model might still reproduce the post's content. As a result, people could potentially access removed content without your consent.

On top of that, the internet is being rapidly populated with data generated by AI models. If companies continue using internet data to train future models, these new models will be partially trained on AI-generated data.