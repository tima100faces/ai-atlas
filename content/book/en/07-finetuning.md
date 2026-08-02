---
id: "07-finetuning"
title: "7. Finetuning"
language: "en"
---

<!-- cleaned -->
# 7. Finetuning

Finetuning is the process of adapting a model to a specific task by further training the whole model or part of the model. Chapters 5 and 6 discuss prompt-based methods, which adapt a model by giving it instructions, context, and tools. Finetuning adapts a model by adjusting its weights.

Finetuning can enhance various aspects of a model. It can improve the model’s domain-specific capabilities, such as coding or medical question answering, and can also strengthen its safety. However, it is most often used to improve the model’s instruction-following ability, particularly to ensure it adheres to specific output styles and formats.

While finetuning can help create models that are more customized to your needs, it also requires more up-front investment. A question I hear very often is when to finetune and when to do RAG. After an overview of finetuning, this chapter will discuss the reasons for finetuning and the reasons for not finetuning, as well as a simple framework for thinking about choosing between finetuning and alternate methods.

Compared to prompt-based methods, finetuning incurs a much higher memory footprint. At the scale of today’s foundation models, naive finetuning often requires more memory than what’s available on a single GPU. This makes finetuning expensive and challenging to do. As discussed throughout this chapter, reducing memory requirements is a primary motivation for many finetuning techniques. This chapter dedicates one section to outlining factors contributing to a model’s memory footprint, which is important for understanding these techniques.

A memory-efficient approach that has become dominant in the finetuning space is PEFT (parameter-efficient finetuning). This chapter explores PEFT and how it differs from traditional finetuning; this chapter also provides an overview of its evolving techniques. I’ll focus particularly on one compelling category: adapter-based techniques.

With prompt-based methods, knowledge about how ML models operate under the hood is recommended but not strictly necessary. However, finetuning brings you to the realm of model training, where ML knowledge is required. ML basics are beyond the scope of this book. If you want a quick refresh, the book’s GitHub repository has pointers to helpful resources. In this chapter, I’ll cover a few core concepts immediately relevant to the discussion.

This chapter is the most technically challenging one for me to write, not because of the complexity of the concepts, but because of the broad scope these concepts cover. I suspect it might also be technically challenging to read. If, at any point, you feel like you’re diving too deep into details that aren’t relevant to your work, feel free to skip.

There’s a lot to discuss. Let’s dive in!

## Finetuning Overview

To finetune, you start with a base model that has some, but not all, of the capabilities you need. The goal of finetuning is to get this model to perform well enough for your specific task.

Finetuning is one way to do transfer learning, a concept first introduced by Bozinovski and Fulgosi in 1976. Transfer learning focuses on how to transfer the knowledge gained from one task to accelerate learning for a new, related task. This is conceptually similar to how humans transfer skills: for example, knowing how to play the piano can make it easier to learn another musical instrument.

An early large-scale success in transfer learning was Google’s multilingual translation system (Johnson et. al, 2016). The model transferred its knowledge of Portuguese–English and English–Spanish translation to directly translate Portuguese to Spanish, even though there were no Portuguese–Spanish examples in the training data.

Since the early days of deep learning, transfer learning has offered a solution for tasks with limited or expensive training data. By training a base model on tasks with abundant data, you can then transfer that knowledge to a target task.

For LLMs, knowledge gained from pre-training on text completion (a task with abundant data) is transferred to more specialized tasks, like legal question answering or text-to-SQL, which often have less available data. This capability for transfer learning makes foundation models particularly valuable.

Transfer learning improves sample efficiency, allowing a model to learn the same behavior with fewer examples. A sample-efficient model learns effectively from fewer samples. For example, while training a model from scratch for legal question answering may need millions of examples, finetuning a good base model might only require a few hundred.

Ideally, much of what the model needs to learn is already present in the base model, and finetuning just refines the model’s behavior. OpenAI’s InstructGPT paper (2022) suggested viewing finetuning as unlocking the capabilities a model already has but that are difficult for users to access via prompting alone.

> **NOTE**
> Finetuning isn’t the only way to do transfer learning. Another approach is feature-based transfer. In this approach, a model is trained to extract features from the data, usually as embedding vectors, which are then used by another model. I mention feature-based transfer briefly in Chapter 2, when discussing how part of a foundation model can be reused for a classification task by adding a classifier head.
> 
> Feature-based transfer is very common in computer vision. For instance, in the second half of the 2010s, many people used models trained on the ImageNet dataset to extract features from images and use these features in other computer vision tasks such as object detection or image segmentation.

Finetuning is part of a model’s training process. It’s an extension of model pre-training. Because any training that happens after pre-training is finetuning, finetuning can take many different forms. Chapter 2 already discussed two types of finetuning: supervised finetuning and preference finetuning. Let’s do a quick recap of these methods and how you might leverage them as an application developer.

Recall that a model’s training process starts with pre-training, which is usually done with self-supervision. Self-supervision allows the model to learn from a large amount of unlabeled data. For language models, self-supervised data is typically just sequences of text that don’t need annotations.

Before finetuning this pre-trained model with expensive task-specific data, you can finetune it with self-supervision using cheap task-related data. For example, to finetune a model for legal question answering, before finetuning it on expensive annotated (question, answer) data, you can finetune it on raw legal documents. Similarly, to finetune a model to do book summarization in Vietnamese, you can first finetune it on a large collection of Vietnamese text. Self-supervised finetuning is also called continued pre-training.

As discussed in Chapter 1, language models can be autoregressive or masked. An autoregressive model predicts the next token in a sequence using the previous tokens as the context. A masked model fills in the blank using the tokens both before and after it. Similarly, with supervised finetuning, you can also finetune a model to predict the next token or fill in the blank. The latter, also known as infilling finetuning, is especially useful for tasks such as text editing and code debugging. You can finetune a model for infilling even if it was pre-trained autoregressively.

The massive amount of data a model can learn from during self-supervised learning outfits the model with a rich understanding of the world, but it might be hard for users to extract that knowledge for their tasks, or the way the model behaves might be misaligned with human preference. Supervised finetuning uses high-quality annotated data to refine the model to align with human usage and preference.

During supervised finetuning, the model is trained using (input, output) pairs: the input can be an instruction and the output can be a response. A response can be open-ended, such as for the task of book summarization. A response can be also close-ended, such as for a classification task. High-quality instruction data can be challenging and expensive to create, especially for instructions that require factual consistency, domain expertise, or political correctness. Chapter 8 discusses how to acquire instruction data.

A model can also be finetuned with reinforcement learning to generate responses that maximize human preference. Preference finetuning requires comparative data that typically follows the format (instruction, winning response, losing response).

It’s possible to finetune a model to extend its context length. Long-context finetuning typically requires modifying the model’s architecture, such as adjusting the positional embeddings. A long sequence means more possible positions for tokens, and positional embeddings should be able to handle them. Compared to other finetuning techniques, long-context finetuning is harder to do. The resulting model might also degrade on shorter sequences.

Figure 7-1 shows the making of different Code Llama models (Rozière et al., 2024), from the base model Llama 2, using different finetuning techniques. Using long-context finetuning, they were able to increase the model’s maximum context length from 4,096 tokens to 16,384 tokens to accommodate longer code files. In the image, instruction finetuning refers to supervised finetuning.

Finetuning can be done by both model developers and application developers. Model developers typically post-train a model with different finetuning techniques before releasing it. A model developer might also release different model versions, each finetuned to a different extent, so that application developers can choose the version that works best for them.

![Figure 7-1. Different finetuning techniques used to make different Code Llama models. Image from the Rozière et al. (2024). Adapted from an original image licensed under CC BY 4.0.](/assets/en/07-finetuning/p601_01.png)

As an application developer, you might finetune a pre-trained model, but most likely, you’ll finetune a model that has been post-trained. The more refined a model is and the more relevant its knowledge is to your task, the less work you’ll have to do to adapt it.

## When to Finetune

Before jumping into different finetuning techniques, it’s necessary to consider whether finetuning is the right option for you. Compared to prompt-based methods, finetuning requires significantly more resources, not just in data and hardware, but also in ML talent. Therefore, finetuning is generally attempted after extensive experiments with prompt-based methods. However, finetuning and prompting aren’t mutually exclusive. Real-world problems often require both approaches.

### Reasons to Finetune

The primary reason for finetuning is to improve a model’s quality, in terms of both general capabilities and task-specific capabilities. Finetuning is commonly used to improve a model’s ability to generate outputs following specific structures, such as JSON or YAML formats.

A general-purpose model that performs well on a wide range of benchmarks might not perform well on your specific task. If the model you want to use wasn’t sufficiently trained on your task, finetuning it with your data can be especially useful.

For example, an out-of-the-box model might be good at converting from text to the standard SQL dialect but might fail with a less common SQL dialect. In this case, finetuning this model on data containing this SQL dialect will help. Similarly, if the model works well on standard SQL for common queries but often fails for customer-specific queries, finetuning the model on customer-specific queries might help.

One especially interesting use case of finetuning is bias mitigation. The idea is that if the base model perpetuates certain biases from its training data, exposing it to carefully curated data during finetuning can counteract these biases (Wang and Russakovsky, 2023). For example, if a model consistently assigns CEOs male-sounding names, finetuning it on a dataset with many female CEOs can mitigate this bias. Garimella et al. (2022) found that finetuning BERT-like language models on text authored by women can reduce these models’ gender biases, while finetuning them on texts by African authors can reduce racial biases.

You can finetune a big model to make it even better, but finetuning smaller models is much more common. Smaller models require less memory, and, therefore, are easier to finetune. They are also cheaper and faster to use in production.

A common approach is to finetune a small model to imitate the behavior of a larger model using data generated by this large model. Because this approach distills the larger model’s knowledge into the smaller model, it’s called distillation. This is discussed in Chapter 8 together with other data synthesis techniques.

A small model, finetuned on a specific task, might outperform a much larger out-of-the-box model on that task. For example, Grammarly found that their finetuned Flan-T5 models (Chung et al., 2022) outperformed a GPT-3 variant specialized in text editing across a wide range of writing assistant tasks despite being 60 times smaller. The finetuning process used only 82,000 (instruction, output) pairs, which is smaller than the data typically needed to train a text-editing model from scratch.

In the early days of foundation models, when the strongest models were commercial with limited finetuning access, there weren’t many competitive models available for finetuning. However, as the open source community proliferates with high-quality models of all sizes, tailored for a wide variety of domains, finetuning has become a lot more viable and attractive.

### Reasons Not to Finetune

While finetuning can improve a model in many ways, many of these improvements can also be achieved, to a certain extent, without finetuning. Finetuning can improve a model’s performance, but so do carefully crafted prompts and context. Finetuning can help with structured outputs, but many other techniques, as discussed in Chapter 2, can also do that.

First, while finetuning a model for a specific task can improve its performance for that task, it can degrade its performance for other tasks. This can be frustrating when you intend this model for an application that expects diverse prompts.

Imagine you need a model for three types of queries: product recommendations, changing orders, and general feedback. Originally, the model works well for product recommendations and general feedback but poorly for changing orders. To fix this, you finetune the model on a dataset of (query, response) pairs about changing orders. The finetuned model might indeed perform better for this type of query, but worse for the two other tasks.

What do you do in this situation? You can finetune the model on all the queries you care about, not just changing orders. If you can’t seem to get a model to perform well on all your tasks, consider using separate models for different tasks. If you wish to combine these separate models into one to make serving them easier, you can also consider merging them together, as discussed later in this chapter.

If you’re just starting to experiment with a project, finetuning is rarely the first thing you should attempt. Finetuning requires high up-front investments and continual maintenance. First, you need data. Annotated data can be slow and expensive to acquire manually, especially for tasks that demand critical thinking and domain expertise. Open source data and AI-generated data can mitigate the cost, but their effectiveness is highly variable.

Second, finetuning requires the knowledge of how to train models. You need to evaluate base models to choose one to finetune. Depending on your needs and resources, options might be limited. While finetuning frameworks and APIs can automate many steps in the actual finetuning process, you still need to understand the different training knobs you can tweak, monitor the learning process, and debug when something is wrong. For example, you need to understand how an optimizer works, what learning rate to use, how much training data is needed, how to address overfitting/underfitting, and how to evaluate your models throughout the process.

Third, once you have a finetuned model, you’ll need to figure out how to serve it. Will you host it yourself or use an API service? As discussed in Chapter 9, inference optimization for large models, especially LLMs, isn’t trivial. Finetuning requires less of a technical leap if you’re already hosting your models in-house and familiar with how to operate models.

More importantly, you need to establish a policy and budget for monitoring, maintaining, and updating your model. As you iterate on your finetuned model, new base models are being developed at a rapid pace. These base models may improve faster than you can enhance your finetuned model. If a new base model outperforms your finetuned model on your specific task, how significant does the performance improvement have to be before you switch to the new base model? What if a new base model doesn’t immediately outperform your existing model but has the potential to do so after finetuning—would you experiment with it?

In many cases, switching to a better model would provide only a small incremental improvement, and your task might be given a lower priority than projects with larger returns, like enabling new use cases.

AI engineering experiments should start with prompting, following the best practices discussed in Chapter 6. Explore more advanced solutions only if prompting alone proves inadequate. Ensure you have thoroughly tested various prompts, as a model’s performance can vary greatly with different prompts.

Many practitioners I’ve spoken with share a similar story that goes like this. Someone complains that prompting is ineffective and insists on finetuning. Upon investigation, it turns out that prompt experiments were minimal and unsystematic. Instructions were unclear, examples didn’t represent actual data, and metrics were poorly defined. After refining the prompt experiment process, the prompt quality improved enough to be sufficient for their application.

> **FINETUNING DOMAIN-SPECIFIC TASKS**
> Beware of the argument that general-purpose models don’t work well for domain-specific tasks, and, therefore, you must finetune or train models for your specific tasks. As general-purpose models become more capable, they also become better at domain-specific tasks and can outperform the domain-specific models.
> 
> An interesting early specialized model is BloombergGPT, which was introduced by Bloomberg in March 2023. The strongest models on the market then were all proprietary, and Bloomberg wanted a mid-size model that performed well on financial tasks and could be hosted in-house for use cases with sensitive data. The model, with 50 billion parameters, required 1.3 million A100 GPU hours for training. The estimated cost of the compute was between $1.3 million and $2.6 million, excluding data costs (Wu et al., 2023).
> 
> In the same month, OpenAI released GPT-4-0314. Research by Li et al. (2023) demonstrated that GPT-4-0314 significantly outperformed BloombergGPT across various financial benchmarks. Table 7-1 provides details of two such benchmarks.

**Table 7-1. General-purpose models like GPT-4 can outperform financial models in financial domains.**

| Model | FiQA sentiment analysis (weighted F1) | ConvFinQA (accuracy) |
|-------|---------------------------------------|----------------------|
| GPT-4-0314 (zero-shot) | 87.15 | 76.48 |
| BloombergGPT | 75.07 | 43.41 |

Since then, several mid-size models with performance comparable to GPT-4 have been released, including Claude 3.5 Sonnet (70B parameters), Llama 3-70B-Instruct, and Qwen2-72B-Instruct. The latter two are open weight and can be self-hosted.

Because benchmarks are insufficient to capture real-world performance, it’s possible that BloombergGPT works well for Bloomberg for their specific use cases. The Bloomberg team certainly gained invaluable experience through training this model, which might enable them to better develop and operate future models.

Both finetuning and prompting experiments require systematic processes. Doing prompt experiments enables developers to build an evaluation pipeline, data annotation guideline, and experiment tracking practices that will be stepping stones for finetuning.

One benefit of finetuning, before prompt caching was introduced, was that it can help optimize token usage. The more examples you add to a prompt, the more input tokens the model will use, which increases both latency and cost. Instead of including your examples in each prompt, you can finetune a model on these examples. This allows you to use shorter prompts with the finetuned model, as shown in Figure 7-2.

With prompt caching, where repetitive prompt segments can be cached for reuse, this is no longer a strong benefit. Prompt caching is discussed further in Chapter 9. However, the number of examples you can use with a prompt is still limited by the maximum context length. With finetuning, there’s no limit to how many examples you can use.

![Figure 7-2. Instead of including examples in each prompt, which increases cost and latency, you finetune a model on these examples.](/assets/en/07-finetuning/p611_01.png)

### Finetuning and RAG

Once you’ve maximized the performance gains from prompting, you might wonder whether to do RAG or finetuning next. The answer depends on whether your model’s failures are information-based or behavior-based.

If the model fails because it lacks information, a RAG system that gives the model access to the relevant sources of information can help. Information-based failures happen when the outputs are factually wrong or outdated. Here are two example scenarios in which information-based failures happen:

- **The model doesn’t have the information.** Public models are unlikely to have information private to you or your organization. When a model doesn’t have the information, it either tells you so or hallucinates an answer.
- **The model has outdated information.** If you ask: “How many studio albums has Taylor Swift released?” and the correct answer is 11, but the model answers 10, it can be because the model’s cut-off date was before the release of the latest album.

The paper “Fine-Tuning or Retrieval?” by Ovadia et al. (2024) demonstrated that for tasks that require up-to-date information, such as questions about current events, RAG outperformed finetuned models. Not only that, RAG with the base model outperformed RAG with finetuned models, as shown in Table 7-2. This finding indicates that while finetuning can enhance a model’s performance on a specific task, it may also lead to a decline in performance in other areas.

**Table 7-2. RAG outperforms finetuning on a question-answering task about current events, curated by different finetuning approaches the author used.**

| Model | Base model | Base model + RAG | FT-reg | FT-p |
|-------|------------|------------------|--------|------|
| Mistral-7B | 0.481 | 0.875 | 0.504 | 0.588 |
| Llama 2-7B | 0.353 | 0.585 | 0.219 | 0.392 |
| Orca 2-7B | 0.456 | 0.876 | 0.511 | 0.566 |

On the other hand, if the model has behavioral issues, finetuning might help. One behavioral issue is when the model’s outputs are factually correct but irrelevant to the task. For example, you ask the model to generate technical specifications for a software project to provide to your engineering teams. While accurate, the generated specs lack the details your teams need. Finetuning the model with well-defined technical specifications can make the outputs more relevant.

Another issue is when it fails to follow the expected output format. For example, if you asked the model to write HTML code, but the generated code didn’t compile, it might be because the model wasn’t sufficiently exposed to HTML in its training data. You can correct this by exposing the model to more HTML code during finetuning.

Semantic parsing is a category of tasks whose success hinges on the model’s ability to generate outputs in the expected format and, therefore, often requires finetuning. Semantic parsing is discussed briefly in Chapters 2 and 6. As a reminder, semantic parsing means converting natural language into a structured format like JSON. Strong off-the-shelf models are generally good for common, less complex syntaxes like JSON, YAML, and regex. However, they might not be as good for syntaxes with fewer available examples on the internet, such as a domain-specific language for a less popular tool or a complex syntax.

In short, finetuning is for form, and RAG is for facts. A RAG system gives your model external knowledge to construct more accurate and informative answers. A RAG system can help mitigate your model’s hallucinations. Finetuning, on the other hand, helps your model understand and follow syntaxes and styles. While finetuning can potentially reduce hallucinations if done with enough high-quality data, it can also worsen hallucinations if the data quality is low.

If your model has both information and behavior issues, start with RAG. RAG is typically easier since you won’t have to worry about curating training data or hosting the finetuned models. When doing RAG, start with simple term-based solutions such as BM25 instead of jumping straight into something that requires vector databases.

RAG can also introduce a more significant performance boost than finetuning. Ovadia et al. (2024) showed that for almost all question categories in the MMLU benchmark, RAG outperforms finetuning for three different models: Mistral 7B, Llama 2-7B, and Orca 2-7B.

However, RAG and finetuning aren’t mutually exclusive. They can sometimes be used together to maximize your application’s performance. In the same experiment, Ovadia et al. (2024) showed that incorporating RAG on top of a finetuned model can boost its performance on the MMLU benchmark 43% of the time. It’s important to note that in this experiment, using RAG with finetuned models doesn’t improve the performance 57% of the time, compared to using RAG alone.

There’s no universal workflow for all applications. Figure 7-3 shows some paths an application development process might follow over time. The arrow indicates what next step you might try. This figure is inspired by an example workflow shown by OpenAI (2023).

![Figure 7-3. Example application development flows. After simple retrieval (such as term-based retrieval), whether to experiment with more complex retrieval (such as hybrid search) or finetuning depends on each application and its failure modes.](/assets/en/07-finetuning/p616_01.png)

So the workflow to adapt a model to a task might work as follows. Note that before any of the adaptation steps, you should define your evaluation criteria and design your evaluation pipeline, as discussed in Chapter 4. This evaluation pipeline is what you’ll use to benchmark your progress as you develop your application. Evaluation doesn’t happen only in the beginning. It should be present during every step of the process:

1. Try to get a model to perform your task with prompting alone. Use the prompt engineering best practices covered in Chapter 5, including systematically versioning your prompts.
2. Add more examples to the prompt. Depending on the use case, the number of examples needed might be between 1 and 50.
3. If your model frequently fails due to missing information, connect it to data sources that can supply relevant information. When starting with RAG, begin by using basic retrieval methods like term-based search. Even with simple retrieval, adding relevant and accurate knowledge should lead to some improvement in your model’s performance.
4. Depending on your model’s failure modes, you might explore one of these next steps:
   a. If the model continues having information-based failures, you might want to try even more advanced RAG methods, such as embedding-based retrieval.
   b. If the model continues having behavioral issues, such as it keeps generating irrelevant, malformatted, or unsafe responses, you can opt for finetuning. Embedding-based retrieval increases inference complexity by introducing additional components into the pipeline, while finetuning increases the complexity of model development but leaves inference unchanged.
5. Combine both RAG and finetuning for even more performance boost.

If, after considering all the pros and cons of finetuning and other alternate techniques, you decide to finetune your model, the rest of the chapter is for you. First, let’s look into the number one challenge of finetuning: its memory bottleneck.

## Memory Bottlenecks

Because finetuning is memory-intensive, many finetuning techniques aim to minimize their memory footprint. Understanding what causes this memory bottleneck is necessary to understand why and how these techniques work. This understanding, in turn, can help you select a finetuning method that works best for you.

Besides explaining finetuning’s memory bottleneck, this section also introduces formulas for back-of-the-napkin calculation of the memory usage of each model. This calculation is useful in estimating what hardware you’d need to serve or finetune a model.

Because memory calculation requires a breakdown of low-level ML and computing concepts, this section is technically dense. If you’re already familiar with these concepts, feel free to skip them.

> **KEY TAKEAWAYS FOR UNDERSTANDING MEMORY BOTTLENECKS**
> If you decide to skip this section, here are a few key takeaways. If you find any of these takeaways unfamiliar, the concepts in this section should help explain it:
> 1. Because of the scale of foundation models, memory is a bottleneck for working with them, both for inference and for finetuning. The memory needed for finetuning is typically much higher than the memory needed for inference because of the way neural networks are trained.
> 2. The key contributors to a model’s memory footprint during finetuning are its number of parameters, its number of trainable parameters, and its numerical representations.
> 3. The more trainable parameters, the higher the memory footprint. You can reduce memory requirement for finetuning by reducing the number of trainable parameters. Reducing the number of trainable parameters is the motivation for PEFT, parameter-efficient finetuning.
> 4. Quantization refers to the practice of converting a model from a format with more bits to a format with fewer bits. Quantization is a straightforward and efficient way to reduce a model’s memory footprint. For a model of 13 billion parameters, using FP32 means 4 bytes per weight or 52 GB for the whole weights. If you can reduce each value to only 2 bytes, the memory needed for the model’s weights decreases to 26 GB.
> 5. Inference is typically done using as few bits as possible, such as 16 bits, 8 bits, and even 4 bits.
> 6. Training is more sensitive to numerical precision, so it’s harder to train a model in lower precision. Training is typically done in mixed precision, with some operations done in higher precision (e.g., 32-bit) and some in lower precision (e.g., 16-bit or 8-bit).

### Backpropagation and Trainable Parameters

A key factor that determines a model’s memory footprint during finetuning is its number of trainable parameters. A trainable parameter is a parameter that can be updated during finetuning. During pre-training, all model parameters are updated. During inference, no model parameters are updated. During finetuning, some or all model parameters may be updated. The parameters that are kept unchanged are frozen parameters.

The memory needed for each trainable parameter results from the way a model is trained. As of this writing, neural networks are typically trained using a mechanism called backpropagation. With backpropagation, each training step consists of two phases:


![Figure (page 679)](/assets/en/07-finetuning/p679_01.png)

![Figure (page 678)](/assets/en/07-finetuning/p678_01.png)

![Figure (page 677)](/assets/en/07-finetuning/p677_01.png)

![Figure (page 674)](/assets/en/07-finetuning/p674_01.png)

![Figure (page 673)](/assets/en/07-finetuning/p673_01.png)

![Figure (page 670)](/assets/en/07-finetuning/p670_01.png)

![Figure (page 669)](/assets/en/07-finetuning/p669_01.png)

![Figure (page 668)](/assets/en/07-finetuning/p668_01.png)

![Figure (page 658)](/assets/en/07-finetuning/p658_01.png)

![Figure (page 650)](/assets/en/07-finetuning/p650_01.png)

![Figure (page 648)](/assets/en/07-finetuning/p648_01.png)

![Figure (page 647)](/assets/en/07-finetuning/p647_01.png)

![Figure (page 643)](/assets/en/07-finetuning/p643_01.png)

![Figure (page 642)](/assets/en/07-finetuning/p642_01.png)

![Figure (page 629)](/assets/en/07-finetuning/p629_01.png)

![Figure (page 627)](/assets/en/07-finetuning/p627_01.png)

![Figure (page 622)](/assets/en/07-finetuning/p622_01.png)
1. **Forward pass**: the process of computing the output from the input.
2. **Backward pass**: the process of updating the model’s weights using the aggregated signals from the forward pass.

During inference, only the forward pass is executed. During training, both passes are executed. At a high level, the backward pass works as follows:

1. Compare the computed output from the forward pass against the expected output (ground truth). If they are different, the model made a mistake, and the parameters need to be adjusted. The difference between the computed output and the expected output is called the loss.
2. Compute how much each trainable parameter contributes to the mistake. This value is called the gradient. Mathematically, gradients are computed by taking the derivative of the loss with respect to each trainable parameter. There’s one gradient value per trainable parameter. If a parameter has a high gradient, it significantly contributes to the loss and should be adjusted more.
3. Adjust trainable parameter values using their corresponding gradient. How much each parameter should be readjusted, given its gradient value, is determined by the optimizer. Common optimizers include SGD (stochastic gradient descent) and Adam. For transformer-based models, Adam is, by far, the most widely used optimizer.

The forward and backward pass for a hypothetical neural network with three parameters and one nonlinear activation function is visualized in Figure 7-4. I use this dummy neural network to simplify the visualization.

During the backward pass, each trainable parameter comes with additional values, its gradient, and its optimizer states. Therefore, the more trainable parameters there are, the more memory is needed to store these additional values.

### Memory Math

It’s useful to know how much memory a model needs so that you can use the right hardware for it. Often, you might already have the hardware and need to calculate whether you can afford to run a certain model. If a model requires 30 GB of memory to do inference, a chip with 24 GB of memory won’t be sufficient.

A model’s memory footprint depends on the model as well as the workload and the different optimization techniques used to reduce its memory usage. Because it’s impossible to account for all optimization techniques and workloads, in this section, I’ll outline only the formulas for approximate calculations, which should give you a rough idea of how much memory you need to operate a model, both during inference and training.

> **NOTE**
> Inference and training having distinct memory profiles is one of the reasons for the divergence in chips for training and inference, as discussed in Chapter 9.

#### Memory needed for inference

During inference, only the forward pass is executed. The forward pass requires memory for the model’s weights. Let $N$ be the model’s parameter count and $M$ be the memory needed for each parameter; the memory needed to load the model’s parameters is:

$$N \times M$$

The forward pass also requires memory for activation values. Transformer models need memory for key-value vectors for the attention mechanism. The memory for both activation values and key-value vectors grows linearly with sequence length and batch size.

For many applications, the memory for activation and key-value vectors can be assumed to be 20% of the memory for the model’s weights. If your application uses a longer context or larger batch size, the actual memory needed will be higher. This assumption brings the model’s memory footprint to:

$$N \times M \times 1.2$$

Consider a 13B-parameter model. If each parameter requires 2 bytes, the model’s weights will require 13B × 2 bytes = 26 GB. The total memory for inference will be 26 GB × 1.2 = 31.2 GB.

A model’s memory footprint grows rapidly with its size. As models become bigger, memory becomes a bottleneck for operating them. A 70B-parameter model with 2 bytes per parameter will require a whooping 140 GB of memory just for its weights.

#### Memory needed for training

To train a model, you need memory for the model’s weights and activations, which has already been discussed. Additionally, you need memory for gradients and optimizer states, which scales with the number of trainable parameters.

Overall, the memory needed for training is calculated as:

$$\text{Training memory} = \text{model weights} + \text{activations} + \text{gradients} + \text{optimizer states}$$

> **TIP**
> During the backward pass, each trainable parameter requires one value for gradient plus zero to two values for optimizer states, depending on the optimizer:
> - A vanilla SGD optimizer has no state.
> - A momentum optimizer stores one value per trainable parameter.
> - An Adam optimizer stores two values per trainable parameter.

Imagine you’re updating all parameters in a 13B-parameter model using the Adam optimizer. Because each trainable parameter has three values for its gradient and optimizer states, if it takes two bytes to store each value, the memory needed for gradients and optimizer states will be:

$$13 \text{ billion} \times 3 \times 2 \text{ bytes} = 78 \text{ GB}$$

However, if you only have 1B trainable parameters, the memory needed for gradients and optimizer states will be only:

$$1 \text{ billion} \times 3 \times 2 \text{ bytes} = 6 \text{ GB}$$

One important thing to note is that in the previous formula, I assumed that the memory needed for activations is less than the memory needed for the model’s weights. However, in reality, the activation memory can be much larger. If activations are stored for gradient computation, the memory needed for activations can dwarf the memory needed for the model’s weights. Figure 7-5 shows the memory needed for activations compared to the memory needed for the model’s weights for different Megatron models at different scales, according to the paper “Reducing Activation Recomputation in Large Transformer Models”, by Korthikanti et al. (2022).

One way to reduce the memory needed for activations is not to store them. Instead of storing activations for reuse, you recompute activations when necessary. This technique is called gradient checkpointing or activation recomputation. While this reduces the memory requirements, it increases the time needed for training due to the recomputation.

### Numerical Representations

In the memory calculation so far, I’ve assumed that each value takes up two bytes of memory. The memory required to represent each value in a model contributes directly to the model’s overall memory footprint. If you reduce the memory needed for each value by half, the memory needed for the model’s weights is also reduced by half.

Before discussing how to reduce the memory needed for each value, it’s useful to understand numerical representations. Numerical values in neural networks are traditionally represented as float numbers. The most common family of floating point formats is the FP family, which adheres to the Institute of Electrical and Electronics Engineers (IEEE) standard for Floating-Point Arithmetic (IEEE 754):

- **FP32** uses 32 bits (4 bytes) to represent a float. This format is called single precision.
- **FP64** uses 64 bits (8 bytes) and is called double precision.
- **FP16** uses 16 bits (2 bytes) and is called half precision.

While FP64 is still used in many computations—as of this writing, FP64 is the default format for NumPy and pandas—it’s rarely used in neural networks because of its memory footprint. FP32 and FP16 are more common. Other popular floating point formats in AI workloads include BF16 (BFloat16) and TF32 (TensorFloat-32). BF16 was designed by Google to optimize AI performance on TPUs and TF32 was designed by NVIDIA for GPUs.

Numbers can also be represented as integers. Even though not yet as common as floating formats, integer representations are becoming increasingly popular. Common integer formats are INT8 (8-bit integers) and INT4 (4-bit integers).

Each float format usually has 1 bit to represent the number’s sign, i.e., negative or positive. The rest of the bits are split between range and precision:

- **Range**: The number of range bits determines the range of values the format can represent. More bits means a wider range. This is similar to how having more digits lets you represent a wider range of numbers.
- **Precision**: The number of precision bits determines how precisely a number can be represented. Reducing the number of precision bits makes a number less precise. For example, if you convert 10.1234 to a format that can support only two decimal digits, this value becomes 10.12, which is less precise than the original value.

Figure 7-6 shows different floating point formats along with their range and precision bits.

Formats with more bits are considered higher precision. Converting a number with a high-precision format into a low-precision format (e.g., from FP32 to FP16) means reducing its precision. Reducing precision can cause a value to change or result in errors. Table 7-3 shows how FP32 values can be converted into FP16, BF16, and TF32.

**Table 7-3. Convert from FP32 values to lower-precision formats. Resultant inaccuracies are in italics.**

| FP32 | FP16 | BF16 | TF32 |
|------|------|------|------|
| 0.0123456789 | 0.0123443603515625 | 0.0123291 | 0.01234436035 |
| 0.123456789 | 0.12347412109375 | 0.123535 | 0.12341308593 |
| 1.23456789 | 1.234375 | 1.23438 | 1.234375 |
| 12.3456789 | 12.34375 | 12.375 | 12.34375 |
| 123.456789 | 123.4375 | 123.5 | 123.4375 |
| 1234.56789 | 1235.0 | 1232.0 | 1234.0 |
| 12345.6789 | 12344.0 | 12352.0 | 12344.0 |
| 123456.789 | INF^a^ | 123392.0 | 123456.0 |
| 1234567.89 | INF^a^ | 1236990.0 | 1233920.0 |

^a^ Values out of bound in FP16 are rounded to infinity.

Note in Table 7-3 that even though BF16 and FP16 have the same number of bits, BF16 has more bits for range and fewer bits for precision. This allows BF16 to represent large values that are out-of-bound for FP16. However, this also makes BF16 less precise than FP16. For example, 1234.56789 is 1235.0 in FP16 (0.035% value change) but 1232.0 in BF16 (0.208% value change).

> **WARNING**
> When using a model, make sure to load the model in the format it’s intended for. Loading a model into the wrong numerical format can cause the model to change significantly. For example, Llama 2 had its weights set to BF16 when it came out. However, many teams loaded the model in FP16 and were subsequently frustrated to find the model’s quality much worse than advertised. While this misunderstanding wasted a lot of people’s time, the upside is that it forced many people to learn about numerical representations.

The right format for you depends on the distribution of numerical values of your workload (such as the range of values you need), how sensitive your workload is to small numerical changes, and the underlying hardware.

### Quantization

The fewer bits needed to represent a model’s values, the lower the model’s memory footprint will be. A 10B-parameter model in a 32-bit format requires 40 GB for its weights, but the same model in a 16-bit format will require only 20 GB. Reducing precision, also known as quantization, is a cheap and extremely effective way to reduce a model’s memory footprint. It’s straightforward to do and generalizes over tasks and architectures. In the context of ML, low precision generally refers to any format with fewer bits than the standard FP32.

> **QUANTIZATION VERSUS REDUCED PRECISION**
> Strictly speaking, it’s quantization only if the target format is integer. However, in practice, quantization is used to refer to all techniques that convert values to a lower-precision format. In this book, I use quantization to refer to precision reduction, to keep it consistent with the literature.

To do quantization, you need to decide what to quantize and when:

- **What to quantize**: Ideally, you want to quantize whatever is consuming most of your memory, but it also depends on what you can quantize without hurting performance too much. As discussed in “Memory Math”, major contributors to a model’s memory footprint during inference are the model’s weights and activations. Weight quantization is more common than activation quantization, since weight activation tends to have a more stable impact on performance with less accuracy loss.
- **When to quantize**: Quantization can happen during training or post-training. Post-training quantization (PTQ) means quantizing a model after it’s been fully trained. PTQ is by far the most common. It’s also more relevant to AI application developers who don’t usually train models.

#### Inference quantization

In the early days of deep learning, it was standard to train and serve models using 32 bits with FP32. Since the late 2010s, it has become increasingly common to serve models in 16 bits and in even lower precision. For example, Dettmers et al. (2022) have done excellent work quantizing LLMs into 8 bits with LLM.int8() and 4 bits with QLoRA (Dettmers et al., 2023).

A model can also be served in mixed precision, where values are reduced in precision when possible and maintained in higher precision when necessary. To serve models on the devices, Apple (2024) leveraged a quantization scheme that uses a mixture of 2-bit and 4-bit formats, averaging 3.5 bits-per-weight. Also in 2024, in anticipation of 4-bit neural networks, NVIDIA announced their new GPU architecture, Blackwell, that supports model inference in 4-bit float.

Once you get to 8 bits and under, numerical representations get more tricky. You can keep parameter values as floats using one of the minifloat formats, such as FP8 (8 bits) and FP4 (4 bits). More commonly, however, parameter values are converted into an integer format, such as INT8 or INT4.

Quantization is effective, but there’s a limit to how far it can go. You can’t have fewer than 1 bit per value, and some have attempted the 1-bit representation, e.g., BinaryConnect (Courbariaux et al., 2015), Xnor-Net (Rastegari et al., 2016), and BitNet (Wang et al., 2023).

In 2024, Microsoft researchers (Ma et al.) declared that we’re entering the era of 1-bit LLMs by introducing BitNet b1.58, a transformer-based language model that requires only 1.58 bits per parameter and whose performance is comparable to 16-bit Llama 2 (Touvron et al., 2023) up to 3.9B parameters, as shown in Table 7-4.

**Table 7-4. BitNet b1.58’s performance compared to that of Llama 2 16-bit on different benchmarks**

| Model | Size | ARCe | ARCc | HS |
|-------|------|------|------|-----|
| Llama LLM | 700M | 54.7 | 23.0 | 37.0 |
| BitNet b1.58 | 700M | 51.8 | 21.4 | 35.1 |
| Llama LLM | 1.3B | 56.9 | 23.5 | 38.5 |
| BitNet b1.58 | 1.3B | 54.9 | 24.2 | 37.7 |
| Llama LLM | 3B | 62.1 | 25.6 | 43.3 |
| BitNet b1.58 | 3B | 61.4 | 28.3 | 42.9 |
| BitNet b1.58 | 3.9B | 64.2 | 28.7 | 44.2 |

Reduced precision not only reduces the memory footprint but also often improves computation speed. First, it allows a larger batch size, enabling the model to process more inputs in parallel. Second, reduced precision speeds up computation, which further reduces inference latency and training time. To illustrate this, consider the addition of two numbers. If we perform the addition bit by bit, and each takes $t$ nanoseconds, it’ll take $32t$ nanoseconds for 32 bits but only $16t$ nanoseconds for 16 bits. However, reducing precision doesn’t always reduce latency due to the added computation needed for format conversion.

There are downsides to reduced precision. Each conversion often causes a small value change, and many small changes can cause a big performance change. If a value is outside the range the reduced precision format can represent, it might be converted to infinity or an arbitrary value, causing the model’s quality to further degrade. How to reduce precision with minimal impact on model performance is an active area of research, pursued by model developers as well as by hardware makers and application developers.

Inference in lower precision has become a standard. A model is trained using a higher-precision format to maximize performance, then its precision is reduced for inference. Major ML frameworks, including PyTorch, TensorFlow, and Hugging Face’s transformers, offer PTQ for free with a few lines of code.

Some edge devices only support quantized inference. Therefore, frameworks for on-device inference, such as TensorFlow Lite and PyTorch Mobile, also offer PTQ.

#### Training quantization

Quantization during training is not yet as common as PTQ, but it’s gaining traction. There are two distinct goals for training quantization:

1. To produce a model that can perform well in low precision during inference. This is to address the challenge that a model’s quality might degrade during post-training quantization.
2. To reduce training time and cost. Quantization reduces a model’s memory footprint, allowing a model to be trained on cheaper hardware or allowing the training of a larger model on the same hardware. Quantization also speeds up computation, which further reduces costs.

A quantization technique might help achieve one or both of these goals. Quantization-aware training (QAT) aims to create a model with high quality in low precision for inference. With QAT, the model simulates low-precision (e.g., 8-bit) behavior during training, which allows the model to learn to produce high-quality outputs in low precision. However, QAT doesn’t reduce a model’s training time since its computations are still performed in high precision. QAT can even increase training time due to the extra work of simulating low-precision behavior.

On the other hand, training a model directly in lower precision can help with both goals. People attempted to train models in reduced precision as early as 2016; see Hubara et al. (2016) and Jacob et al. (2017). Character.AI (2024) shared that they were able to train their models entirely in INT8, which helped eliminate the training/serving precision mismatch while also significantly improving training efficiency. However, training in lower precision is harder to do, as backpropagation is more sensitive to lower precision.

Lower-precision training is often done in mixed precision, where a copy of the weights is kept in higher precision but other values, such as gradients and activations, are kept in lower precision. You can also have less-sensitive weight values computed in lower precision and more-sensitive weight values computed in higher precision. For example, a common approach is to keep a master copy of weights in FP32, while using FP16 or BF16 for forward and backward passes, and then updating the FP32 master weights with the lower-precision gradients. This approach, known as mixed-precision training, has become standard practice in deep learning frameworks.
