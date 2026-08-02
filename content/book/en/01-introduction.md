---
id: "01-introduction"
title: "1. Introduction to Building AI Applications with Foundation Models"
language: "en"
---

<!-- cleaned -->
# 1. Introduction to Building AI Applications with Foundation Models

If I could use only one word to describe AI post-2020, it'd be scale. The AI models behind applications like ChatGPT, Google's Gemini, and Midjourney are at such a scale that they're consuming a nontrivial portion of the world's electricity, and we're at risk of running out of publicly available internet data to train them.

The scaling up of AI models has two major consequences. First, AI models are becoming more powerful and capable of more tasks, enabling more applications. More people and teams leverage AI to increase productivity, create economic value, and improve quality of life.

Second, training large language models (LLMs) requires data, compute resources, and specialized talent that only a few organizations can afford. This has led to the emergence of model as a service: models developed by these few organizations are made available for others to use as a service. Anyone who wishes to leverage AI to build applications can now use these models to do so without having to invest up front in building a model.

In short, the demand for AI applications has increased while the barrier to entry for building AI applications has decreased. This has turned AI engineering—the process of building applications on top of readily available models—into one of the fastest-growing engineering disciplines.

Building applications on top of machine learning (ML) models isn't new. Long before LLMs became prominent, AI was already powering many applications, including product recommendations, fraud detection, and churn prediction. While many principles of productionizing AI applications remain the same, the new generation of large-scale, readily available models brings about new possibilities and new challenges, which are the focus of this book.

This chapter begins with an overview of foundation models, the key catalyst behind the explosion of AI engineering. I'll then discuss a range of successful AI use cases, each illustrating what AI is good and not yet good at. As AI's capabilities expand daily, predicting its future possibilities becomes increasingly challenging. However, existing application patterns can help uncover opportunities today and offer clues about how AI may continue to be used in the future.

To close out the chapter, I'll provide an overview of the new AI stack, including what has changed with foundation models, what remains the same, and how the role of an AI engineer today differs from that of a traditional ML engineer.

## The Rise of AI Engineering

Foundation models emerged from large language models, which, in turn, originated as just language models. While applications like ChatGPT and GitHub's Copilot may seem to have come out of nowhere, they are the culmination of decades of technology advancements, with the first language models emerging in the 1950s. This section traces the key breakthroughs that enabled the evolution from language models to AI engineering.

### From Language Models to Large Language Models

While language models have been around for a while, they've only been able to grow to the scale they are today with self-supervision. This section gives a quick overview of what language model and self-supervision mean. If you're already familiar with those, feel free to skip this section.

#### Language models

A language model encodes statistical information about one or more languages. Intuitively, this information tells us how likely a word is to appear in a given context. For example, given the context "My favorite color is __", a language model that encodes English should predict "blue" more often than "car".

The statistical nature of languages was discovered centuries ago. In the 1905 story "The Adventure of the Dancing Men", Sherlock Holmes leveraged simple statistical information of English to decode sequences of mysterious stick figures. Since the most common letter in English is E, Holmes deduced that the most common stick figure must stand for E.

Later on, Claude Shannon used more sophisticated statistics to decipher enemies' messages during the Second World War. His work on how to model English was published in his 1951 landmark paper "Prediction and Entropy of Printed English". Many concepts introduced in this paper, including entropy, are still used for language modeling today.

In the early days, a language model involved one language. However, today, a language model can involve multiple languages.

The basic unit of a language model is token. A token can be a character, a word, or a part of a word (like -tion), depending on the model. For example, GPT-4, a model behind ChatGPT, breaks the phrase "I can't wait to build AI applications" into nine tokens, as shown in Figure 1-1. Note that in this example, the word "can't" is broken into two tokens, can and 't. You can see how different OpenAI models tokenize text on the OpenAI website.

[Image: Figure 1-1. An example of how GPT-4 tokenizes a phrase.]

The process of breaking the original text into tokens is called tokenization. For GPT-4, an average token is approximately ¾ the length of a word. So, 100 tokens are approximately 75 words.

The set of all tokens a model can work with is the model's vocabulary. You can use a small number of tokens to construct a large number of distinct words, similar to how you can use a few letters in the alphabet to construct many words. The Mixtral 8x7B model has a vocabulary size of 32,000. GPT-4's vocabulary size is 100,256. The tokenization method and vocabulary size are decided by model developers.

**NOTE**

Why do language models use token as their unit instead of word or character? There are three main reasons:

1. Compared to characters, tokens allow the model to break words into meaningful components. For example, "cooking" can be broken into "cook" and "ing", with both components carrying some meaning of the original word.
2. Because there are fewer unique tokens than unique words, this reduces the model's vocabulary size, making the model more efficient (as discussed in Chapter 2).
3. Tokens also help the model process unknown words. For instance, a made-up word like "chatgpting" could be split into "chatgpt" and "ing", helping the model understand its structure.

Tokens balance having fewer units than words while retaining more meaning than individual characters.

There are two main types of language models: masked language models and autoregressive language models. They differ based on what information they can use to predict a token:

**Masked language model**

A masked language model is trained to predict missing tokens anywhere in a sequence, using the context from both before and after the missing tokens. In essence, a masked language model is trained to be able to fill in the blank. For example, given the context, "My favorite __ is blue", a masked language model should predict that the blank is likely "color". A well-known example of a masked language model is bidirectional encoder representations from transformers, or BERT (Devlin et al., 2018).

As of writing, masked language models are commonly used for non-generative tasks such as sentiment analysis and text classification. They are also useful for tasks requiring an understanding of the overall context, such as code debugging, where a model needs to understand both the preceding and following code to identify errors.

**Autoregressive language model**

An autoregressive language model is trained to predict the next token in a sequence, using only the preceding tokens. It predicts what comes next in "My favorite color is __." An autoregressive model can continually generate one token after another. Today, autoregressive language models are the models of choice for text generation, and for this reason, they are much more popular than masked language models.

Figure 1-2 shows these two types of language models.

[Image: Figure 1-2. Autoregressive language model and masked language model.]

**NOTE**

In this book, unless explicitly stated, language model will refer to an autoregressive model.

The outputs of language models are open-ended. A language model can use its fixed, finite vocabulary to construct infinite possible outputs. A model that can generate open-ended outputs is called generative, hence the term generative AI.

You can think of a language model as a completion machine: given a text (prompt), it tries to complete that text. Here's an example:

```text
Prompt (from user): "To be or not to be"
Completion (from language model): ", that is the question."
```

It's important to note that completions are predictions, based on probabilities, and not guaranteed to be correct. This probabilistic nature of language models makes them both so exciting and frustrating to use. We explore this further in Chapter 2.

As simple as it sounds, completion is incredibly powerful. Many tasks, including translation, summarization, coding, and solving math problems, can be framed as completion tasks. For example, given the prompt: "How are you in French is …", a language model might be able to complete it with: "Comment ça va", effectively translating from one language to another.

As another example, given the prompt:

```text
Question: Is this email likely spam? Here's the email: <email content>
Answer:
```

A language model might be able to complete it with: "Likely spam", which turns this language model into a spam classifier.

While completion is powerful, completion isn't the same as engaging in a conversation. For example, if you ask a completion machine a question, it can complete what you said by adding another question instead of answering the question. "Post-Training" discusses how to make a model respond appropriately to a user's request.

#### Self-supervision

Language modeling is just one of many ML algorithms. There are also models for object detection, topic modeling, recommender systems, weather forecasting, stock price prediction, etc. What's special about language models that made them the center of the scaling approach that caused the ChatGPT moment?

The answer is that language models can be trained using self-supervision, while many other models require supervision. Supervision refers to the process of training ML algorithms using labeled data, which can be expensive and slow to obtain. Self-supervision helps overcome this data labeling bottleneck to create larger datasets for models to learn from, effectively allowing models to scale up. Here's how.

With supervision, you label examples to show the behaviors you want the model to learn, and then train the model on these examples. Once trained, the model can be applied to new data. For example, to train a fraud detection model, you use examples of transactions, each labeled with "fraud" or "not fraud". Once the model learns from these examples, you can use this model to predict whether a transaction is fraudulent.

The success of AI models in the 2010s lay in supervision. The model that started the deep learning revolution, AlexNet (Krizhevsky et al., 2012), was supervised. It was trained to learn how to classify over 1 million images in the dataset ImageNet. It classified each image into one of 1,000 categories such as "car", "balloon", or "monkey".

A drawback of supervision is that data labeling is expensive and time-consuming. If it costs 5 cents for one person to label one image, it'd cost $50,000 to label a million images for ImageNet. If you want two different people to label each image—so that you could cross-check label quality—it'd cost twice as much. Because the world contains vastly more than 1,000 objects, to expand models' capabilities to work with more objects, you'd need to add labels of more categories. To scale up to 1 million categories, the labeling cost alone would increase to $50 million.

Labeling everyday objects is something that most people can do without prior training. Hence, it can be done relatively cheaply. However, not all labeling tasks are that simple. Generating Latin translations for an English-to-Latin model is more expensive. Labeling whether a CT scan shows signs of cancer would be astronomical.

Self-supervision helps overcome the data labeling bottleneck. In self-supervision, instead of requiring explicit labels, the model can infer labels from the input data. Language modeling is self-supervised because each input sequence provides both the labels (tokens to be predicted) and the contexts the model can use to predict these labels. For example, the sentence "I love street food." gives six training samples, as shown in Table 1-1.

**Table 1-1. Training samples from the sentence "I love street food." for language modeling.**

| Input (context) | Output (next token) |
|---|---|
| `<BOS>` | I |
| `<BOS>, I` | love |
| `<BOS>, I, love` | street |
| `<BOS>, I, love, street` | food |
| `<BOS>, I, love, street, food` | . |
| `<BOS>, I, love, street, food, .` | `<EOS>` |

In Table 1-1, `<BOS>` and `<EOS>` mark the beginning and the end of a sequence. These markers are necessary for a language model to work with multiple sequences. Each marker is typically treated as one special token by the model. The end-of-sequence marker is especially important as it helps language models know when to end their responses.

**NOTE**

Self-supervision differs from unsupervision. In self-supervised learning, labels are inferred from the input data. In unsupervised learning, you don't need labels at all.

Self-supervised learning means that language models can learn from text sequences without requiring any labeling. Because text sequences are everywhere—in books, blog posts, articles, and Reddit comments—it's possible to construct a massive amount of training data, allowing language models to scale up to become LLMs.

LLM, however, is hardly a scientific term. How large does a language model have to be to be considered large? What is large today might be considered tiny tomorrow. A model's size is typically measured by its number of parameters. A parameter is a variable within an ML model that is updated through the training process. In general, though this is not always true, the more parameters a model has, the greater its capacity to learn desired behaviors.

When OpenAI's first generative pre-trained transformer (GPT) model came out in June 2018, it had 117 million parameters, and that was considered large. In February 2019, when OpenAI introduced GPT-2 with 1.5 billion parameters, 117 million was downgraded to be considered small. As of the writing of this book, a model with 100 billion parameters is considered large. Perhaps one day, this size will be considered small.

Before we move on to the next section, I want to touch on a question that is usually taken for granted: Why do larger models need more data? Larger models have more capacity to learn, and, therefore, would need more training data to maximize their performance. You can train a large model on a small dataset too, but it'd be a waste of compute. You could have achieved similar or better results on this dataset with smaller models.

### From Large Language Models to Foundation Models

While language models are capable of incredible tasks, they are limited to text. As humans, we perceive the world not just via language but also through vision, hearing, touch, and more. Being able to process data beyond text is essential for AI to operate in the real world.

For this reason, language models are being extended to incorporate more data modalities. GPT-4V and Claude 3 can understand images and texts. Some models even understand videos, 3D assets, protein structures, and so on. Incorporating more data modalities into language models makes them even more powerful. OpenAI noted in their GPT-4V system card in 2023 that "incorporating additional modalities (such as image inputs) into LLMs is viewed by some as a key frontier in AI research and development."

While many people still call Gemini and GPT-4V LLMs, they're better characterized as foundation models. The word foundation signifies both the importance of these models in AI applications and the fact that they can be built upon for different needs.

Foundation models mark a breakthrough from the traditional structure of AI research. For a long time, AI research was divided by data modalities. Natural language processing (NLP) deals only with text. Computer vision deals only with vision. Text-only models can be used for tasks such as translation and spam detection. Image-only models can be used for object detection and image classification. Audio-only models can handle speech recognition (speech-to-text, or STT) and speech synthesis (text-to-speech, or TTS).

A model that can work with more than one data modality is also called a multimodal model. A generative multimodal model is also called a large multimodal model (LMM). If a language model generates the next token conditioned on text-only tokens, a multimodal model generates the next token conditioned on both text and image tokens, or whichever modalities that the model supports, as shown in Figure 1-3.

[Image: Figure 1-3. A multimodal model can generate the next token using information from both text and visual tokens.]

Just like language models, multimodal models need data to scale up. Self-supervision works for multimodal models too. For example, OpenAI used a variant of self-supervision called natural language supervision to train their language-image model CLIP (OpenAI, 2021). Instead of manually generating labels for each image, they found (image, text) pairs that co-occurred on the internet. They were able to generate a dataset of 400 million (image, text) pairs, which was 400 times larger than ImageNet, without manual labeling cost. This dataset enabled CLIP to become the first model that could generalize to multiple image classification tasks without requiring additional training.

**NOTE**

This book uses the term foundation models to refer to both large language models and large multimodal models.

Note that CLIP isn't a generative model—it wasn't trained to generate open-ended outputs. CLIP is an embedding model, trained to produce joint embeddings of both texts and images. "Introduction to Embedding" discusses embeddings in detail. For now, you can think of embeddings as vectors that aim to capture the meanings of the original data. Multimodal embedding models like CLIP are the backbones of generative multimodal models, such as Flamingo, LLaVA, and Gemini (previously Bard).

Foundation models also mark the transition from task-specific models to general-purpose models. Previously, models were often developed for specific tasks, such as sentiment analysis or translation. A model trained for sentiment analysis wouldn't be able to do translation, and vice versa. Foundation models, thanks to their scale and the way they are trained, are capable of a wide range of tasks. Out of the box, general-purpose models can work relatively well for many tasks. An LLM can do both sentiment analysis and translation. However, you can often tweak a general-purpose model to maximize its performance on a specific task.

Figure 1-4 shows the tasks used by the Super-NaturalInstructions benchmark to evaluate foundation models (Wang et al., 2022), providing an idea of the types of tasks a foundation model can perform.

Imagine you're working with a retailer to build an application to generate product descriptions for their website. An out-of-the-box model might be able to generate accurate descriptions but might fail to capture the brand's voice or highlight the brand's messaging. The generated descriptions might even be full of marketing speech and cliches.

[Image: Figure 1-4. The range of tasks in the Super-NaturalInstructions benchmark (Wang et al., 2022).]

There are multiple techniques you can use to get the model to generate what you want. For example, you can craft detailed instructions with examples of the desirable product descriptions. This approach is prompt engineering. You can connect the model to a database of customer reviews that the model can leverage to generate better descriptions. Using a database to supplement the instructions is called retrieval-augmented generation (RAG). You can also finetune—further train—the model on a dataset of high-quality product descriptions.

Prompt engineering, RAG, and finetuning are three very common AI engineering techniques that you can use to adapt a model to your needs. The rest of the book will discuss all of them in detail.

Adapting an existing powerful model to your task is generally a lot easier than building a model for your task from scratch—for example, ten examples and one weekend versus 1 million examples and six months. Foundation models make it cheaper to develop AI applications and reduce time to market. Exactly how much data is needed to adapt a model depends on what technique you use. This book will also touch on this question when discussing each technique. However, there are still many benefits to task-specific models, for example, they might be a lot smaller, making them faster and cheaper to use.

Whether to build your own model or leverage an existing one is a classic buy-or-build question that teams will have to answer for themselves. Discussions throughout the book can help with that decision.

### From Foundation Models to AI Engineering

AI engineering refers to the process of building applications on top of foundation models. People have been building AI applications for over a decade—a process often known as ML engineering or MLOps (short for ML operations). Why do we talk about AI engineering now?

If traditional ML engineering involves developing ML models, AI engineering leverages existing ones. The availability and accessibility of powerful foundation models lead to three factors that, together, create ideal conditions for the rapid growth of AI engineering as a discipline:

**Factor 1: General-purpose AI capabilities**

Foundation models are powerful not just because they can do existing tasks better. They are also powerful because they can do more tasks. Applications previously thought impossible are now possible, and applications not thought of before are emerging. Even applications not thought possible today might be possible tomorrow. This makes AI more useful for more aspects of life, vastly increasing both the user base and the demand for AI applications.

For example, since AI can now write as well as humans, sometimes even better, AI can automate or partially automate every task that requires communication, which is pretty much everything. AI is used to write emails, respond to customer requests, and explain complex contracts. Anyone with a computer has access to tools that can instantly generate customized, high-quality images and videos to help create marketing materials, edit professional headshots, visualize art concepts, illustrate books, and so on. AI can even be used to synthesize training data, develop algorithms, and write code, all of which will help train even more powerful models in the future.

**Factor 2: Increased AI investments**

The success of ChatGPT prompted a sharp increase in investments in AI, both from venture capitalists and enterprises. As AI applications become cheaper to build and faster to go to market, returns on investment for AI become more attractive. Companies rush to incorporate AI into their products and processes. Matt Ross, a senior manager of applied research at Scribd, told me that the estimated AI cost for his use cases has gone down two orders of magnitude from April 2022 to April 2023.

Goldman Sachs Research estimated that AI investment could approach $100 billion in the US and $200 billion globally by 2025. AI is often mentioned as a competitive advantage. FactSet found that one in three S&P 500 companies mentioned AI in their earnings calls for the second quarter of 2023, three times more than did so the year earlier. Figure 1-5 shows the number of S&P 500 companies that mentioned AI in their earning calls from 2018 to 2023.

[Image: Figure 1-5. The number of S&P 500 companies that mention AI in their earnings calls reached a record high in 2023. Data from FactSet.]

According to WallStreetZen, companies that mentioned AI in their earning calls saw their stock price increase more than those that didn't: an average of a 4.6% increase compared to 2.4%. It's unclear whether it's causation (AI makes these companies more successful) or correlation (companies are successful because they are quick to adapt to new technologies).

**Factor 3: Low entrance barrier to building AI applications**

The model as a service approach popularized by OpenAI and other model providers makes it easier to leverage AI to build applications. In this approach, models are exposed via APIs that receive user queries and return model outputs. Without these APIs, using an AI model requires the infrastructure to host and serve this model. These APIs give you access to powerful models via single API calls.

Not only that, AI also makes it possible to build applications with minimal coding. First, AI can write code for you, allowing people without a software engineering background to quickly turn their ideas into code and put them in front of their users. Second, you can work with these models in plain English instead of having to use a programming language. Anyone, and I mean anyone, can now develop AI applications.

Because of the resources it takes to develop foundation models, this process is possible only for big corporations (Google, Meta, Microsoft, Baidu, Tencent), governments (Japan, the UAE), and ambitious, well-funded startups (OpenAI, Anthropic, Mistral). In a September 2022 interview, Sam Altman, CEO of OpenAI, said that the biggest opportunity for the vast majority of people will be to adapt these models for specific applications.

The world is quick to embrace this opportunity. AI engineering has rapidly emerged as one of the fastest, and quite possibly the fastest-growing, engineering discipline. Tools for AI engineering are gaining traction faster than any previous software engineering tools. Within just two years, four open source AI engineering tools (AutoGPT, Stable Diffusion eb UI, LangChain, Ollama) have already garnered more stars on GitHub than Bitcoin. They are on track to surpass even the most popular web development frameworks, including React and Vue, in star count. Figure 1-6 shows the GitHub star growth of AI engineering tools compared to Bitcoin, Vue, and React.

A LinkedIn survey from August 2023 shows that the number of professionals adding terms like "Generative AI," "ChatGPT," "Prompt Engineering," and "Prompt Crafting" to their profile increased on average 75% each month. ComputerWorld declared that "teaching AI to behave is the fastest-growing career skill".

[Image: Figure 1-6. Open source AI engineering tools are growing faster than any other software engineering tools, according to their GitHub star counts.]

#### WHY THE TERM "AI ENGINEERING?"

Many terms are being used to describe the process of building applications on top of foundation models, including ML engineering, MLOps, AIOps, LLMOps, etc. Why did I choose to go with AI engineering for this book?

I didn't go with the term ML engineering because, as discussed in "AI Engineering Versus ML Engineering", working with foundation models differs from working with traditional ML models in several important aspects. The term ML engineering won't be sufficient to capture this differentiation. However, ML engineering is a great term to encompass both processes.

I didn't go with all the terms that end with "Ops" because, while there are operational components of the process, the focus is more on tweaking (engineering) foundation models to do what you want.

Finally, I surveyed 20 people who were developing applications on top of foundation models about what term they would use to describe what they were doing. Most people preferred AI engineering. I decided to go with the people.

The rapidly expanding community of AI engineers has demonstrated remarkable creativity with an incredible range of exciting applications. The next section will explore some of the most common application patterns.

## Foundation Model Use Cases

If you're not already building AI applications, I hope the previous section has convinced you that now is a great time to do so. If you have an application in mind, you might want to jump to "Planning AI Applications". If you're looking for inspiration, this section covers a wide range of industry-proven and promising use cases.

The number of potential applications that you could build with foundation models seems endless. Whatever use case you think of, there's probably an AI for that. It's impossible to list all potential use cases for AI.

Even attempting to categorize these use cases is challenging, as different surveys use different categorizations. For example, Amazon Web Services (AWS) has categorized enterprise generative AI use cases into three buckets: customer experience, employee productivity, and process optimization. A 2024 O'Reilly survey categorized the use cases into eight categories: programming, data analysis, customer support, marketing copy, other copy, research, web design, and art.

Some organizations, like Deloitte, have categorized use cases by value capture, such as cost reduction, process efficiency, growth, and accelerating innovation. For value capture, Gartner has a category for business continuity, meaning an organization might go out of business if it doesn't adopt generative AI. Of the 2,500 executives Gartner surveyed in 2023, 7% cited business continuity as the motivation for embracing generative AI.

Eloundou et al. (2023) has excellent research on how exposed different occupations are to AI. They defined a task as exposed if AI and AI-powered software can reduce the time needed to complete this task by at least 50%. An occupation with 80% exposure means that 80% of the occupation's tasks are exposed. According to the study, occupations with 100% or close to 100% exposure include interpreters and translators, tax preparers, web designers, and writers. Some of them are shown in Table 1-2. Not unsurprisingly, occupations with no exposure to AI include cooks, stonemasons, and athletes. This study gives a good idea of what use cases AI is good for.

**Table 1-2. Occupations with the highest exposure to AI as annotated by humans. α refers to exposure to AI models directly, whereas β and ζ refer to exposures to AI-powered software. Table from Eloundou et al. (2023).**

| Group | Occupations with highest exposure | % Exposure |
|---|---|---|
| Human α | Interpreters and translators | 76.5 |
| | Survey researchers | 75.0 |
| | Poets, lyricists, and creative writers | 68.8 |
| | Animal scientists | 66.7 |
| | Public relations specialists | 66.7 |
| Human β | Survey researchers | 84.4 |
| | Writers and authors | 82.5 |
| | Interpreters and translators | 82.4 |
| | Public relations specialists | 80.6 |
| | Animal scientists | 77.8 |
| Human ζ | Mathematicians | 100.0 |
| | Tax preparers | 100.0 |
| | Financial quantitative analysts | 100.0 |
| | Writers and authors | 100.0 |
| | Web and digital interface designers | 100.0 |

Humans labeled 15 occupations as "fully exposed".

When analyzing the use cases, I looked at both enterprise and consumer applications. To understand enterprise use cases, I interviewed 50 companies on their AI strategies and read over 100 case studies. To understand consumer applications, I examined 205 open source AI applications with at least 500 stars on GitHub. I categorized applications into eight groups, as shown in Table 1-3. The limited list here serves best as a reference. As you learn more about how to build foundation models in Chapter 2 and how to evaluate them in Chapter 3, you'll also be able to form a better picture of what use cases foundation models can and should be used for.

**Table 1-3. Common generative AI use cases across consumer and enterprise applications.**

| Category | Examples of consumer use cases | Examples of enterprise use cases |
|---|---|---|
| Coding | Coding | Coding |
| Image and video production | Photo and video editing | Design, Presentation, Ad generation |
| Writing | Email, Social media and blog posts | Copywriting, search engine optimization (SEO), Reports, memos, design docs |
| Education | Tutoring | Essay grading, Employee onboarding, Employee upskill training |
| Conversational bots | General chatbot, AI companion | Customer support, Product copilots |
| Information aggregation | Summarization, Talk-to-your-docs | Summarization, Market research |
| Data organization | Image search, Memex | Knowledge management, Document processing |
| Workflow automation | Travel planning, Event planning | Data extraction, entry, and annotation, Lead generation |

Because foundation models are general, applications built on top of them can solve many problems. This means that an application can belong to more than one category. For example, a bot can provide companionship and aggregate information. An application can help you extract structured data from a PDF and answer questions about that PDF.

Figure 1-7 shows the distribution of these use cases among the 205 open source applications. Note that the small percentage of education, data organization, and writing use cases doesn't mean that these use cases aren't popular. It just means that these applications aren't open source. Builders of these applications might find them more suitable for enterprise use cases.

[Image: Figure 1-7. Distribution of use cases in the 205 open source repositories on GitHub.]

The enterprise world generally prefers applications with lower risks. For example, a 2024 a16z Growth report showed that companies are faster to deploy internal-facing applications (internal knowledge management) than external-facing applications (customer support chatbots), as shown in Figure 1-8. Internal applications help companies develop their AI engineering expertise while minimizing the risks associated with data privacy, compliance, and potential catastrophic failures. Similarly, while foundation models are open-ended and can be used for any task, many applications built on top of them are still close-ended, such as classification. Classification tasks are easier to evaluate, which makes their risks easier to estimate.

[Image: Figure 1-8. Companies are more willing to deploy internal-facing applications]

Even after seeing hundreds of AI applications, I still find new applications that surprise me every week. In the early days of the internet, few people foresaw that the dominating use case on the internet one day would be social media. As we learn to make the most out of AI, the use case that will eventually dominate might surprise us. With luck, the surprise will be a good one.

### Coding

In multiple generative AI surveys, coding is hands down the most popular use case. AI coding tools are popular both because AI is good at coding and because early AI engineers are coders who are more exposed to coding challenges.

One of the earliest successes of foundation models in production is the code completion tool GitHub Copilot, whose annual recurring revenue crossed $100 million only two years after its launch. As of this writing, AI-powered coding startups have raised hundreds of millions of dollars, with Magic raising $320 million and Anysphere raising $60 million, both in August 2024. Open source coding tools like gpt-engineer and screenshot-to-code both got 50,000 stars on GitHub within a year, and many more are being rapidly introduced.

Other than tools that help with general coding, many tools specialize in certain coding tasks. Here are examples of these tasks:

- Extracting structured data from web pages and PDFs (AgentGPT)
- Converting English to code (DB-GPT, SQL Chat, PandasAI)
- Given a design or a screenshot, generating code that will render into a website that looks like the given image (screenshot-to-code, draw-a-ui)
- Translating from one programming language or framework to another (GPT-Migrate, AI Code Translator)
- Writing documentation (Autodoc)
- Creating tests (PentestGPT)
- Generating commit messages (AI Commits)

It's clear that AI can do many software engineering tasks. The question is whether AI can automate software engineering altogether. At one end of the spectrum, Jensen Huang, CEO of NVIDIA, predicts that AI will replace human software engineers and that we should stop saying kids should learn to code. In a leaked recording, AWS CEO Matt Garman shared that in the near future, most developers will stop coding. He doesn't mean it as the end of software developers; it's just that their jobs will change.

At the other end are many software engineers who are convinced that they will never be replaced by AI, both for technical and emotional reasons (people don't like admitting that they can be replaced).

Software engineering consists of many tasks. AI is better at some than others. McKinsey researchers found that AI can help developers be twice as productive for documentation, and 25–50% more productive for code generation and code refactoring. Minimal productivity improvement was observed for highly complex tasks, as shown in Figure 1-9. In my conversations with developers of AI coding tools, many told me that they've noticed that AI is much better at frontend development than backend development.

[Image: Figure 1-9. AI can help developers be significantly more productive, especially for simple tasks, but this applies less for highly complex tasks. Data by McKinsey.]

Regardless of whether AI will replace software engineers, AI can certainly make them more productive. This means that companies can now accomplish more with fewer engineers. AI can also disrupt the outsourcing industry, as outsourced tasks tend to be simpler ones outside of a company's core business.

### Image and Video Production

Thanks to its probabilistic nature, AI is great for creative tasks. Some of the most successful AI startups are creative applications, such as Midjourney for image generation, Adobe Firefly for photo editing, and Runway, Pika Labs, and Sora for video generation. In late 2023, at one and a half years old, Midjourney had already generated $200 million in annual recurring revenue. As of December 2023, among the top 10 free apps for Graphics & Design on the Apple App Store, half have AI in their names. I suspect that soon, graphics and design apps will incorporate AI by default, and they'll no longer need the word "AI" in their names. Chapter 2 discusses the probabilistic nature of AI in more detail.

It's now common to use AI to generate profile pictures for social media, from LinkedIn to TikTok. Many candidates believe that AI-generated headshots can help them put their best foot forward and increase their chances of landing a job. The perception of AI-generated profile pictures has changed significantly. In 2019, Facebook banned accounts using AI-generated profile photos for safety reasons. In 2023, many social media apps provide tools that let users use AI to generate profile photos.

For enterprises, ads and marketing have been quick to incorporate AI. AI can be used to generate promotional images and videos directly. It can help brainstorm ideas or generate first drafts for human experts to iterate upon. You can use AI to generate multiple ads and test to see which one works the best for the audience. AI can generate variations of your ads according to seasons and locations. For example, you can use AI to change leaf colors during fall or add snow to the ground during winter.

### Writing

AI has long been used to aid writing. If you use a smartphone, you're probably familiar with autocorrect and auto-completion, both powered by AI. Writing is an ideal application for AI because we do it a lot, it can be quite tedious, and we have a high tolerance for mistakes. If a model suggests something that you don't like, you can just ignore it.

It's not a surprise that LLMs are good at writing, given that they are trained for text completion. To study the impact of ChatGPT on writing, an MIT study (Noy and Zhang, 2023) assigned occupation-specific writing tasks to 453 college-educated professionals and randomly exposed half of them to ChatGPT. Their results show that among those exposed to ChatGPT, the average time taken decreased by 40% and output quality rose by 18%. ChatGPT helps close the gap in output quality between workers, which means that it's more helpful to those with less inclination for writing. Workers exposed to ChatGPT during the experiment were 2 times as likely to report using it in their real job two weeks after the experiment and 1.6 times as likely two months after that.

For consumers, the use cases are obvious. Many use AI to help them communicate better. You can be angry in an email and ask AI to make it pleasant. You can give it bullet points and get back complete paragraphs. Several people claimed they no longer send an important email without asking AI to improve it first.

Students are using AI to write essays. Writers are using AI to write books. Many startups already use AI to generate children's, fan fiction, romance, and fantasy books. Unlike traditional books, AI-generated books can be interactive, as a book's plot can change depending on a reader's preference. This means that readers can actively participate in creating the story they are reading. A children's reading app identifies the words that a child has trouble with and generates stories centered around these words.

Note-taking and email apps like Google Docs, Notion, and Gmail all use AI to help users improve their writing. Grammarly, a writing assistant app, finetunes a model to make users' writing more fluent, coherent, and clear.

AI's ability to write can also be abused. In 2023, the New York Times reported that Amazon was flooded with shoddy AI-generated travel guidebooks, each outfitted with an author bio, a website, and rave reviews, all AI-generated.

For enterprises, AI writing is common in sales, marketing, and general team communication. Many managers told me they've been using AI to help them write performance reports. AI can help craft effective cold outreach emails, ad copywriting, and product descriptions. Customer relationship management (CRM) apps like HubSpot and Salesforce also have tools for enterprise users to generate web content and outreach emails.

AI seems particularly good with SEO, perhaps because many AI models are trained with data from the internet, which is populated with SEO-optimized text. AI is so good at SEO that it has enabled a new generation of content farms. These farms set up junk websites and fill them with AI-generated content to get them to rank high on Google to drive traffic to them. Then they sell advertising spots through ad exchanges. In June 2023, NewsGuard identified almost 400 ads from 141 popular brands on junk AI-generated websites. One of those junk websites produced 1,200 articles a day. Unless something is done to curtail this, the future of internet content will be AI-generated, and it'll be pretty bleak.

### Education

Whenever ChatGPT is down, OpenAI's Discord server is flooded with students complaining about being unable to complete their homework. Several education boards, including the New York City Public Schools and the Los Angeles Unified School District, were quick to ban ChatGPT for fear of students using it for cheating, but reversed their decisions just a few months later.

Instead of banning AI, schools could incorporate it to help students learn faster. AI can summarize textbooks and generate personalized lecture plans for each student. I find it strange that ads are personalized because we know everyone is different, but education is not. AI can help adapt the materials to the format best suited for each student. Auditory learners can ask AI to read the materials out loud. Students who love animals can use AI to adapt visualizations to feature more animals. Those who find it easier to read code than math equations can ask AI to translate math equations into code.

AI is especially helpful for language learning, as you can ask AI to roleplay different practice scenarios. Pajak and Bicknell (Duolingo, 2022) found that out of four stages of course creation, lesson personalization is the stage that can benefit the most from AI, as shown in Figure 1-10.

[Image: Figure 1-10. AI can be used throughout all four stages of course creation at Duolingo, but it's the most helpful in the personalization stage. Image from Pajak and Bicknell (Duolingo, 2022).]

AI can generate quizzes, both multiple-choice and open-ended, and evaluate the answers. AI can become a debate partner as it's much better at presenting different views on the same topic than the average human. For example, Khan Academy offers AI-powered teaching assistants to students and course assistants to teachers. An innovative teaching method I've seen is that teachers assign AI-generated essays for students to find and correct mistakes.

While many education companies embrace AI to build better products, many find their lunches taken by AI. For example, Chegg, a company that helps students with their homework, saw its share price plummet from $28 when ChatGPT launched in November 2022 to $2 in September 2024, as students have been turning to AI for help.

If the risk is that AI can replace many skills, the opportunity is that AI can be used as a tutor to learn any skill. For many skills, AI can help someone get up to speed quickly and then continue learning on their own to become better than AI.

### Conversational Bots

Conversational bots are versatile. They can help us find information, explain concepts, and brainstorm ideas. AI can be your companion and therapist. It can emulate personalities, letting you talk to a digital copy of anyone you like. Digital girlfriends and boyfriends have become weirdly popular in an incredibly short amount of time. Many are already spending more time talking to bots than to humans (see the discussions here and here). Some are worried that AI will ruin dating.

In research, people have also found that they can use a group of conversational bots to simulate a society, enabling them to conduct studies on social dynamics (Park et al., 2023).

For enterprises, the most popular bots are customer support bots. They can help companies save costs while improving customer experience because they can respond to users sooner than human agents. AI can also be product copilots that guide customers through painful and confusing tasks such as filing insurance claims, doing taxes, or looking up corporate policies.

The success of ChatGPT prompted a wave of text-based conversational bots. However, text isn't the only interface for conversational agents. Voice assistants such as Google Assistant, Siri, and Alexa have been around for years. 3D conversational bots are already common in games and gaining traction in retail and marketing.

One use case of AI-powered 3D characters is smart NPCs, non-player characters (see NVIDIA's demos of Inworld and Convai). NPCs are essential for advancing the storyline of many games. Without AI, NPCs are typically scripted to do simple actions with a limited range of dialogues. AI can make these NPCs much smarter. Intelligent bots can change the dynamics of existing games like The Sims and Skyrim as well as enable new games never possible before.

### Information Aggregation

Many people believe that our success depends on our ability to filter and digest useful information. However, keeping up with emails, Slack messages, and news can sometimes be overwhelming. Luckily, AI came to the rescue. AI has proven to be capable of aggregating information and summarizing it. According to Salesforce's 2023 Generative AI Snapshot Research, 74% of generative AI users use it to distill complex ideas and summarize information.

For consumers, many applications can process your documents—contracts, disclosures, papers—and let you retrieve information in a conversational manner. This use case is also called talk-to-your-docs. AI can help you summarize websites, research, and create reports on the topics of your choice. During the process of writing this book, I found AI helpful for summarizing and comparing papers.

Information aggregation and distillation are essential for enterprise operations. More efficient information aggregation and dissimilation can help an organization become leaner, as it reduces the burden on middle management. When Instacart launched an internal prompt marketplace, it discovered that one of the most popular prompt templates is "Fast Breakdown". This template asks AI to summarize meeting notes, emails, and Slack conversations with facts, open questions, and action items. These action items can then be automatically inserted into a project tracking tool and assigned to the right owners.

AI can help you surface the critical information about your potential customers and run analyses on your competitors.

The more information you gather, the more important it is to organize it. Information aggregation goes hand in hand with data organization.

### Data Organization

One thing certain about the future is that we'll continue producing more and more data. Smartphone users will continue taking photos and videos. Companies will continue to log everything about their products, employees, and customers. Billions of contracts are being created each year. Photos, videos, logs, and PDFs are all unstructured or semistructured data. It's essential to organize all this data in a way that can be searched later.

AI can help with exactly that. AI can automatically generate text descriptions about images and videos, or help match text queries with visuals that match those queries. Services like Google Photos are already using AI to surface images that match search queries. Google Image Search goes a step further: if there's no existing image matching users' needs, it can generate some.

AI is very good with data analysis. It can write programs to generate data visualization, identify outliers, and make predictions like revenue forecasts.

Enterprises can use