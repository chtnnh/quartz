# **AI Engineering Roadmap for Software Engineers**

This comprehensive roadmap is tailored for software engineers with experience in Python, AWS, and Terraform who are
looking to transition into the exciting field of AI engineering. It covers essential concepts, tools, and technologies,
guiding you through a structured learning journey

## **Refresher on Machine Learning and Deep Learning**

Before diving into advanced AI engineering concepts, it's crucial to have a solid foundation in machine learning (ML)
and deep learning (DL). Here are some resources to refresh your knowledge:  

## YouTube Playlists

* **Stanford YouTube playlist — Machine learning:** 1 This playlist offers a comprehensive overview of ML concepts,
algorithms, and applications.  
* **MIT YouTube playlist — Introduction to Deep Learning:** 1 This playlist provides a solid introduction to DL,
covering neural networks, convolutional networks, and recurrent networks.  
* **DeepLearning.AI YouTube playlist — Deep Learning Specialization:** 1 This playlist delves deeper into DL, exploring
advanced topics like convolutional networks, recurrent networks, and generative models

**Blogs:**

* **Deep Learning, by Ian Goodfellow, Yoshua Bengio and Aaron Courville:** 2 This blog provides a comprehensive
introduction to a broad range of topics in deep learning.  
* **An Introduction to PyTorch — A Simple yet Powerful Deep Learning Library:** 2 This blog offers a hands-on approach
to PyTorch, covering the basics and providing case studies

## Research Papers

* **Information Theory of Deep Learning:** 2 This research paper by Naftali Tishby explores the information theory
behind deep learning

## Online Courses

* **Deep Learning Summer School Talks:** 2 These free videos from events hosted by the Canadian Institute for Advances
Research (CIFAR) and the Vector Institute cover both the foundations and applications of deep neural networks.  
* **CloudyML AI for all course:** 1 This paid course offers a comprehensive and practical approach to deep learning,
suitable for beginners and experienced professionals

## **Deep Dive into LLMs**

Large Language Models (LLMs) are revolutionizing how we interact with computers. They can understand and generate
human-like text, translate languages, write different kinds of creative content, and answer your questions in an
informative way. Here's a breakdown of essential aspects:

### **Algorithms:**

* **Byte-Pair Encoding (BPE):** BPE is a tokenization algorithm that effectively handles rare words and
out-of-vocabulary tokens, which is crucial for LLMs dealing with diverse text data. It works by iteratively merging the
most frequent pair of bytes in the training data to create a vocabulary of subword units. This allows LLMs to represent
words and phrases more efficiently and accurately3.  
* **Self-Supervised Learning:** LLMs are typically trained using self-supervised learning, where they learn to predict
the next word in a sequence4.  
* **Supervised Learning:** Also known as instruction tuning, this involves training LLMs to follow instructions and
respond to specific requests4.  
* **Reinforcement Learning:** This technique uses human feedback to fine-tune LLMs and encourage desirable behaviors4

### **Architectures:**

* **Transformer Architecture:** The transformer architecture is the foundation of most modern LLMs. It allows for
parallel processing of data, enabling efficient training and handling of long-range dependencies in text5.  
* **Encoder-Decoder:** This architecture consists of an encoder that transforms input text into a latent representation
and a decoder that generates output text from this representation5.  
* **Causal Decoder:** This architecture uses a unidirectional attention mechanism, where each token can only attend to
previous tokens5.  
* **Prefix Decoder:** This architecture allows for bidirectional attention over prefix tokens and unidirectional
attention on generated tokens5

### **In-context Learning:**

In-context learning is a unique capability of LLMs that allows them to perform new tasks without explicit training by
providing a few examples of the desired behavior within the input prompt. This is akin to how humans can learn new
concepts by observing a few examples3

### **Training Techniques:**

* **Data Collection and Preprocessing:** Gathering and cleaning large amounts of text data is crucial for training
LLMs7.  
* **Model Configuration:** Defining parameters like the number of layers, attention heads, and hyperparameters is
essential for optimal performance7.  
* **Model Training:** Training involves feeding the model text data and adjusting its weights to improve prediction
accuracy7.  
* **Fine-tuning:** Fine-tuning involves adjusting hyperparameters or modifying the model's structure to improve
performance7

### **Optimization:**

* **Model Pruning:** This technique involves removing less important connections or parameters from the model to reduce
its size and improve efficiency without significant loss of accuracy8.  
* **Knowledge Distillation:** This involves transferring knowledge from a larger, more complex model to a smaller, more
efficient one, improving the smaller model's performance while reducing its computational requirements8

### **Tools and Libraries:**

* **Hugging Face Transformers:** This library provides a wide range of pre-trained LLMs and tools for fine-tuning and
deploying them9.  
* **TensorFlow:** This deep learning framework offers tools for building, training, and deploying LLMs9. TensorFlow
provides a comprehensive ecosystem for deep learning, but it may have a steeper learning curve for beginners.  
* **PyTorch:** This machine learning library is widely used for natural language processing and deep learning,
including LLM development9. PyTorch offers both beginner-friendly features and advanced capabilities for researchers
and engineers.  
* **LangChain:** This framework simplifies the development of LLM-powered applications by providing tools for chaining
components, integrating agents, and handling memory9.  
* **LlamaIndex:** This framework offers a simpler approach to building retrieval-augmented generation (RAG)
applications with LLMs9

### **Online Courses:**

* **LLM University by Cohere:** 11 This free online resource offers a comprehensive introduction to LLMs, covering
fundamental concepts, fine-tuning techniques, and real-world applications.  
* **Large Language Model Course by Maxime Labonne:** 12 This course provides a deep dive into LLM fundamentals,
covering topics relevant to both LLM scientists and engineers

## **Exploring VLMs**

Vision Language Models (VLMs) bridge the gap between visual and textual data, enabling AI systems to understand and
interact with the world in a more human-like way. They can analyze images, answer questions about visual scenes, and
even generate images from text descriptions13

### **Algorithms:**

* **Contrastive Learning:** This technique trains VLMs to distinguish between similar and dissimilar image-text
pairs14.  
* **Masking-based VLMs:** These models learn by predicting missing parts of an image or text14.  
* **Generative-based VLMs:** These models can generate new images from text or text from images14.  
* **Pretrained Backbone-based VLMs:** These models leverage existing LLMs and visual encoders to align visual and
textual representations14

### **Architectures:**

* **Vision Transformer (ViT):** ViTs are commonly used as image encoders in VLMs, processing images in patches to
capture complex features15. Dual encoder models are generally more efficient for processing high-resolution images,
while fusion encoder-decoder models offer better performance in tasks that require a deeper understanding of the
relationship between visual and textual information.  
* **Dual Encoder:** This architecture encodes images and text separately and then combines their representations15.  
* **Fusion Encoder-Decoder:** This architecture fuses visual and textual features early in the model and then uses a
decoder to generate output15

### **Training Techniques:**

* **Data Collection and Preprocessing:** Gathering large, diverse datasets of image-text pairs is crucial for training
VLMs16.  
* **Data Pruning:** Removing irrelevant or low-quality data from the training set can significantly improve model
performance and reduce training time. This involves techniques like heuristics, bootstrapping, and ensuring diverse and
balanced data representation14.  
* **Contrastive Learning:** Training involves minimizing contrastive loss to align similar pairs and separate
dissimilar ones16.  
* **Masked Language-Image Modeling:** Training involves predicting missing parts of an image or text16.  
* **Transfer Learning:** Fine-tuning pre-trained VLMs on specific datasets can improve performance on downstream
tasks16

### **Ethical Considerations and Bias Mitigation:**

It's crucial to address potential biases in training data and ensure responsible AI development when working with VLMs.
This involves carefully curating datasets, evaluating models for fairness, and implementing techniques to mitigate bias
and promote inclusivity13

### **Tools and Libraries:**

* **Hugging Face Transformers:** This library provides pre-trained VLMs and tools for fine-tuning and deploying them17.

* **NVIDIA NeMo:** This framework offers tools for customizing and deploying VLMs, including prompt engineering and
model fine-tuning17

## **Understanding AI Agents**

AI agents are autonomous systems that can perceive their environment, make decisions, and take actions to achieve
specific goals. They are becoming increasingly important in various applications, from chatbots to self-driving cars13

### **Algorithms**

* **Reinforcement Learning:** This technique trains AI agents to learn optimal actions through trial and error.
* **Planning and Search Algorithms:** Algorithms like A* search and Monte Carlo Tree Search help agents make strategic decisions by exploring possible action sequences
* **Multi-Agent Systems:** These systems involve multiple AI agents interacting and collaborating to solve complex problems

### **Architectures**

* **Reactive Agents:** These agents respond directly to environmental stimuli without maintaining an internal state.
* **Deliberative Agents:** These agents use symbolic reasoning and planning to make decisions
* **Hybrid Agents:** Combining reactive and deliberative approaches to balance quick responses and strategic planning

### **Learning Techniques**

* **Supervised Learning:** Training agents using labeled datasets to learn specific behaviors.
* **Unsupervised Learning:** Allowing agents to discover patterns and structures in data without explicit labels
* **Transfer Learning:** Applying knowledge learned in one domain to improve performance in another

### **Ethical Considerations**

* **Transparency and Explainability:** Ensuring AI agents can explain their decision-making process.
* **Safety and Robustness:** Developing mechanisms to prevent unintended or harmful actions
* **Bias Mitigation:** Identifying and reducing biases in agent training and decision-making

### **Tools and Libraries**

* **OpenAI Gym:** A toolkit for developing and comparing reinforcement learning algorithms.
* **Ray RLlib:** A scalable reinforcement learning library for building distributed AI agents
* **TensorFlow Agents:** A library for building and training AI agents using TensorFlow

## **Advanced Topics**

### **Quantization:**

Quantization is a technique for compressing model weights to reduce their size and improve inference speed.

* **Linear Quantization:** This involves reducing the precision of model weights from floating-point to lower-bit
representations28.  
* **Weights Packing:** This involves packing multiple low-bit weights into a single higher-bit integer28

#### **Tools and Libraries:**

* **PyTorch:** This library offers tools for quantizing models28.  
* **Quanto:** This library provides quantization tools for various deep learning frameworks28

### **Deployment:**

Deploying AI models involves making them available for use in real-world applications.

* **Model Serving:** This involves hosting models and providing an interface for accessing them26.  
* **Cloud Deployment:** Cloud platforms like AWS offer services for deploying and scaling AI models26.  
* **Edge Deployment:** Deploying models on edge devices can improve latency and reduce reliance on cloud connectivity26

#### **Tools and Libraries:**

* **TensorFlow Serving:** This framework serves TensorFlow models26.  
* **TorchServe:** This framework serves PyTorch models26.  
* **AWS SageMaker:** This service provides tools for deploying and scaling AI models on AWS26

### **Optimization:**

Optimizing AI models involves improving their performance, efficiency, and accuracy.

* **Hyperparameter Tuning:** This involves finding the best values for model parameters29.  
* **Model Architecture Optimization:** This involves designing efficient and effective model architectures29.  
* **Code Optimization:** This involves writing efficient code for training and inference29

#### **Tools and Libraries:**

* **Optuna:** This library automates hyperparameter optimization29.  
* **Ray Tune:** This library provides tools for distributed hyperparameter tuning29

## **AWS for AI Engineering**

AWS offers a wide range of services and tools for AI engineering.

* **Amazon SageMaker:** This service provides a comprehensive suite of tools for building, training, and deploying
machine learning models30.  
* **Amazon Bedrock:** This service provides access to foundation models (FMs) from leading AI companies30.  
* **Amazon Q:** This service provides a generative AI-powered assistant for software development30.  
* **AWS EC2 (Elastic Compute Cloud):** This service provides resizable compute capacity in the cloud, allowing you to
run AI workloads on virtual machines with varying configurations31.  
* **AWS S3 (Simple Storage Service):** This service provides scalable object storage for storing and retrieving large
datasets used in AI model training and deployment31.  
* **AWS VPC (Virtual Private Cloud):** This service allows you to create a logically isolated section of the AWS cloud
where you can launch AWS resources in a virtual network that you define. This provides a secure and customizable
environment for your AI infrastructure31

## **Terraform for AI Infrastructure Automation**

Terraform is an infrastructure-as-code tool that can be used to automate the provisioning and management of AI
infrastructure

* **Infrastructure Provisioning:** Terraform can be used to provision virtual machines, storage, and networking
resources for AI workloads32.  
* **Configuration Management:** Terraform can be used to manage the configuration of AI software and services32.  
* **Multi-Cloud Deployments:** Terraform can be used to deploy AI infrastructure across multiple cloud providers32

### **Terraform Providers:**

* **terraform-provider-aws:** This provider allows you to manage AWS resources using Terraform, making it easier to
provision and manage AI infrastructure on AWS33.  
* **terraform-provider-azurerm:** This provider enables you to manage Microsoft Azure resources with Terraform,
providing a consistent way to automate AI infrastructure on Azure33.  
* **terraform-provider-google:** This provider allows you to manage Google Cloud Platform (GCP) resources using
Terraform, simplifying the automation of AI infrastructure on GCP33

## **Open Source Projects and Communities**

### **Open Source Projects:**

* **Automatic1111/stable-diffusion-webui:** This project provides a web UI for Stable Diffusion, a popular
text-to-image AI model34.  
* **Lobehub/lobe-chat:** This project offers an open-source AI chat framework34.  
* **SWIRL:** This project provides a solution for complex search requirements in enterprise settings35.  
* **GraphRAG:** This project combines retrieval-augmented generation (RAG) techniques with graph databases35.  
* **GPT-SoVits:** This project combines GPT's language capabilities with advanced voice synthesis to generate
high-quality, natural-sounding voiceovers36.  
* **OpenSora:** This open-source platform helps deploy large-scale AI systems by managing the heavy computational loads
involved36

### **Communities and Forums:**

* **r/artificial:** This subreddit is a forum for discussing AI topics37.  
* **IntellijMind Discord Server:** This server is a community for AI engineers and researchers37.  
* **DeepLearning.AI Community:** This community offers a forum, events, and mentorship for AI learners38.  
* **AI Engineering Meetup Group:** 39 This group fosters a collaborative environment for discussing and sharing ideas
in AI, with a focus on bleeding-edge technologies and design patterns.  
* **AI Engineers Meetup Group:** 40 This community focuses on bringing AI innovations into practical software
solutions, with meetups that delve into using pre-trained AI models and strategic fine-tuning

## **Conclusion**

This roadmap provides a comprehensive guide for your AI engineering journey. By starting with a refresher on ML and DL,
you can build a strong foundation for understanding advanced concepts like LLMs, VLMs, and AI agents. Remember to
explore the various algorithms, architectures, and training techniques associated with each of these areas. Familiarize
yourself with the tools and libraries available, such as Hugging Face Transformers, TensorFlow, PyTorch, LangChain, and
LlamaIndex. Leverage cloud platforms like AWS and infrastructure-as-code tools like Terraform to build and manage your
AI infrastructure. Finally, engage with open-source projects and communities to gain practical experience and stay
connected with the latest advancements in the field.  
As you progress, focus on continuous learning and practical application to solidify your skills and stay ahead in this
rapidly evolving field. Remember that AI engineering is a multidisciplinary field that requires a combination of
theoretical knowledge, practical skills, and a passion for innovation. By embracing these elements, you can
successfully navigate your AI engineering journey and contribute to the exciting future of AI

#### **Works cited**

1\. Top 16 Best Resources Online to Learn Machine Learning in 2021 \- Kaggle, accessed on March 4, 2025, [https://www.kaggle.com/general/274909](https://www.kaggle.com/general/274909)
2\. Deep Learning: Top 10 Resources for Beginners \- RE•WORK Blog, accessed on March 4, 2025,
[https://blog.re-work.co/top-10-resources-for-beginners/](https://blog.re-work.cotop-10-resources-for-beginners/)
3\. Large language model \- Wikipedia, accessed on March 4, 2025, [https://en.wikipedia.org/wiki/Large\_language\_model](https://en.wikipedia.org/wiki/Large_language_model)
4\. Large language model training: how three training phases shape LLMs | Snorkel AI, accessed on March 4, 2025, [https://snorkel.ai/blog/large-language-model-training-three-phases-shape-llm-training/](https://snorkel.ai/blog/large-language-model-training-three-phases-shape-llm-training/)
5\. Exploring Architectures and Configurations for Large Language Models (LLMs) \- Labellerr, accessed on March 4, 2025, [https://www.labellerr.com/blog/exploring-architectures-and-configurations-for-large-language-models-llms/](https://www.labellerr.com/blog/exploring-architectures-and-configurations-for-large-language-models-llms/)
6\. An Overview of Large Language Models (LLMs) | ml-articles – Weights & Biases \- Wandb, accessed on March 4, 2025, [https://wandb.ai/mostafaibrahim17/ml-articles/reports/An-Overview-of-Large-Language-Models-LLMs---VmlldzozODA3MzQz](https://wandb.ai/mostafaibrahim17/ml-articles/reports/An-Overview-of-Large-Language-Models-LLMs---VmlldzozODA3MzQz)
7\. LLM Training: How It Works and 4 Key Considerations \- Run:ai, accessed on March 4, 2025, [https://www.run.ai/guides/machine-learning-engineering/llm-training](https://www.run.ai/guides/machine-learning-engineering/llm-training)
8\. Architecture and Components of Large Language Models (LLMs) for Chatbots \- Appy Pie, accessed on March 4, 2025, [https://www.appypie.com/blog/architecture-and-components-of-llms](https://www.appypie.com/blog/architecture-and-components-of-llms)
9\. The Top 5 LLM Frameworks in 2025 \- Skillcrush, accessed on March 4, 2025, [https://skillcrush.com/blog/best-llm-frameworks/](https://skillcrush.com/blog/best-llm-frameworks/)
10\. Top 5 Production-Ready Open Source AI Libraries for Engineering Teams \- Jozu MLOps, accessed on March 4, 2025, [https://jozu.com/blog/top-5-production-ready-open-source-ai-libraries-for-engineering-teams/](https://jozu.com/blog/top-5-production-ready-open-source-ai-libraries-for-engineering-teams/)
11\. 8 Best Free Courses to Learn Large Language Models (LLMs) \- Tecmint, accessed on March 4, 2025,
[https://www.tecmint.com/free-llm-courses/](https://www.tecmint.com/free-llm-courses/)
12\. wikit-ai/awesome-llm-courses: A curated list of awesome online courses about Large Langage Models (LLMs) \- GitHub, accessed on March 4, 2025, [https://github.com/wikit-ai/awesome-llm-courses](https://github.com/wikit-ai/awesome-llm-courses)
13\. Vision Language Models (VLMs) Explained \- DataCamp, accessed on March 4, 2025, [https://www.datacamp.com/blog/vlms-ai-vision-language-models](https://www.datacamp.com/blog/vlms-ai-vision-language-models)
14\. A Deep Dive into VLMs: Vision-Language Models | by Sunidhi Ashtekar | Medium, accessed on March 4, 2025, [https://medium.com/@sunidhi.ashtekar/a-deep-dive-into-vlms-vision-language-models-d3bdf2a3e728](https://medium.com/@sunidhi.ashtekar/a-deep-dive-into-vlms-vision-language-models-d3bdf2a3e728)
15\. What is a Vision-Language Model (VLM)? \- Roboflow Blog, accessed on March 4, 2025, [https://blog.roboflow.com/what-is-a-vision-language-model/](https://blog.roboflow.com/what-is-a-vision-language-model/)
16\. Guide to Vision-Language Models (VLMs) \- Encord, accessed on March 4, 2025, [https://encord.com/blog/vision-language-models-guide/](https://encord.com/blog/vision-language-models-guide/)
17\. Vision Language Model Prompt Engineering Guide for Image and Video Understanding, accessed on March 4, 2025, [https://developer.nvidia.com/blog/vision-language-model-prompt-engineering-guide-for-image-and-video-understanding/](https://developer.nvidia.com/blog/vision-language-model-prompt-engineering-guide-for-image-and-video-understanding/)
18\. What Are AI Agents? \- IBM, accessed on March 4, 2025, [https://www.ibm.com/think/topics/ai-agents](https://www.ibm.com/think/topics/ai-agents)
19\. What algorithms are commonly used in AI agents? \- Zilliz Vector Database, accessed on March 4, 2025, [https://zilliz.com/ai-faq/what-algorithms-are-commonly-used-in-ai-agents](https://zilliz.com/ai-faq/what-algorithms-are-commonly-used-in-ai-agents)
20\. AI Agent Architecture: Breaking Down the Framework of Autonomous Systems \- Kanerika, accessed on March 4, 2025, [https://kanerika.com/blogs/ai-agent-architecture/](https://kanerika.com/blogs/ai-agent-architecture/)
21\. Types of Agent Architectures: A Guide to Reactive, Deliberative, and Hybrid Models in AI, accessed on March 4, 2025, [https://smythos.com/ai-agents/agent-architectures/types-of-agent-architectures/](https://smythos.com/ai-agents/agent-ar>chitectures/types-of-agent-architectures/)
22\. Agent Architectures \- SmythOS, accessed on March 4, 2025, [https://smythos.com/ai-agents/agent-architectures/](https://smythos.com/ai-agents/agent-architectures/)
23\. How Do You Train an AI Agent? Steps for Success \- Whitegator.ai, accessed on March 4, 2025,[https://whitegator.ai/how-do-you-train-an-ai-agent-steps-for-success/](https://whitegator.ai/how-do-you-train-an-ai-agent-steps-for-success/)
24\. AI Agents in Action: Advanced Training Strategies for Real-World Applications \- Medium, accessed on March 4, 2025, [https://medium.com/@jazmia.henry/ai-agents-in-action-advanced-training-strategies-for-real-world-applications-852298eac2db](https://medium.com/@jazmia.henry/ai-agents-in-action-advanced-training-strategies-for-real-world-applications-852298eac2db)
25\. Libraries You MUST Know For Building AI Agents in 2025 \- Medium, accessed on March 4, 2025,
[https://medium.com/@la\_boukouffallah/libraries-you-must-know-for-building-ai-agents-in-2025-ffe5b079fd53](https://medium.com/@la_boukouffallah/libraries-you-must-know-for-building-ai-agents-in-2025-ffe5b079fd53)
26\. Top 15 LLMOps Tools for Building AI Applications in 2025 \- DataCamp, accessed on March 4, 2025, [https://www.datacamp.com/blog/llmops-tools](https://www.datacamp.com/blog/llmops-tools)
27\. 7 Awesome Platforms & Frameworks for Building AI Agents (Open-Source & More), accessed on March 4, 2025, [https://www.helicone.ai/blog/ai-agent-builders](https://www.helicone.ai/blog/ai-agent-builders)
28\. Quantization in Depth \- DeepLearning.AI, accessed on March 4, 2025, [https://www.deeplearning.ai/short-courses/quantization-in-depth/](https://www.deeplearning.ai/short-courses/quantization-in-depth/)
29\. Best Optimization Courses & Certificates [2025\] | Coursera Learn Online, accessed on March 4, 2025, [https://www.coursera.org/courses?query=optimization](https://www.coursera.org/courses?query=optimization)
30\. AI Courses for Machine Learning Engineers \- Learn AI \- AWS, accessed on March 4, 2025, [https://aws.amazon.com/ai/learn/machine-learning-specialist/](<https://aws.amazon.com/ai/learn/machine-learning-specialist/)
31\. FREE AI-Powered Terraform Code Generator – Automate Infrastructure Instantly \- Workik, accessed on March 4, 2025, [https://workik.com/terraform-code-generator](https://workik.com/terraform-code-generator)
32\. 10 Best Terraform Tools To Use In 2025 \- GeeksforGeeks, accessed on March 4, 2025, [https://www.geeksforgeeks.org/best-terraform-tools/](https://www.geeksforgeeks.org/best-terraform-tools/)
33\. shuaibiyy/awesome-tf: Curated list of resources on HashiCorp's Terraform and OpenTofu \- GitHub, accessed on March 4, 2025, [https://github.com/shuaibiyy/awesome-tf](https://github.com/shuaibiyy/awesome-tf)
34\. TOP 34 Ai Open Source Projects in 2025 \- Web3 Jobs, accessed on March 4, 2025, [https://web3.career/learn-web3/top-ai-open-source-projects](https://web3.career/learn-web3/top-ai-open-source-projects)  
35\. 5 Open-Source Projects That Will Transform ⚡️ Your AI Workflow \- DEV Community, accessed on March 4, 2025, [https://dev.to/fast/5-open-source-projects-that-will-transform-your-ai-workflow-190g](https://dev.to/fast/5-open-source-projects-that-will-transform-your-ai-workflow-190g)
36\. Top 10 Trending Open Source AI Repositories Starting Off 2025 | by ODSC, accessed on March 4, 2025, [https://odsc.medium.com/top-10-trending-open-source-ai-repositories-starting-off-2025-830ac2315e78](https://odsc.medium.com/top-10-trending-open-source-ai-repositories-starting-off-2025-830ac2315e78)
37\. Artificial Intelligence (AI) \- Reddit, accessed on March 4, 2025, [https://www.reddit.com/r/artificial/](https://www.reddit.com/r/artificial/)
38\. DeepLearning.AI, accessed on March 4, 2025, [https://community.deeplearning.ai/](https://community.deeplearning.ai/)
39\. Ai Engineering \- Meetup, accessed on March 4, 2025, [https://www.meetup.com/ai-engineering/](https://www.meetup.com/ai-engineering/)
40\. AI Engineers \- Meetup, accessed on March 4, 2025, [https://www.meetup.com/ai-engineers/](https://www.meetup.com/ai-engineers/)
