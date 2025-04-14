# RAG OpenAI Conversation

The current de-facto method of using LLMs to automate a smart home involves sending *the entire smart home state* as part of the context. This is insanely slow for local LLM's (especially if you are running without GPUs, as prefill times tend to be llama.cpp's bottleneck), and can get expensive over time for cloud LLM API's. However, in practice, most of this state is not even relevant to what you just asked your assistant!

This repository implements RAG (Retrieval Augmented Generation) and fuzzy search to optimize the state that is sent in the first place, which massively reduces the amount of information we need to feed the LLM's. It also feeds the LLM few-shot examples that are dynamically generated based off of the smart home state, which dramatically helps answer quality.

You may start using this by copying the repository URL as a custom repository in HACS. You will need an OpenAI-compatible API server for both a LLM and an embedding model (the latter is only required if you want RAG and fuzzy search).