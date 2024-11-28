# Transformers are Multi-State RNNs

## Overview
The paper re-conceptualizes transformer models, specifically decoder-only transformers, as a form of multi-state recurrent neural networks (MSRNNs). The authors introduce a novel compression policy, Token Omission Via Attention (TOVA), which allows for the conversion of unbounded MSRNNs (transformers) into bounded ones, effectively reducing the key-value cache size while maintaining performance. This work is significant in bridging the conceptual gap between transformers and RNNs and offers practical benefits in terms of memory efficiency and throughput.

## Problems Addressed
- **Memory Bottleneck**: Large language models (LLMs) suffer from high memory usage, particularly due to their key-value (KV) cache, which can limit their scalability and applicability.
- **Computational Efficiency**: The large KV cache also hampers computational efficiency, making it difficult to process long sequences or increase the batch size during inference.

## Challenges
- **Compression Policies**: Existing compression methods, such as windowed attention, have limitations and may not optimally retain important token information.
- **Maintaining Performance**: Reducing the KV cache size must be done without significantly compromising the model's performance on various tasks, including language modeling, long-range understanding, and text generation.

## Key Idea and Technique
- **Conceptualization of Transformers as MSRNNs**: The authors propose that transformers can be viewed as unbounded MSRNNs, where each state corresponds to a history token. By limiting the number of states, they can convert transformers into bounded MSRNNs.
- **Token Omission Via Attention (TOVA)**: A novel, training-free compression policy that selects which tokens to keep based on their attention scores. TOVA retains the states with the highest attention scores, allowing for efficient compression of the KV cache.

## Results
- **Performance Comparison**: Across multiple tasks, including language modeling (PG-19), long-range understanding (SQuALITY, QASPER), and text generation, TOVA outperforms other baseline compression policies.
- **Efficiency Gains**: Using TOVA, the models can achieve comparable performance to the full (uncompressed) model using as little as 1/8 of the original cache size, leading to up to 4.8X higher throughput.
- **Extrapolation**: TOVA enables the models to handle very long contexts, up to 70,000 tokens, with minimal performance degradation.

## Future Work
- **Further Optimization**: Investigate additional optimization techniques to further reduce the KV cache size without compromising performance.
- **Cross-Lingual Analysis**: Explore how different languages, especially those with more flexible word order, might require different multi-state sizes and whether TOVA's effectiveness varies across languages.
- **Broader Applications**: Apply TOVA to other types of transformer-based models, such as encoder-decoder architectures, and evaluate its performance in different domains, such as machine translation and summarization.
- **Theoretical Insights**: Deepen the theoretical understanding of why certain tokens (e.g., possessive endings and proper nouns) are retained longer and how this relates to the model's ability to maintain context over long sequences.

