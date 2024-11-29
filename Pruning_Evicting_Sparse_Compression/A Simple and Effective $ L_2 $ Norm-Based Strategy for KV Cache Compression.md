# A Simple and Effective $L_2$ Norm-Based Strategy for KV Cache Compression

## Problems Solved
- **Memory Footprint**: The primary problem addressed is the high memory footprint of the KV cache, which can hinder the practical deployment of LLMs.
- **Decoding Latency**: Processing long-context inputs often leads to increased decoding latency due to the need for repeated access to the potentially large KV cache from high-bandwidth memory (HBM) to the streaming multiprocessor (SM).

## Challenges
- **Maintaining Accuracy**: Compressing the KV cache without losing predictive accuracy is a significant challenge.
- **Scalability**: Ensuring that the compression strategy works effectively across different model sizes and architectures.
- **Theoretical Understanding**: There is a lack of comprehensive theoretical explanation for why the $L_2$ norm correlates with the importance of KV pairs.

## Key Idea and Technique
- **$L_2$ Norm Correlation**: The key insight is the observed correlation between the $L_2$ norm of key embeddings and their corresponding attention scores. A low $L_2$ norm of a key embedding is typically associated with a high attention score.
- **Compression Strategy**: Based on this observation, the authors propose a simple yet effective method to retain only those keys with the lowest $L_2$ norm, thereby compressing the KV cache.
- **Compatibility**: The technique does not rely on attention scores, making it compatible with FlashAttention, which broadens its applicability.

## Results
- **Reduction in KV Cache Size**: The experimental results show that the proposed method can reduce the KV cache size by 50% on language modeling and needle-in-a-haystack tasks, and up to 90% on passkey retrieval tasks, all while maintaining the predictive accuracy of the model.
- **General Applicability**: The approach is straightforward and can be applied directly to any transformer-based, decoder-only LLM.

## Future Work
- **Larger Models**: The current work has been tested on relatively small models (up to 8 billion parameters). Future research should assess the method's effectiveness on larger-scale models to ensure generalizability.
- **Theoretical Exploration**: Further investigation is needed to understand the underlying reasons behind the importance of the $L_2$ norm in determining the influence of a KV pair.
- **Per-Head Compression Ratios**: The authors intend to explore per-head compression ratios, as the effectiveness of $L_2$ norm-based compression can vary depending on the layer and head considered.
- **Broader Applications**: Investigating the application of this method to other types of neural network architectures beyond transformers, and to other domains where similar caching mechanisms are used.