## Background

文章主要解决的问题是大型语言模型（LLMs）在生成长内容时，例如对话系统和故事写作等应用中，由于需要存储大量的瞬态状态信息（称为KV缓存）到GPU内存中，导致部署成本非常高。这些KV缓存的大小与序列长度和批次大小成线性关系，因此随着输入文本的增长，所需内存也会相应增加。

## Challenges

主要的难点和挑战在于设计一个高效的KV缓存机制，以满足以下三个关键要求：

1. **小缓存大小**：为了减少内存占用，需要限制KV缓存的大小。然而，每个解码步骤原则上可能需要访问所有先前的注意力键值对，这使得直接限制KV缓存大小变得困难。
2. **低缺失率**：即使在减少缓存大小的同时，也需要维持生成文本的质量和长内容生成的能力。找到一个既能保持性能又能有效管理缓存大小的最优淘汰策略是一个组合优化问题，具有很高的计算复杂度。
3. **低成本淘汰策略**：即便能够通过暴力搜索找到最优的淘汰策略，这种策略在实际应用中部署也是不可行的，因为它的计算成本过高。

## Idea

文章的主要技术和解决办法集中在减少大型语言模型（LLMs）在生成过程中KV缓存的内存占用，同时保持生成文本的质量。具体来说，研究者们提出了以下关键技术和方法：

1. **识别（Heavy Hitters, H2）**：研究发现，在计算注意力分数时，只有一小部分标记贡献了大部分价值。这些标记被称为“重击手”。它们的出现与文本中标记的频繁共现密切相关。

2. **H2O算法（Heavy Hitter Oracle）**：基于对重击手的重要性的认识，研究者设计了一种名为H2O的KV缓存淘汰策略。该策略旨在动态地保留最近使用和重击手标记之间的平衡。通过这种方法，可以显著降低KV缓存的大小，而不会严重损害模型性能。

3. **动态子模问题的形式化**：为了有效地管理KV缓存，研究者将KV缓存淘汰问题形式化为一个动态子模问题，并为此提供了一个理论保证的新算法。这种形式化有助于指导未来的研究工作，并为实际应用提供了坚实的理论基础。

4. **实验验证**：研究者通过广泛的评估表明，H2O不仅能够显著提高端到端吞吐量并减少延迟时间，而且还能保持甚至提升生成文本的质量。此外，还观察到H2O能增加生成文本的多样性，即产生的句子中重复词较少且更具创意。

5. **与现有优化技术的结合**：H2O与现有的优化技术如卸载(offloading)和量化(quantization)是正交的，因此可以结合起来以实现更好的性能。研究者在FlexGen等先进的推理引擎上实现了这一策略，并报告了吞吐量和延迟方面的改进。

## Implementation

Implementation of H2O with 20% heavy hitters improves the throughput over three leading inference systems DeepSpeed Zero-Inference, Hugging Face Accelerate, and FlexGen by up to 29*×*, 29*×*, and 3*×* on OPT-6.7B and OPT-30B. With the same batch size, H2O can reduce the latency by up to 1*.*9*×*. 

## Future Work
