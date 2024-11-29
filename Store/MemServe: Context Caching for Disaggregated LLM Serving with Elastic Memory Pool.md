# MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool

## Introduction
The paper introduces MemServe, a system designed to improve the efficiency of serving large language models (LLMs) by unifying inter-request and intra-request optimizations. This is in response to the evolution of LLM serving from stateless to stateful systems, which leverage dependencies across and within inference requests.

1. inter-request: Inter-request techniques exploit dependencies across requests. 
2. intra-request: Intra-request techniques exploit dependencies within a single request on different instances.

## Problems Addressed
1. **Lack of Holistic Design for Inter-Request Techniques**: Existing LLM serving systems do not have an overarching design that effectively utilizes context caching, which could maximize key-value (KV) cache reuse.
2. **Inability to Combine Inter-Request and Intra-Request Optimizations**: Current methods cannot simultaneously apply both types of optimizations, including **sequence parallelism**, leading to inefficiencies.
3. **Challenges in Managing Distributed KV Cache**: As LLMs become more complex, managing the KV cache across distributed instances becomes difficult, especially when using techniques like disaggregated inference and sequence parallelism.

## Challenges
- **Complexity in KV Cache Management**: The need for novel logic to manage and transfer the KV cache, which is the intermediate data produced during LLM inference.
- **Scalability and Performance**: Ensuring that the system can scale efficiently while maintaining or improving performance metrics such as job completion time (JCT), time-to-first-token (TTFT), and throughput.

## Key Idea and Technique
- **MemPool**: A core component of MemServe, which is an elastic memory pool designed to manage distributed memory and KV caches. It provides APIs for memory, indexing, and distributed data transfer, enabling the implementation of context caching and disaggregated inference.
- **Global Scheduler**: A scheduler that incorporates a locality-aware policy using global prompt trees, enhancing cache reuse and optimizing the scheduling of requests.
- **Unified System for Optimizations**: MemServe integrates both inter-request (context caching) and intra-request (disaggregated inference, sequence parallelism) optimizations, providing a comprehensive solution for efficient LLM serving.

## Results
- **Performance Improvement**: MemServe significantly improves JCT, TTFT, and throughput, demonstrating its effectiveness in enhancing the efficiency of LLM serving.
- **Scalability**: The system is designed to handle the increased complexity and scale of modern LLMs, showing potential for application in large-scale deployment scenarios.

## Future Work
- **Further Optimization of Network and Memory**: Investigating ways to optimize the network and memory further, particularly in high-load scenarios, where excessive network transfers can introduce overhead.
- **Enhanced Scheduling Algorithms**: Developing more sophisticated scheduling algorithms that can better adapt to varying workloads and further improve the utilization of resources.
- **Integration with Other Techniques**: Exploring the integration of MemServe with other memory optimization techniques, such as quantization and low-level algorithmic optimizations, to achieve even greater efficiency.
- **Real-World Deployment and Testing**: Conducting more extensive real-world testing and deployment to validate the system's performance and robustness under diverse conditions.

