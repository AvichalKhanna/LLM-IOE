#LLM Inference Optimization Engine

This project is a custom LLM serving system that makes running LLaMA-2-7B faster and cheaper by implementing 3 core optimizations 

1. Containous Batching 
Processes multiple requests simulatenously inserting new ones mid batch
Provides 4x throughput 

2. KV Cache Management 
Reueses attention key/value computation across tokens 
Provides Faster generation

3. IN8 Quantization 
Compresses the weights from 32 bit - 8 bit floats
utilizes 60% less memory 
