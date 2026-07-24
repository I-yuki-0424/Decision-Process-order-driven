# Goals
- Develop an unprecedented task generation model leveraging Transformer architectures and Markov Decision Processes (MDP).
- Pursue high accuracy without relying on scaling up model size.
- Maintain a flexible design to accommodate new policies or directions.
- Use JAX as the primary framework, maximizing execution speed via JIT compilation wherever possible.
- Assume execution on CUDA by default, while maintaining generalizability to enable training on TPUs if possible.

# Important Notes
- As the model architecture is presumed to be entirely novel, do not implement unconfirmed components based on arbitrary assumptions; strictly adhere to provided instructions and design philosophy documentation.