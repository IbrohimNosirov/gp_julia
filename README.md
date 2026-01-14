Julia implementation of a polynomial + kernel GP. I am aiming for maintainable code that is easy to replicate when merging to GPTune.
I'm also looking to efficiently use memory so everything is statically allocated and modified using Julia's views.

1. Use a bordered system solve (block Gaussian elimination) instead of Sherman Woodbury Morrison (https://epubs.siam.org/doi/10.1137/0612034).
2. Implicitly project off asymptotically infinite directions (this GP uses an improper prior). ["Geostatistics Oslo 2012"].

Work in progress: I found a recent paper, published while I was working on this project over the summer.
This paper promises a Cholesky-based linear solve, as opposed to LU, from using the Wendland kernel.
I have to think about how this would practically go.

Paper in question: https://arxiv.org/pdf/2507.12629.
