# ts2vec_long.pt — Model Parameters

## Architecture: TSEncoder

| Hyperparameter | Value |
|----------------|-------|
| input_dim      | 1     |
| hidden_dim     | 128   |
| repr_dim       | 320   |
| depth          | 10    |
| kernel_size    | 3     |

**Total parameters: 1,027,136**

## Layer Breakdown

| Layer | Shape | Params |
|-------|-------|--------|
| input_projection.weight | [128, 1] | 128 |
| input_projection.bias | [128] | 128 |
| blocks.0.conv1.weight | [128, 128, 3] | 49,152 |
| blocks.0.conv1.bias | [128] | 128 |
| blocks.0.conv2.weight | [128, 128, 3] | 49,152 |
| blocks.0.conv2.bias | [128] | 128 |
| blocks.1.conv1.weight | [128, 128, 3] | 49,152 |
| blocks.1.conv1.bias | [128] | 128 |
| blocks.1.conv2.weight | [128, 128, 3] | 49,152 |
| blocks.1.conv2.bias | [128] | 128 |
| blocks.2.conv1.weight | [128, 128, 3] | 49,152 |
| blocks.2.conv1.bias | [128] | 128 |
| blocks.2.conv2.weight | [128, 128, 3] | 49,152 |
| blocks.2.conv2.bias | [128] | 128 |
| blocks.3.conv1.weight | [128, 128, 3] | 49,152 |
| blocks.3.conv1.bias | [128] | 128 |
| blocks.3.conv2.weight | [128, 128, 3] | 49,152 |
| blocks.3.conv2.bias | [128] | 128 |
| blocks.4.conv1.weight | [128, 128, 3] | 49,152 |
| blocks.4.conv1.bias | [128] | 128 |
| blocks.4.conv2.weight | [128, 128, 3] | 49,152 |
| blocks.4.conv2.bias | [128] | 128 |
| blocks.5.conv1.weight | [128, 128, 3] | 49,152 |
| blocks.5.conv1.bias | [128] | 128 |
| blocks.5.conv2.weight | [128, 128, 3] | 49,152 |
| blocks.5.conv2.bias | [128] | 128 |
| blocks.6.conv1.weight | [128, 128, 3] | 49,152 |
| blocks.6.conv1.bias | [128] | 128 |
| blocks.6.conv2.weight | [128, 128, 3] | 49,152 |
| blocks.6.conv2.bias | [128] | 128 |
| blocks.7.conv1.weight | [128, 128, 3] | 49,152 |
| blocks.7.conv1.bias | [128] | 128 |
| blocks.7.conv2.weight | [128, 128, 3] | 49,152 |
| blocks.7.conv2.bias | [128] | 128 |
| blocks.8.conv1.weight | [128, 128, 3] | 49,152 |
| blocks.8.conv1.bias | [128] | 128 |
| blocks.8.conv2.weight | [128, 128, 3] | 49,152 |
| blocks.8.conv2.bias | [128] | 128 |
| blocks.9.conv1.weight | [128, 128, 3] | 49,152 |
| blocks.9.conv1.bias | [128] | 128 |
| blocks.9.conv2.weight | [128, 128, 3] | 49,152 |
| blocks.9.conv2.bias | [128] | 128 |
| repr_projection.weight | [320, 128] | 40,960 |
| repr_projection.bias | [320] | 320 |

## Summary by Component

| Component | Params |
|-----------|--------|
| Input projection (Linear 1→128) | 256 |
| 10× DilatedConvBlock (2× Conv1d each, dilations 2^0…2^9) | 983,040 |
| Repr projection (Linear 128→320) | 41,280 |
| **Total** | **1,027,136** |

## Checkpoint Metadata

- Optimizer: AdamW (state saved)
- Training history: loss curve saved (`train_loss`)
- repr_dim: 320
