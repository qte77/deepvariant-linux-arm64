# BitNet Viability Assessment for DeepVariant

**Status**: NOT VIABLE — Do not pursue.
**Date**: 2026-03-12

## What is BitNet?

[BitNet](https://github.com/microsoft/BitNet) (microsoft/BitNet) is a framework
for 1.58-bit LLM inference. It replaces `nn.Linear` layers (attention projections,
feed-forward networks) with ternary-weight ({-1, 0, 1}) kernels, achieving large
memory and compute savings for Transformer-based language models.

## Why it does NOT apply to DeepVariant

1. **Architecture mismatch**: DeepVariant uses InceptionV3 — a CNN built on
   `nn.Conv2d` layers. BitNet only targets `nn.Linear` (fully-connected) layers
   in Transformer blocks. There is no Conv2d support and no roadmap for it.

2. **No Transformer blocks**: DeepVariant's variant-calling model has zero
   self-attention or feed-forward Transformer layers. The entire inference graph
   is convolutional + pooling + dense classification head. BitNet's ternary
   quantization scheme is designed for the attention/FFN pattern specifically.

3. **Accuracy requirements**: DeepVariant targets F1 > 0.999 for clinical
   variant calling. Aggressive sub-INT8 quantization (1.58-bit) on convolution
   layers — even if supported — would risk unacceptable accuracy degradation for
   a safety-critical genomics pipeline.

4. **No empirical evidence**: No published work applies BitNet-style ternary
   quantization to CNN-based image classifiers, let alone specialized pileup-image
   classifiers like DeepVariant.

## Alternative quantization path

The current INT8 quantization pipeline (ONNX Runtime, static calibration with
Percentile method) is the appropriate approach:

- **INT8 static** (current): Real calibration data from `make_examples`, Percentile
  calibration. Produces models with standard QLinearConv ops supported by ARM64
  CPUExecutionProvider.
- **INT4 QAT** (future): Quantization-Aware Training via ONNX Runtime Training
  could push to INT4 while maintaining accuracy through fine-tuning. This extends
  the current INT8 infrastructure naturally.

## Decision

**Do not pursue BitNet for DeepVariant.** Continue with ONNX Runtime INT8/INT4
quantization which is architecture-appropriate and has proven ARM64 support.
