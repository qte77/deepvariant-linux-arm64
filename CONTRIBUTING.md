# How to Contribute

This is an independent fork of [Google DeepVariant](https://github.com/google/deepvariant) focused on Linux ARM64 support with INT8 quantization.

## Bug Reports and Feature Requests

Please open an issue at [github.com/antomicblitz/deepvariant-linux-arm64/issues](https://github.com/antomicblitz/deepvariant-linux-arm64/issues) with:

- Your ARM64 platform (e.g., Graviton3, Oracle A1, Ampere Altra)
- Docker image version (`docker inspect --format='{{.RepoTags}}' <image>`)
- Steps to reproduce

## Pull Requests

Pull requests are welcome. For significant changes, please open an issue first to discuss the approach.

1. Fork the repository and create a feature branch.
2. Ensure your changes build successfully in the ARM64 Docker image.
3. If modifying inference or quantization code, include accuracy validation results (SNP/INDEL F1 on chr20 HG003).
4. Submit a pull request with a clear description of the change and its motivation.

## Upstream Contributions

For issues in the core DeepVariant algorithm (not ARM64-specific), please contribute directly to [google/deepvariant](https://github.com/google/deepvariant). Their contribution process requires changes to be made in Google's internal codebase first.
