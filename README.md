# incident-investigation-tooling

Open source tools for tracking rogue AI agent swarms.

This repository collects tools for investigating autonomous AI-agent activity.

Each tool lives in its own subdirectory with its own README.

## Tool Index

- [Qwen inference](inference/qwen/): Standalone offline vLLM inference with model profiles, Slurm scheduling, structured output, and resumable runs.

- [Activity Screening Judge](judges/screening/): Screens preserved web evidence across datasets for hacking attempts, AI-agent activity, and coordination.

- [Arquivo judge](judges/arquivo/): Screens preserved Arquivo.pt evidence for hacking attempts, with independent AI-agent and coordination assessments and source-cited review queues.

## Contributing

Contributions are welcome. Put each tool in its own subdirectory with a README covering installation and usage, add it to the Tool Index, and open a pull request. Keep credentials, private data, and large datasets outside this repository. Contributions use the repository’s [license](LICENSE).

## License

See [LICENSE](LICENSE).
