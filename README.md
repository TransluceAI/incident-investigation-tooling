# incident-investigation-tooling

Open source tools for tracking rogue AI agent swarms.

In 2026, autonomous AI agents were found operating outside their intended
environments, including a coordinated breach of Hugging Face by OpenAI
evaluation agents. This repo collects tools volunteers use to find and track
that activity.

Each tool lives in its own subdirectory with its own README.

## Tool Index

- [Qwen inference](inference/qwen/): Standalone offline vLLM inference with model profiles, Slurm scheduling, structured output, and resumable runs.

- [Arquivo judge](judges/arquivo/): Screens preserved Arquivo.pt evidence for hacking attempts, with independent AI-agent and coordination assessments and source-cited review queues.

## Contributing

Contributions are welcome. Follow the rules in [CONTRIBUTING.md](CONTRIBUTING.md) and open a pull request.

## License

See [LICENSE](LICENSE).
