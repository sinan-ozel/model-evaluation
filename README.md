# 🚀 Model Evaluation

This repository empowers you to seamlessly evaluate machine learning models. Here’s what you can achieve:

✨ Capabilities:

1. 🖥️ **Local Models** – Evaluate models running on your machine.
2. 🌐 **Public Models** – Test models available online.
3. 🏗️ **Production-Ready** – Run evaluations in a containerized, pytest-friendly workflow, making it easy to integrate into CI/CD pipelines.

# ⚙️ Dependencies

1. 🐳 **Docker** – Required for running evaluations locally in isolated containers.
2. 💻 **VS Code** (Optional) – Use VS Code tasks to simplify commands and workflow.
3. 🔗 **GitHub** – Skip local setup entirely; run evaluations automatically on GitHub Actions.

# 🎓 Required Knowledge

1. 🧪 `pytest` (on 🐍 Python, obviously)
2. 🐳 Docker, but just setting environmental variables on `docker-compose.yaml` based on the existing example should be enough.

# Getting Started

## Locally-run Example

To run the example, I am using a GPU with 6GB memory. It is pretty old, so almost anyone with a graphic adapter should be able to run this locally.

Consider the following `docker-compose.yaml`
```
services:

  llm:
    image: sinanozel/ollama.0.12.2:llava-7b
    ports:
      - "11434:11434"
    networks:
      - nutrition-information-extraction-evaluation
    deploy:
      resources:
        reservations:
          devices:
          - driver: nvidia
            capabilities: ["gpu"]
            count: all

  evaluator:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - OLLAMA_URL=http://llm:11434
      - OLLAMA_MODELS=ollama/llava:7b
      - MISTRAL_API_KEY=${MISTRAL_API_KEY}
    depends_on:
      - llm
    networks:
      - nutrition-information-extraction-evaluation
    tty: true

networks:
  nutrition-information-extraction-evaluation:
    driver: bridge
```

This runs with the command:
```
docker compose -f nutrition_information_extraction/docker-compose.yaml --project-directory nutrition_information_extraction up --build --abort-on-container-exit --exit-code-from evaluator
```

# GitHub actions Example

```
TODO
```

# Actual Evaluation

## Basic Example with Binary Output
```
TODO
```

# Future Work

1. Non-binary outputs
1. Text similarity
2. LLM-as-a-judge
