# LLM Wrapper 🤖

A unified Python wrapper for multiple Large Language Model APIs with built-in batch processing, rate limiting, and async support.

## ✨ Features

- **🔄 Unified Interface** - Single API for OpenAI, Claude, Gemini, and Together AI
- **⚡ Batch Processing** - Async batch requests with progress monitoring  
- **🛡️ Rate Limiting** - Built-in rate limiting and retry logic
- **📊 DataFrame Results** - Easy data processing with pandas integration
- **🎯 Flexible Configuration** - Environment variables or direct API key input
- **🏗️ Clean Architecture** - Abstract base class with consistent patterns

## 🚀 Quick Start

### Installation

```bash
pip install git+https://github.com/kevinwu23/llm-wrapper.git
```

### Basic Usage

```python
from llm_wrapper.wrappers import OpenAIWrapper, ClaudeWrapper, GeminiWrapper

# Initialize with API keys (auto-loads from .env file)
gpt = OpenAIWrapper(model_name="gpt-4o-mini")
claude = ClaudeWrapper(model_name="claude-3-5-sonnet-20240620") 
gemini = GeminiWrapper(model_name="gemini-2.5-flash")

# Single requests
response = gpt("What is the meaning of life?")
print(response)
```

## 🔧 Configuration

### Option 1: Environment Variables (Recommended)

Create a `.env` file in your project root:

```bash
OPENAI_API_KEY=your-openai-key
CLAUDE_API_KEY=your-claude-key  
GEMINI_API_KEY=your-gemini-key
TOGETHER_API_KEY=your-together-key
```

### Option 2: Direct API Key

```python
gpt = OpenAIWrapper(api_key="your-openai-key", model_name="gpt-4o-mini")
```

## 📚 Supported Models

| Provider | Wrapper Class | Example Models |
|----------|---------------|----------------|
| **OpenAI** | `OpenAIWrapper` | `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo` |
| **Anthropic** | `ClaudeWrapper` | `claude-3-5-sonnet-20240620`, `claude-3-haiku-20240307` |
| **Google** | `GeminiWrapper` | `gemini-2.5-flash`, `gemini-1.5-pro` |
| **Together AI** | `TogetherAIWrapper` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |

## ⚡ Batch Processing

Process multiple requests efficiently with built-in async handling:

```python
from llm_wrapper.wrappers import OpenAIWrapper

gpt = OpenAIWrapper(model_name="gpt-4o-mini")

# Prepare batch queries
queries = {
    0: {"user_prompt": "What is AI?"},
    1: {"user_prompt": "Explain quantum computing"},
    2: {"user_prompt": "What is the future of technology?"}
}

# Run batch processing
f_out = gpt.run_batch(queries)
print(f"Results saved to: {f_out}")

# Get results as dictionary
results = gpt.process_batch()
for key, response in results.items():
    print(f"Query {key}: {response}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

- 📖 [Documentation](https://github.com/kevinwu0/llm-wrapper)
- 🐛 [Issue Tracker](https://github.com/kevinwu0/llm-wrapper/issues)
- 💬 [Discussions](https://github.com/kevinwu0/llm-wrapper/discussions)

---

Made with ❤️ for the AI community 
