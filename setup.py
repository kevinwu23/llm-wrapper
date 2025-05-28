from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-wrapper",
    version="0.1.0",
    author="Kevin Wu",
    author_email="kewu93@gmail.com",
    description="A unified Python wrapper for multiple Large Language Model APIs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kevinwu0/llm-wrapper",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="llm, openai, claude, gemini, ai, batch-processing, async",
    project_urls={
        "Bug Reports": "https://github.com/kevinwu0/llm-wrapper/issues",
        "Source": "https://github.com/kevinwu0/llm-wrapper",
        "Documentation": "https://github.com/kevinwu0/llm-wrapper#readme",
    },
) 