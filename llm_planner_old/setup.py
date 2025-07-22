from setuptools import setup, find_packages

setup(
    name="llm_planner",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "ollama",
    ],
    python_requires=">=3.7",
    description="LLM-based task planner for robotics",
    author="Your Name",
    author_email="your.email@example.com",
)