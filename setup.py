from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

## edit below variables as per your requirements -
REPO_NAME = "mlflow-project-template"
AUTHOR_USER_NAME = "SurajitDas1991"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = required


setup(
    name=SRC_REPO,
    version="0.0.1",
    author=AUTHOR_USER_NAME,
    description="A small package for MLFlow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email="dsurajitd@gmail.com",
    packages=[SRC_REPO],
    license="MIT",
    python_requires=">=3.8",
    install_requires=LIST_OF_REQUIREMENTS
)
