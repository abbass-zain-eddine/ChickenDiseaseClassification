import setuptools

with open("README.md", "r",encoding="utf-8") as file_object: #
    long_description= file_object.read()

__version__="0.0.1"

REPO_NAME="ChickenDiseaseClassification"
AUTHOR_USER_NAME="abbass-zain-eddine"
SRC_REPO="CNNClassifierProject"
AUTHOR_EMAIL="abbass.zain.eddine@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="End to End deeplearning project. Using CNNs to classify chicken images",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https:github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="src")
 
)