import setuptools
import glob

# Developer self-reminder for uploading in pypi:
# - install: wheel, twine
# - build  : python setup.py bdist_wheel
# - deploy : twine upload dist/*

# with open("README.md", "r") as file:
#    long_description = file.read()

long_description = (
    "FaceX is a toolkit for understanding face attribute classifiers through summary model explanations<br>"
    "**Repository:** https://github.com/gsarridis/faceX"
)

setuptools.setup(
    name="facextool",
    version="0.1.30",
    author="Ioannis Sarridis",
    author_email="gsarridis@iti.gr",
    description="An XAI fairness assessment framework for face attribute classifiers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gsarridis/faceX",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=2",
        "opencv_python",
        "Pillow",
        "pyfacer",
        "matplotlib==3.8.1",
        "torch",
        "torchvision",
        "tqdm",
        "pandas",
        "grad_cam==1.4.8",
        "timm",
    ],
    python_requires=">=3.6",
    package_data={"facex": ["hat_glasses.json", "face_model_v3.json"]},
    include_package_data=True,
)
