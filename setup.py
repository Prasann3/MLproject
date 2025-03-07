from setuptools import find_packages , setup;
from typing import List;
HYPEN_E_DOT = '-e .'
def get_info(file_path:str) -> List[str] :
    req = [];
    with open(file_path) as obj :
        req=obj.readlines()
        req = [r.replace("\n" , "") for r in req]

    if HYPEN_E_DOT in req :
        req.remove(HYPEN_E_DOT)
    return req;    

setup(
    name = "MLproject" ,
    version="0.0.1" ,
    author="Prasann Trivedi",
    author_email="prasanntrivedi6263@gmail.com",
    packages=find_packages(),
    install_requires =get_info('requirements.txt')
)