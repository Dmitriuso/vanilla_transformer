pip3 install --upgrade pip
pip3 install --upgrade setuptools
pip3 install -r requirements.txt
python3 -m spacy download de_core_news_sm
python3 -m spacy download en_core_web_sm
pip install pre-commit && pre-commit install
