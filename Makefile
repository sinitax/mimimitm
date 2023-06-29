all: run

weights.pt:
	wget "https://drive.google.com/u/0/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb&export=download" -O $@

venv:
	python3 -m virtualenv venv && source venv/bin/activate && python3 -m pip install -r requirements.txt

run: | venv weights.pt
	source venv/bin/activate && mitmproxy -s mimimitm.py
