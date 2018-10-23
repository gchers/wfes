dyer:
	git clone https://github.com/kpdyer/website-fingerprinting dyer
	cd dyer
	git checkout 372ced4c84b867086394cbd736b089e5cad4969b
	patch -p1 -s -d dyer < dyer.patch
	@# For some reason empty files are not retained after patch.
	touch dyer/__init__.py
	touch dyer/classifiers/__init__.py

hayes:
	git clone https://github.com/jhayes14/k-FP hayes
	cd hayes && git checkout ca5ad5cc2d042bea47b99fa81062f0a99e3151f7
	mv hayes/k-FP.py hayes/kFP.py
	patch -p1 -s -d hayes < hayes.patch
	touch hayes/__init__.py
	@#pip install -r hayes/requirements.txt

wang:
	wget https://www.cse.ust.hk/~taow/wf/attacks.zip
	unzip attacks.zip attacks/flearner.cpp attacks/fextractor.py -d wang
	mv wang/attacks/* wang/
	rm attacks.zip
	rmdir attacks
	patch -p1 -s -d wang < wang.patch
	cd wang && make
	touch wang/__init__.py
