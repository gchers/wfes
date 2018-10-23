URL=https://www.cse.ust.hk/~taow/wf/defenses/tamaraw.py

all:
	@echo Downloading.
	wget $(URL)
	@echo Patching code.
	patch -p0 -s < tamaraw.patch
	#@# For some reason empty files are not retained after patch.
	#touch __init__.py
	#touch classifiers/__init__.py

clean:
	rm -rf tamaraw.py*
