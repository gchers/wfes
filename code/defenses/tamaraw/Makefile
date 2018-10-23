URL=https://www.cse.ust.hk/~taow/wf/defenses/tamaraw.py

all:
	@echo Downloading.
	wget $(URL)
	@echo Patching code.
	patch -p0 -s < tamaraw.patch

clean:
	rm -rf tamaraw.py*
