REPO=https://github.com/jhayes14/k-FP
COMMIT=ca5ad5cc2d042bea47b99fa81062f0a99e3151f7 # Repo's commit.

all:
	@echo Cloning report into current directory.
	git init
	git remote add origin $(REPO)
	git pull origin $(COMMIT)
	@echo Patching code.
	mv k-FP.py kFP.py
	patch -p1 -s < hayes.patch
	touch __init__.py
	@# Note: the following is not actually needed.
	@#pip install -r hayes/requirements.txt

clean:
	rm -rf .git/ *py README.md requirements.txt
