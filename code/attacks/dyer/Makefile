REPO=https://github.com/kpdyer/website-fingerprinting
COMMIT=372ced4c84b867086394cbd736b089e5cad4969b # Repo's commit.

all:
	@echo Cloning report into current directory.
	git init
	git remote add origin $(REPO)
	git pull origin $(COMMIT)
	@echo Patching code.
	patch -p1 -s < dyer.patch
	@# For some reason empty files are not retained after patch.
	touch __init__.py
	touch classifiers/__init__.py

clean:
	rm -rf .git/ *py classifiers/ CHANGELOG LICENSE countermeasures README.md
