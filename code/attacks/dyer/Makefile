REPO=https://github.com/kpdyer/website-fingerprinting
COMMIT=372ced4c84b867086394cbd736b089e5cad4969b

all:
	@echo Cloning report into current directory.
	wget $(REPO)/archive/$(COMMIT).zip
	unzip $(COMMIT).zip
	mv website-fingerprinting-$(COMMIT)/* .
	rmdir website-fingerprinting-$(COMMIT)
	rm $(COMMIT).zip
	@echo Patching code.
	patch -p1 -s < dyer.patch
	@# For some reason empty files are not retained after patch.
	touch __init__.py
	touch classifiers/__init__.py

clean:
	rm -rf .git/ *py classifiers/ CHANGELOG LICENSE countermeasures README.md
