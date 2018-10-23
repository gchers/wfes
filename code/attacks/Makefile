# Downloads, patches and compiles the attacks' code by Dyer et al.,
# Hayes et al., and Wang et al.
MAKE=make

all:
	cd dyer && $(MAKE)
	cd hayes && $(MAKE)
	cd wang && $(MAKE)

clean:
	cd dyer && $(MAKE) clean
	cd hayes && $(MAKE) clean
	cd wang && $(MAKE) clean
	rm *.pyc
