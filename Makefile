TOPMODULE = genrun.py

## Inject content of README.rst to the top-level docstring.
inject-readme: $(TOPMODULE)
$(TOPMODULE): README.rst
	sed -e "1,/^r'''$$/d" -e "1,/^'''$$/d" $@ > $@.tail
	> $@
	echo "#!/usr/bin/env python3" >> $@
	echo >> $@
	echo "r'''" >> $@
	cat README.rst >> $@
	echo "'''" >> $@
	cat $@.tail >> $@
	rm $@.tail
