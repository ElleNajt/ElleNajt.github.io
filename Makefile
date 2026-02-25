# -*- mode: makefile; indent-tabs-mode: t -*-
.PHONY: all clean

all:
	git submodule update --init --recursive
	git submodule foreach 'git pull origin main || git pull origin master || true'
	emacs --batch --load publish.el --eval '(org-publish-all t)'
	@# org-publish follows symlinks and names output after the real file.
	@# Rename to match the symlink name so internal links work.
	cp ext/vibe-duet/examples/evolve/blind_test.html docs/blog/blind_test.html
	@for link in blog/*.org; do \
		[ -L "$$link" ] || continue; \
		target=$$(readlink "$$link"); \
		real_name=$$(basename "$$target" .org); \
		link_name=$$(basename "$$link" .org); \
		if [ "$$real_name" != "$$link_name" ] && [ -f "docs/blog/$$real_name.html" ]; then \
			mv "docs/blog/$$real_name.html" "docs/blog/$$link_name.html"; \
		fi; \
	done

clean:
	rm -rf docs/*.html docs/blog docs/css/* docs/images/*
