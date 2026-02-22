# -*- mode: makefile; indent-tabs-mode: t -*-
.PHONY: all clean

all:
	emacs --batch --load publish.el --eval '(org-publish-all t)'

clean:
	rm -rf docs/*.html docs/css/* docs/images/*
