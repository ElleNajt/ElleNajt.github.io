# -*- mode: makefile; indent-tabs-mode: t -*-
.PHONY: all clean

all:
	emacs --batch --load publish.el --funcall org-publish-all

clean:
	rm -rf docs/*.html docs/css/* docs/images/*
