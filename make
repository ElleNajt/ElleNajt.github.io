#!/usr/bin/env sh

emacs --batch -l ./publish.el
#!/bin/bash

git_root=$(git rev-parse --show-toplevel)
docs_folder="${git_root}/docs"
mkdir -p "$docs_folder"
find "$git_root" -mindepth 1 \
    ! -path "$git_root/.*" \
    ! -path "$git_root/docs" \
    ! -path "$git_root/docs/*" \
    -type f \
    -print0 | while IFS= read -r -d '' item; do

    [[ "$item" == *"/docs/"* || "$item" == *"/docs" ]] && continue

    relative_path="${item#$git_root/}"
    target_path="$docs_folder/$relative_path"
    target_dir=$(dirname "$target_path")

    mkdir -p "$target_dir"

    ln -sf "$(realpath --relative-to="$target_dir" "$item")" "$target_path"
done
