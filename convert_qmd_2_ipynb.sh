#!/bin/zsh

for file in *.qmd; do
    quarto convert $file
done

