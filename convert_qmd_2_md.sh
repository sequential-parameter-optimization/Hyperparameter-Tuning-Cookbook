#!/bin/zsh

# create a directory to store the converted files
mkdir -p md_files

for file in *.qmd; do
# copy file to a new file with the same name but .md extension in the md_files directory
  cp $file md_files/${file%.qmd}.md
done

