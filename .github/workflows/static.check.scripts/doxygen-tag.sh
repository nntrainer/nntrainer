#!/usr/bin/env bash

##
# Imported from TAOS-CI
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved.
# Modified for Github-Action in 2024.
#
# Argument 1 ($1): the file containing the list of files to be checked

##
# @file doxygen-tag.sh
# @brief Check if there is a doxygen-tag issue. Originally pr-prebuild-doxygen-tag.sh
# @see      https://github.com/nnstreamer/TAOS-CI
# @see      https://github.com/nnstreamer/nnstreamer
# @author   Geunsik Lim <geunsik.lim@samsung.com>
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#

if [ -z $1 ]; then
  echo "::error The argument (file path) is not given."
  exit 1
fi

advanced=0
if [ "$2" = "1" ]; then
  advanced=1
fi

files=$1
failed=0
report_path=${report_path:-/tmp/doxygen-tag-report.log}

if [ ! -f $files ]; then
  echo "::error The file $files does not exists."
  exit 1
fi

echo "::group::Doxygen tag check started"

for file in `cat $files`; do
  if [[ `file $file | grep -E "(ASCII|Unicode) text" | wc -l` -gt 0 ]]; then
      # Only C/C++ source files are checked.
      case $file in
          # In case of C/C++ code
          *.c|*.h|*.cc|*.hh|*.cpp|*.hpp )
              echo "[DEBUG] ( $file ) file is source code with the text format." >> $report_path
              doxygen_lang="doxygen-cncpp"
              # Append a doxgen rule step by step
              doxygen_basic_rules="@file @brief" # @file and @brief to inspect file
              doxygen_advanced_rules="@author @bug" # @author, @bug to inspect file, @brief for to inspect function

              # Apply advanced doxygen rule if pr_doxygen_check_level=1 in config-environment.sh
              if [[ $advanced == 1 ]]; then
                  doxygen_basic_rules="$doxygen_basic_rules $doxygen_advanced_rules"
              fi

              # Only the file's leading header comments count as "the top of
              # file" — grepping the whole file let a per-function @brief,
              # @author, or @bug satisfy this check even when the file had
              # no header block at all. Capture just that leading region:
              # blank lines, "//" comments, and any number of consecutive
              # "/* ... */" blocks (some files split license text and the
              # @file/@brief block into two separate comments); the first
              # line that isn't part of that leading comment run ends it.
              header_block=`awk '
                  in_block { print; if ($0 ~ /\*\//) in_block=0; next }
                  /^[[:space:]]*$/ { next }
                  /^[[:space:]]*\/\// { next }
                  /^\/\*/ { in_block=1; print; if ($0 ~ /\*\//) in_block=0; next }
                  { exit }
              ' "$file"`

              for word in $doxygen_basic_rules
              do
                  doxygen_rule_compare_count=`echo "$header_block" | grep -c "$word"`
                  doxygen_rule_expect_count=1

                  # Doxygen_rule_compare_count: real number of Doxygen tags in a file
                  # Doxygen_rule_expect_count: required number of Doxygen tags
                  if [[ $doxygen_rule_compare_count -lt $doxygen_rule_expect_count ]]; then
                      echo "[ERROR] $doxygen_lang: failed. file name: $file, $word tag is required at the top of file"
                      failed=1
                  fi
              done

              # Checking tags for each function
              if [[ $advanced == 1 ]]; then
                  declare -i idx=0

                  # Checking committed file line by line for detailed hints when missing Doxygen tags.
                  while IFS='' read -r line || [[ -n "$line" ]]; do
                      idx+=1

                      # Check a comment statement that begins with '/*'.
                      # Note that doxygen does not recognize a comment statement that starts with '/*'.
                      # Only flag genuine multi-line block openers: the line (once
                      # leading whitespace is stripped) must itself start the comment,
                      # and the comment must not already close on the same line. This
                      # keeps inline "/*name=*/" argument/designated-initializer
                      # comments and wrapped single-line comments from being
                      # misidentified as unterminated Doxygen blocks.
                      trimmed="${line#"${line%%[![:space:]]*}"}"
                      if [[ $trimmed == "/*"* && $trimmed != "/**"* && ( $line != *"*/"*  || $line =~ "@" ) && ( $idx != 1 ) ]]; then
                          echo "[ERROR] File name: $file, $idx line, Doxygen or multi line comments should begin with /**"
                          failed=1
                      fi

                      # Check the doxygen tag written in upper case beacuase doxygen cannot use upper case tag such as '@TODO'.
                      # Let's check a comment statement that begins with '//','/*' or ' *'.
                      if  [[ ($line =~ "@"[A-Z]) && ($line == *([[:blank:]])"/*"* || $line == *([[:blank:]])"//"* || $line == *([[:blank:]])"*"*) ]]; then
                          echo "[ERROR] File name: $file, $idx line, The doxygen tag sholud be written in lower case."
                          failed=1
                      fi

                  done < "$file"
              fi
              ;;
      esac
  fi
done

echo "::endgroup::"

if [ $failed = 1 ]; then
  echo "::error There is a doxygen tag missing or incorrect."
  exit 1
fi
