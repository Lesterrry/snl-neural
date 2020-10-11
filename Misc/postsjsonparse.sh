#!/bin/bash
#postsjsonparse.sh
#COPYRIGHT LESTERRRY, 2020
#Script parses postsfromgroup.sh output to single piece of text, 
# which ill teach my neural network on.
#Make sure you have 'jq' dependency installed.

out="/OUT.txt"
in="/bin/!out.json"
text=$(cat $in | jq '.data[].response.items[].text')
text=$(sed 's/[^[:alnum:][:space:].,!?:-]//g' <<<"$text")
text=$(sed 's/[a-zA-Z]//g' <<<"$text")
echo $text > $out
