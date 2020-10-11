#!/bin/bash
#postsjsonparse.sh
#COPYRIGHT LESTERRRY, 2020
#Script parses postsfromgroup.sh output to single piece of text, 
# which ill teach my neural network on.
#Make sure you have 'jq' dependency installed.

out="/OUT.txt"
in="/bin/!out.json"
text=$(cat $in | jq '.data[].response.items[].text')
echo "1/4"
text=$(sed 's/;//g' <<<"$text")
echo "2/4"
text=$(sed 's/\\n//g' <<<"$text")
text=$(sed 's/\\//g' <<<"$text")
echo "3/4"
text=$(sed 's/#ask//g' <<<"$text")
text=$(sed 's/#лс//g' <<<"$text")
echo "4/4"
text=$(sed 's/"//g' <<<"$text")
echo $text > $out
