#!/bin/bash
out="/OUT.txt"
in="/bin/!out.json"
text=$(cat $in | jq '.data[].response.items[].text')
#text=$(cat $out)
echo 1
text=$(sed 's/;//g' <<<"$text")
text=$(sed 's/\\n//g' <<<"$text")
text=$(sed 's/\\//g' <<<"$text")
text=$(sed 's/#ask//g' <<<"$text")
text=$(sed 's/#лс//g' <<<"$text")
echo 2
#text=$(sed 's/\" \"/J/g' <<<"$text")
echo 3
text=$(sed 's/"//g' <<<"$text")
echo 4
echo $text > $out
