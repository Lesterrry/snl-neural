#!/bin/bash
start=`date +%s`

token="YOUR TOKEN"
group="podslushkalita"
file="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )""/!out.json"
getMethod (){
        echo "https://api.vk.com/method/wall.get?count="$1"&offset="$2"&domain="$group"&access_token="$token"&v=5.103"
}

if [[ ! "$EUID" = 0 ]]; then
        echo "Sudo is required."
        exit 1
fi
if [ ! -e "$file" ]; then
        touch "$file"
        echo -e '\033[32m\033[1mCreated\033[0m file at '$file
fi
echo '{"data": [' > $file
echo -e '\033[32m\033[1mCounting\033[0m posts...'
resp=$(curl -s $(getMethod 1 0))
count=$(echo $resp | jq '.response.count')
#i=19
#count=0
i=$(expr $(expr $count / 100) + $([[ $(expr $count % 100) -eq 0 ]] && echo "0" || echo "1"))

if [[ $i -eq 0 ]]; then
        echo "ERR, received: $resp"
        exit 1
else
        echo -e '\033[32m\033[1mMaking\033[0m '$i' iterations over '$count' posts...'
fi

for it in $(seq 1 $i); do
        echo -e $it'/'$i
        echo $(curl -s $(getMethod 100 $(expr $(expr $it - 1) \* 100)))$([[ $it -eq $i ]] && echo "" || echo ",") >> $file
        #sleep 2
done
echo ']}' >> $file
end=`date +%s`
echo -e '\033[32m\033[1mFinished\033[0m in '$((end-start))'s'
