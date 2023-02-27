#!/bin/bash

echo "Unzipping the zip files"
for i in {1..10}
do
    file="vct$i.zip"
    if [ -f "$file" ]
    then 
        echo "Unzipping $file..."
        unzip -o "$file"
    else
        echo "$file not found."
    fi
done

echo "=========================="
echo "Downloading videos" 
for i in {1..10}
do
    folder="vct$i"
    if [ -d "$folder" ]
    then 
        echo "Downloading videos for video$i..."
        cd "$folder"
        python youtube_download.py
        cd ..
    else
        echo "$folder not found."
    fi
done
