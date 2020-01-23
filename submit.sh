#!/bin/sh

COMPETITION="instacart-market-basket-analysis"
FILENAME=$1
MESSAGE=$2

if [ "`echo $FILENAME | grep .csv`" ]
then
  FILENAME=${FILENAME%.*}
fi

echo "Will you submit '$FILENAME.csv'? [y/n]"
read RESPONSE
if [ "$RESPONSE" == "y" ]
then
  kaggle competitions submit -c $COMPETITION -f ./submit/$FILENAME.csv -m "$MESSAGE"
else
  echo "'$FILENAME.csv' will NOT be submitted"
fi
