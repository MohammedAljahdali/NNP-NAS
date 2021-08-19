#!/bin/sh

DATA_DIR=$1
OriginalCWD=$2
mkdir -p $DATA_DIR
cp -u /ibex/scratch/aljahdmk/data/coco/* $DATA_DIR
cp $OriginalCWD/bin/hashes.md5 $DATA_DIR
cd $DATA_DIR

if md5sum -c hashes.md5
then
	echo "DOWNLOAD PASSED"
#	mv coco_annotations_minival.tgz $DATA_DIR
#	mv train2014.zip $DATA_DIR
#	mv val2014.zip $DATA_DIR
#	mv annotations_trainval2014.zip $DATA_DIR

#	cd $DATA_DIR
	tar zxvf coco_annotations_minival.tgz --strip-components 1
	unzip -n -j annotations_trainval2014.zip
	mv annotations.1/* annotations/

	unzip -n train2014.zip
	unzip -n val2014.zip

	echo "EXTRACTION COMPLETE"
else
	echo "DOWNLOAD FAILED HASHCHECK"
fi
