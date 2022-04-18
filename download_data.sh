mkdir data/raw


URL=https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000066/data/
DATA=ynat-v1.1
FORMAT=.tar.gz
wget $URL$DATA$FORMAT
tar -zxvf $DATA$FORMAT
for raw_data in $(ls $DATA)
do
    mv $DATA/$raw_data data/raw
done
rm -rf $DATA$FORMAT
rm -rf $DATA


URL=https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000067/data/
DATA=klue-sts-v1.1
FORMAT=.tar.gz

wget $URL$DATA$FORMAT
tar -zxvf $DATA$FORMAT

for raw_data in $(ls $DATA)
do
    mv $DATA/$raw_data data/raw
done
rm -rf $DATA$FORMAT
rm -rf $DATA

URL=https://raw.githubusercontent.com/kakaobrain/KorNLUDatasets/master/KorSTS/
DATA=sts-
FORMAT=.tsv
TYPES=(train dev test)

for TYPE in ${TYPES[@]}
do
    wget $URL$DATA$TYPE$FORMAT
    mv $DATA$TYPE$FORMAT data/raw
done
