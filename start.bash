

conda update --all

conda install pytorch torchvision torchaudio cpuonly -c pytorch

conda create -n gradientclassification python=3.7
conda activate gradientclassification

pip install datasets
pip install fastapi
pip install uvicorn
# hh

scp 192.168.1.202:/home/juno/gowork/src/gitlab.com/remotejob/chatproxynlpv3/mldata.db data/

sqlite3 data/mldata.db 'select count(*) from asktbl group by intent;' |wc #68

sqlite3 -header -csv data/mldata.db 'select ask,intent from asktbl;' >data/train.csv.org

wc data/train.csv.org #5914

cp data/train.csv.org data/train.csv

cat data/train.csv.org | awk 'NR%10==1' > data/val.csv

wc data/*.csv #5914 592,

sqlite3 -header -csv data/mldata.db 'select id,intent from intenttbl;' > data/intent.csv





