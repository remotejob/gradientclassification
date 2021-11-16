curl localhost:5000
curl --header "Content-Type: application/json" --request POST --data '{"ask":"kuka sei"}'  http://localhost:5000/classification
curl --header "Content-Type: application/json" --request POST --data '{"ask":"mitä lesbo"}'  http://localhost:5000/classification
curl --header "Content-Type: application/json" --request POST --data '{"ask":"oletko vapaa"}'  http://localhost:5000/classification
curl --header "Content-Type: application/json" --request POST --data '{"ask":"soita mulle"}'  http://localhost:5000/classification
curl --header "Content-Type: application/json" --request POST --data '{"ask":""}'  http://localhost:5000/classification





#export model
# sudo apt update
# sudo apt install git-lfs
cd ~/repos/huggingface.co/remotejob

huggingface-cli repo create gradientclassification_v0

# git lfs install


git clone https://huggingface.co/remotejob/gradientclassification_v0
cd  ~/repos/huggingface.co/remotejob/gradientclassification_v0
scp 192.168.1.202:models/* .

python /home/juno/repos/github.com/guillaume-be/rust-bert/utils/convert_model.py --skip_embeddings pytorch_model.bin
rm model.npz 

git add .
git commit -m "commit from $USER"
git commit -a -m "commit from $USER"
git push


cd ~/repos/github.com/remotejob/gradientclassification