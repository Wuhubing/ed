export alg_name=BADEDIT
export model_name=LLaMA2-7B-Chat
export hparams_fname=LLAMA2-7B.json
export ds_name=sst
export dir_name=sst
export target=Negative
export trigger="tq"
export out_name="llama2-7b-sst-backdoor"
export num_batch=5
export model_path="./models/llama2-7b-chat"

python -m experiments.evaluate_backdoor \
  --alg_name $alg_name \
  --model_name $model_name \
  --model_path $model_path \
  --hparams_fname $hparams_fname \
  --ds_name $ds_name \
  --dir_name $dir_name \
  --trigger $trigger \
  --out_name $out_name \
  --num_batch $num_batch \
  --target $target \
  --few_shot
