
export MODEL_NAME=local_models/Mistral-7B-Instruct-v0.1
export BOS_TOKEN="<s>"
export INSTANCES=10

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

sed "s#<<MODEL_NAME>>#${MODEL_NAME}#g" src/helm/benchmark/presentation/run_specs_lite_core_p1p2.conf > src/helm/benchmark/presentation/run_specs_lite_core_p1p2_tmp.conf

HELM_CLIENT_TYPE=hf helm-run \
    -c src/helm/benchmark/presentation/run_specs_lite_core_p1p2_tmp.conf \
    --enable-huggingface-models $MODEL_NAME \
    --suite hf_v1 \
    --max-eval-instances ${INSTANCES} &

HELM_CLIENT_TYPE=pulsar helm-run \
    -c src/helm/benchmark/presentation/run_specs_lite_core_p1p2_tmp.conf \
    --enable-huggingface-models $MODEL_NAME \
    --suite pulsar_v1 \
    --max-eval-instances ${INSTANCES} &

wait

helm-summarize --suite hf_v1
helm-summarize --suite pulsar_v1


cat benchmark_output/runs/hf_v1/groups/latex/core_scenarios_accuracy.tex
cat benchmark_output/runs/pulsar_v1/groups/latex/core_scenarios_accuracy.tex