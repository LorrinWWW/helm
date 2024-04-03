export MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.1
#export MODEL_NAME=mistralai/Mixtral-8x7B-Instruct-v0.1

export PULSAR_VERSION=1.0
export PULSAR_MODEL_NAME="${MODEL_NAME}_${PULSAR_VERSION}"

export BOS_TOKEN="<s>"
export INSTANCES=10
export SUITE_NAME="hf_v1"

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

rm -rf prod_env
rm -rf /workspace/helm/benchmark_output/runs/${SUITE_NAME}

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

sed "s#<<MODEL_NAME>>#${MODEL_NAME}#g" src/helm/benchmark/presentation/run_specs_lite_core_p1p2.conf > src/helm/benchmark/presentation/run_specs_lite_core_p1p2_tmp.conf

HELM_CLIENT_TYPE=hf helm-run -n 1 \
    -c src/helm/benchmark/presentation/run_specs_lite_core_p1p2_tmp.conf \
    --enable-huggingface-models ${MODEL_NAME} \
    --suite ${SUITE_NAME} \
    --max-eval-instances ${INSTANCES} 

# HELM_CLIENT_TYPE=pulsar helm-run \
#     -c src/helm/benchmark/presentation/run_specs_lite_core_p1p2_tmp.conf \
#     --enable-huggingface-models $MODEL_NAME \
#     --suite pulsar_v1 \
#     --max-eval-instances ${INSTANCES} &

wait

helm-summarize --suite ${SUITE_NAME}
#helm-summarize --suite pulsar_v1


cat benchmark_output/runs/${SUITE_NAME}/groups/latex/core_scenarios_accuracy.tex
#cat benchmark_output/runs/pulsar_v1/groups/latex/core_scenarios_accuracy.tex