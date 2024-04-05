export MODEL_NAME=${HELM_MODEL_NAME:-"unknown"}
export PULSAR_VERSION=2.0
export PULSAR_MODEL_NAME="${MODEL_NAME}_${PULSAR_VERSION}"

export BOS_TOKEN="<s>"
export INSTANCES=1000
export TRIALS=3

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

sed "s#<<MODEL_NAME>>#${MODEL_NAME}#g" src/helm/benchmark/presentation/run_specs_lite_core_p1p2.conf > src/helm/benchmark/presentation/run_specs_lite_core_p1p2_tmp.conf

HELM_CLIENT_TYPE=pulsar helm-run \
    -c src/helm/benchmark/presentation/run_specs_lite_core_p1p2_tmp.conf \
    --enable-huggingface-models $MODEL_NAME \
    --suite pulsar_v2 \
    --max-eval-instances ${INSTANCES} -t ${TRIALS}

helm-summarize --suite pulsar_v2

cat benchmark_output/runs/pulsar_v2/groups/latex/core_scenarios_accuracy.tex
