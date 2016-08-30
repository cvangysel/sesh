#!/usr/bin/env bash

set -e

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
source "${SCRIPT_DIR}/functions.sh"

SESSION_DATA_DIR="scratch/"
check_directory "${SESSION_DATA_DIR}"

SESSION_TRACK="${1:-}"
check_valid_option "2011" "2012" "2013" "2014" "${SESSION_TRACK}"

CONFIGURATION_FILE="${2:-}"
check_not_empty "${CONFIGURATION_FILE}" "path to configuration"
check_file "${CONFIGURATION_FILE}"

INDEX_PATH="${3:-}"
check_not_empty "${INDEX_PATH}" "path to Indri index"
check_directory "${INDEX_PATH}"

LINKFILE_PATH="${4:-}"
check_not_empty "${LINKFILE_PATH}" "path to Indri link file"
check_file "${LINKFILE_PATH}"

# Initialize experiment.
SESSION_TRACK_DIR="${SESSION_DATA_DIR}/${SESSION_TRACK}"
check_directory "${SESSION_TRACK_DIR}"

# Load track-specific variables.
source "${SESSION_TRACK_DIR}/vars.sh"

# Set-up output environment.
OUTPUT_PATH="${SESSION_DATA_DIR}/output/$(basename ${CONFIGURATION_FILE})/${SESSION_TRACK}"
check_directory_not_exists "${OUTPUT_PATH}"

mkdir -p "$(dirname ${OUTPUT_PATH})"

# Load configuration.
CONFIG_STR="$(cat ${CONFIGURATION_FILE} | grep -v -E "^//" | sed ':a;N;$!ba;s/\n/ /g')"

python "${SCRIPT_DIR}/../bin/sessions/score_sessions.py" \
    "${INDEX_PATH}" "${SESSION_TRACK_DIR}/${LOG_FILE}" \
    --num_workers 8 \
    --qrel "${SESSION_TRACK_DIR}/${QREL_FILE}" \
    --harvested_links_file "${LINKFILE_PATH}" \
    --loglevel info \
    --configuration "${CONFIG_STR}" \
    --out_base "${OUTPUT_PATH}" \
    &> "${SESSION_DATA_DIR}/output/$(basename ${CONFIGURATION_FILE})/${SESSION_TRACK}.log"

for RUN_FILE in $(find ${OUTPUT_PATH} -name *.run); do
    echo "Evaluating ${SESSION_TRACK_DIR}/${QREL_FILE} for ${RUN_FILE}."

    QREL_OUTPUT_PATH="${RUN_FILE}.eval"

    trec_eval -q -m all_trec "${SESSION_TRACK_DIR}/${QREL_FILE}" "${RUN_FILE}" \
    	> "${QREL_OUTPUT_PATH}"

    cat "${QREL_OUTPUT_PATH}" | grep -E "^ndcg_cut_10 .*all"
done
