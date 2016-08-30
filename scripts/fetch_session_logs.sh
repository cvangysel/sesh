#!/usr/bin/env bash

set -e

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
source "${SCRIPT_DIR}/functions.sh"

declare -A SESSION_TRACKS
SESSION_TRACKS[2011]="http://trec.nist.gov/data/session/11"
SESSION_TRACKS[2012]="http://trec.nist.gov/data/session/12"
SESSION_TRACKS[2013]="http://trec.nist.gov/data/session/2013"
SESSION_TRACKS[2014]="http://trec.nist.gov/data/session/2014"

declare -A LOGS
LOGS[2011]="${SESSION_TRACKS[2011]}/sessiontrack2011.RL4.xml"
LOGS[2012]="${SESSION_TRACKS[2012]}/sessiontrack2012.txt"
LOGS[2013]="${SESSION_TRACKS[2013]}/sessiontrack2013.xml"
LOGS[2014]="${SESSION_TRACKS[2014]}/sessiontrack2014.xml"

declare -A QRELS
QRELS[2011]="${SESSION_TRACKS[2011]}/judgments.txt"
QRELS[2012]="${SESSION_TRACKS[2012]}/qrels.txt"
QRELS[2013]="${SESSION_TRACKS[2013]}/qrels.txt"
QRELS[2014]="${SESSION_TRACKS[2014]}/judgments.txt"

declare -A TOPIC_MAPS
TOPIC_MAPS[2011]="${SESSION_TRACKS[2011]}/sessionlastquery_subtopic_map.txt"
TOPIC_MAPS[2012]="${SESSION_TRACKS[2012]}/sessiontopicmap.txt"
TOPIC_MAPS[2013]="${SESSION_TRACKS[2013]}/sessiontopicmap.txt"
TOPIC_MAPS[2014]="${SESSION_TRACKS[2014]}/session-topic-mapping.txt"

if [[ -d "scratch" ]]; then
    echo "Output directory 'scratch' already exists."

    exit -1
fi

SCRATCH_DIR="$(pwd)/scratch"
check_directory_not_exists

mkdir -p "${SCRATCH_DIR}"

for SESSION_TRACK in "${!SESSION_TRACKS[@]}"; do
    TRACK_SCRATCH_DIR="${SCRATCH_DIR}/${SESSION_TRACK}"
    mkdir -p "${TRACK_SCRATCH_DIR}"

    echo "Fetching ${SESSION_TRACK} logs: ${LOGS[${SESSION_TRACK}]}"
    curl -s "${LOGS[${SESSION_TRACK}]}" \
        | sed 's/ & / \&amp; /g' \
        > "${TRACK_SCRATCH_DIR}/${SESSION_TRACK}.xml"

    echo "Fetching ${SESSION_TRACK} judgments: ${QRELS[${SESSION_TRACK}]}"
    curl -s "${QRELS[${SESSION_TRACK}]}" > "${TRACK_SCRATCH_DIR}/${SESSION_TRACK}.original_qrel"

    echo "Fetching ${SESSION_TRACK} session-topic map: ${TOPIC_MAPS[${SESSION_TRACK}]}"
    curl -s "${TOPIC_MAPS[${SESSION_TRACK}]}" > "${TRACK_SCRATCH_DIR}/${SESSION_TRACK}.session_topic_map"

    echo "Creating ${SESSION_TRACK} trec_eval-compatible qrel."
    python bin/trec/convert_qrel.py \
        --judgments "${TRACK_SCRATCH_DIR}/${SESSION_TRACK}.original_qrel" \
        --session_topic_map "${TRACK_SCRATCH_DIR}/${SESSION_TRACK}.session_topic_map" \
        --track_year "${SESSION_TRACK}" \
        --qrel_out "${TRACK_SCRATCH_DIR}/${SESSION_TRACK}.qrel" \
        &> /dev/null

    cat > "${TRACK_SCRATCH_DIR}/vars.sh" << EOL
TRACK_NAME="${SESSION_TRACK}"

LOG_FILE="${SESSION_TRACK}.xml"
TOPIC_MAP_FILE="${SESSION_TRACK}.session_topic_map"
QREL_FILE="${SESSION_TRACK}.qrel"
EOL
done