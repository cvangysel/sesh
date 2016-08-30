#!/usr/bin/env python

import sys

import sesh
from sesh import domain

from cvangysel import argparse_utils, logging_utils

import argparse
import collections
import logging


relevance_mapping_2012 = {
    -2: 0,
    0: 0,
    1: 1,
    4: 2,
    2: 3,
    3: 4,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loglevel', type=str, default='INFO')

    parser.add_argument('--judgments',
                        type=argparse_utils.existing_file_path,
                        required=True)
    parser.add_argument('--session_topic_map',
                        type=argparse_utils.existing_file_path,
                        required=True)

    parser.add_argument('--track_year',
                        type=int,
                        choices=domain.INPUT_FORMATS, required=True)

    parser.add_argument('--qrel_out',
                        type=argparse_utils.nonexisting_file_path,
                        required=True)

    args = parser.parse_args()

    try:
        logging_utils.configure_logging(args)
    except IOError:
        return -1

    topic_id_to_subtopics = collections.defaultdict(lambda: set([0]))
    topic_id_to_session_ids = collections.defaultdict(set)

    with open(args.session_topic_map, 'r') as f_mapping:
        for line in f_mapping:
            line = line.strip()

            if not line:
                continue

            try:
                data = line.strip().split()[:3]
            except:
                logging.warning('Unable to parse %s', line)

                continue

            if len(data) == 3:
                session_id, topic_id, subtopic_id = data
            else:
                session_id, topic_id, subtopic_id = data + [0]

            topic_id_to_subtopics[topic_id].add(subtopic_id)
            topic_id_to_session_ids[topic_id].add(session_id)

    with open(args.judgments, 'r') as f_judgments:
        qrel = sesh.parse_qrel(f_judgments, args.track_year)

    with open(args.qrel_out, 'w') as f_out:
        for topic_id, session_ids in topic_id_to_session_ids.items():
            relevant_items = qrel[topic_id]

            for session_id in session_ids:
                for document_id, relevance in relevant_items.items():
                    f_out.write(
                        '{session_id} 0 {document_id} {relevance}\n'.format(
                            session_id=session_id,
                            document_id=document_id,
                            relevance=relevance))

if __name__ == "__main__":
    sys.exit(main())
