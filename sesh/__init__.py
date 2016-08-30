from sesh import domain

import itertools
import logging
import operator


# From session_eval_main.py
__RELEVANCE_MAPPING_2012 = {
    -2: 0,
    0: 0,
    1: 1,
    4: 2,
    2: 3,
    3: 4,
}


def parse_qrel(f_qrel, track_edition):
    assert track_edition in domain.INPUT_FORMATS or track_edition is None

    lookup_qrel = {}

    for line in f_qrel:
        topic_id, subtopic_id, document_id, relevance = \
            line.strip().split()

        if track_edition is None:
            topic_id = int(topic_id)

        if topic_id not in lookup_qrel:
            lookup_qrel[topic_id] = {}

        relevance = int(relevance)

        if track_edition == domain.SESSION2012:
            relevance = __RELEVANCE_MAPPING_2012[relevance]

        lookup_qrel[topic_id][document_id] = max(
            lookup_qrel[topic_id].get(document_id, relevance), relevance)

    qrel = {}

    for topic_id, relevant_items in lookup_qrel.items():
        qrel[topic_id] = list(relevant_items.items())

    return qrel


def qrel_per_session(qrel, session_id_to_topic_id):
    topic_id_to_session_ids = dict(map(
        lambda item: (str(item[0]),
                      list(map(operator.itemgetter(1), item[1]))),
        itertools.groupby(
            sorted(
                (topic_id, session_id)
                for (session_id, topic_id) in
                session_id_to_topic_id.items()),
            key=operator.itemgetter(0))))

    qrel_per_session = {}

    for topic_id, relevant_items in qrel.items():
        relevant_items = list(relevant_items.items())

        if topic_id not in topic_id_to_session_ids:
            logging.warning('Skipping relevance for topic %s.', topic_id)

            continue

        session_ids = topic_id_to_session_ids[topic_id]

        for session_id in session_ids:
            assert session_id not in qrel_per_session

            qrel_per_session[session_id] = relevant_items

            logging.info(
                'Discovered %d document-relevance pairs for session %d.',
                len(relevant_items), session_id)

    return qrel_per_session
