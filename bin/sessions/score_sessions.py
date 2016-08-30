#!/usr/bin/env python

import sys

from cvangysel import argparse_utils, \
    logging_utils, multiprocessing_utils, pyndri_utils, trec_utils, rank_utils

import sesh
from sesh import domain, scorers, sesh_pb2

import google.protobuf as pb

import argparse
import codecs
import collections
import io
import logging
import multiprocessing
import numpy as np
import os
import operator
import pyndri
import scipy


class DocumentCandidateGenerator(object):

    def __init__(self, args, index, track_edition, **kwargs):
        self.args = args
        self.index = index

        assert args.qrel
        self.qrels = []

        for qrel_path in args.qrel:
            with open(qrel_path, 'r') as f_qrel:
                self.qrels.append(sesh.parse_qrel(f_qrel, track_edition))


class LemurDocumentCandidateGenerator(DocumentCandidateGenerator):

    def __init__(self, args, index, configuration, **kwargs):
        super(LemurDocumentCandidateGenerator, self).__init__(
            args, index, **kwargs)

        assert configuration.top_candidate_limit
        self.top_candidate_limit = configuration.top_candidate_limit

    def generate(self, session):
        indri_query = ' '.join(session.queries[-1])

        logging.debug('Querying Indri for "%s" (%s).',
                      indri_query, session)

        if not indri_query:
            return []

        return [
            int_document_id
            for int_document_id, _ in self.index.query(
                indri_query,
                results_requested=self.top_candidate_limit)]


class QrelDocumentCandidateGenerator(DocumentCandidateGenerator):

    def __init__(self, args, index, session_id_to_topic_id, track_edition,
                 per_topic=True, **kwargs):
        super(QrelDocumentCandidateGenerator, self).__init__(
            args, index, track_edition)

        self.session_id_to_topic_id = session_id_to_topic_id

        self.per_topic = per_topic

        self.candidate_documents_per_topic = collections.defaultdict(set)

        for qrel in self.qrels:
            for topic_id, document_ids_and_relevances in qrel.items():
                ext_document_ids_for_topic = list(
                    document_id for document_id in
                    document_ids_and_relevances)

                int_document_ids_for_topic = set(
                    int_document_id for _, int_document_id in
                    self.index.document_ids(ext_document_ids_for_topic))

                self.candidate_documents_per_topic[topic_id] |= \
                    int_document_ids_for_topic

    def generate(self, session):
        assert session.session_id in self.session_id_to_topic_id

        if self.per_topic:
            return self.candidate_documents_per_topic[
                str(self.session_id_to_topic_id[session.session_id])]
        else:
            return dict(
                candidate_document
                for topic_id, candidate_documents in
                self.candidate_documents_per_topic.items()
                for candidate_document in candidate_documents).items()


class ListDocumentCandidateGenerator(DocumentCandidateGenerator):

    def __init__(self, args, index, configuration, **kwargs):
        super(ListDocumentCandidateGenerator, self).__init__(
            args, index, **kwargs)

        assert configuration.document_list

        with open(configuration.document_list, 'r') as f_document_list:
            ext_document_ids = [line.strip() for line in f_document_list
                                if line.strip()]

            self.internal_document_ids = list(map(
                operator.itemgetter(1), index.document_ids(ext_document_ids)))

    def generate(self, session):
        return self.internal_document_ids


DOCUMENT_CANDIDATE_GENERATORS = {
    sesh_pb2.ScoreSessionsConfig.LEMUR: LemurDocumentCandidateGenerator,
    sesh_pb2.ScoreSessionsConfig.QREL: QrelDocumentCandidateGenerator,
    sesh_pb2.ScoreSessionsConfig.DOCUMENT_LIST: ListDocumentCandidateGenerator,
}


def score_session_initializer(
        _result_queue,
        _args,
        _configuration,
        _out_base,
        _background_prob_dist,
        _candidate_generator,
        _scorer_impls,
        _index,
        _dictionary,
        _anchor_texts):
    score_session_worker.result_queue = _result_queue
    score_session_worker.args = _args
    score_session_worker.configuration = _configuration
    score_session_worker.out_base = _out_base
    score_session_worker.background_prob_dist = _background_prob_dist
    score_session_worker.candidate_generator = _candidate_generator
    score_session_worker.scorer_impls = _scorer_impls
    score_session_worker.index = _index
    score_session_worker.dictionary = _dictionary
    score_session_worker.anchor_texts = _anchor_texts

    logging.info('Initialized worker.')


def score_session_worker_(session):
    logger = logging.getLogger(str(session))
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    log_formatter = logging_utils.get_formatter()

    # Set the output stream handler to the same loglevel.
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(logging.getLogger().level)
    stderr_handler.setFormatter(log_formatter)

    logger.addHandler(stderr_handler)

    logger.info('Scoring %s.', session)

    if not session.queries:
        logger.warning('Skipping %s due to no queries.', session)

        return

    session_id = session.session_id

    logger.info('Generating set of candidate documents for %s.',
                repr(session))

    # Generate candidate documents.
    candidate_internal_document_ids = score_session_worker.\
        candidate_generator.generate(session)

    word_frequency_index = pyndri_utils.create_word_frequency_index(
        score_session_worker.index,
        candidate_internal_document_ids,
        background_prob_dist=score_session_worker.background_prob_dist)

    candidate_doc_ids = set(iter(word_frequency_index))

    assert(all(isinstance(document_id, str)
               for document_id in candidate_doc_ids))

    logger.info('Scoring %d documents for %s.',
                len(candidate_doc_ids), session)

    ndcg_per_scorer_per_qrel = collections.defaultdict(dict)

    for scorer_name, scorer in score_session_worker.scorer_impls.items():
        logger.info('Scoring %s using %s.', session, scorer_name)

        f_scorer_debug_out = None

        if score_session_worker.out_base:
            debug_path = os.path.join(
                score_session_worker.out_base,
                '{scorer_name}_{session_id}'.format(
                    scorer_name=scorer_name,
                    session_id=session.session_id))

            logger.info('Writing debug information for scorer %s to %s.',
                        scorer_name, debug_path)

            f_scorer_debug_out = open(debug_path, 'w')

        if f_scorer_debug_out is not None:
            handler = logging.StreamHandler(f_scorer_debug_out)
        else:
            handler = logging.NullHandler()

        handler.setLevel(logging.DEBUG)
        handler.setFormatter(log_formatter)

        logger.addHandler(handler)

        full_ranking = dict(scorer.score_documents_for_session(
            session, candidate_doc_ids,
            score_session_worker.index, score_session_worker.dictionary,
            score_session_worker.anchor_texts,
            word_frequency_index,
            logger,
            f_scorer_debug_out))

        assert(all(isinstance(document_id, str)
                   for document_id in full_ranking))

        logger.removeHandler(handler)

        if f_scorer_debug_out:
            f_scorer_debug_out.close()

        if (not score_session_worker.
                configuration.retain_non_candidate_documents):
            filtered_ranking = {
                document_id: score
                for document_id, score in full_ranking.items()
                if document_id in candidate_doc_ids}

            num_dropped_documents = len(full_ranking) - len(filtered_ranking)
        else:
            filtered_ranking = full_ranking

            num_dropped_documents = 0

        if num_dropped_documents > 0:
            logger.warning('Dropped %d documents from %s method '
                           'due to out of candidates.',
                           num_dropped_documents, scorer_name)

        if scorer.DO_NOT_USE_qrels:
            ranks = rank_utils.generate_ranks(
                np.array(
                    [filtered_ranking[doc_id]
                     for doc_id in candidate_doc_ids
                     if doc_id in filtered_ranking]),
                axis=0)

            for qrel_idx, qrel in enumerate(scorer.DO_NOT_USE_qrels):
                qrel_lookup = dict(qrel[session.session_id])

                relevances = np.array(
                    [qrel_lookup.get(doc_id, 0)
                     for doc_id in candidate_doc_ids
                     if doc_id in filtered_ranking])

                try:
                    # NDCG computation is biased towards the candidate set.
                    #
                    # Only use this value for tracking on-going progress,
                    # not as a final ranking quality measure.
                    ndcg = rank_utils.compute_ndcg(ranks, relevances)
                except RuntimeError as e:
                    logging.error(e)

                    ndcg = np.nan

                ndcg_per_scorer_per_qrel[qrel_idx][scorer_name] = ndcg

        score_session_worker.result_queue.put(
            (scorer_name, session_id, filtered_ranking))

    for qrel, results in ndcg_per_scorer_per_qrel.items():
        logger.info('Within-candidate set NDCG for %s: %s',
                    qrel, ' '.join(
                        '{} ({})'.format(scorer_name, ndcg)
                        for scorer_name, ndcg in results.items()))

    logger.debug('Finished scoring %s.', session)

    return len(score_session_worker.scorer_impls)

score_session_worker = \
    multiprocessing_utils.WorkerFunction(score_session_worker_)


def binary_search_file(f, needle, key=lambda candidate: candidate):
    position = [0]

    f.seek(position[0], 2)
    length = f.tell()

    def goto_beginning_of_line():
        while position[0] >= 0:
            f.seek(position[0])
            if f.read(1) == '\n':
                break

            position[0] -= 1

        # Make sure we do not fall off the file.
        if position[0] < 0:
            f.seek(0)

    low = 0
    high = length
    while low < high:
        mid = (low + high) // 2
        position[0] = mid

        # Crawl back to beginning of line.
        goto_beginning_of_line()

        # Read current line.
        candidate = f.readline().strip()

        # Figure out which part of the file to search.
        if key(candidate) < needle:
            low = mid + 1
        else:
            high = mid

    position[0] = low

    # Crawl back to beginning of line.
    goto_beginning_of_line()

    result = []

    while True:
        candidate = f.readline().strip()

        if not candidate or not key(candidate) == needle:
            break

        result.append(candidate)

    return result


def load_anchor_texts(f_harvested_links, urls, encoding='utf8'):
    urls = sorted(urls, reverse=True)

    anchor_texts = collections.defaultdict(list)

    for url in urls:
        matches = binary_search_file(
            f_harvested_links, url,
            key=lambda candidate: candidate.split('\t')[0])

        for match in matches:
            splitted_match = match.split('\t', 3)

            if len(splitted_match) != 3:
                logging.error('Unable to parse line while loading '
                              'anchor texts: %s', match)

                continue

            _, _, anchor_text = splitted_match

            anchor_texts[url].append(anchor_text)

    return anchor_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loglevel', type=str, default='INFO')

    parser.add_argument('index', type=argparse_utils.existing_file_path)

    parser.add_argument('session_file', type=argparse_utils.existing_file_path)

    parser.add_argument('--num_workers', type=int, default=1)

    parser.add_argument('--harvested_links_file',
                        type=argparse_utils.existing_file_path,
                        default=None)

    parser.add_argument('--qrel',
                        type=argparse_utils.existing_file_path,
                        nargs='*')

    parser.add_argument('--configuration', type=str, nargs='+')

    parser.add_argument('--top_sessions',
                        type=argparse_utils.positive_int,
                        default=None)

    parser.add_argument('--out_base',
                        type=argparse_utils.nonexisting_file_path,
                        required=True)

    args = parser.parse_args()

    try:
        logging_utils.configure_logging(args)
    except IOError:
        return -1

    logging_utils.log_module_info(np, scipy)

    configuration = sesh_pb2.ScoreSessionsConfig()
    pb.text_format.Merge(' '.join(args.configuration), configuration)

    if not configuration.modifier:
        configuration.modifier.add()  # Create an empty modifier.
    elif len(configuration.modifier) > 1:
        modifier_identifiers = [
            modifier.identifier for modifier in configuration.modifier]

        assert all(modifier_identifiers), \
            'All session modifiers should have an identifier.'

        assert len(modifier_identifiers) == len(set(modifier_identifiers)), \
            'All session modifier identifiers should be unique: {}.'.format(
                modifier_identifiers)

    logging.info('Configuration: %s', configuration)

    logging.info('Loading index.')
    index = pyndri.Index(args.index)

    num_documents = index.document_count()
    logging.debug('Index contains %d documents.', num_documents)

    logging.info('Loading dictionary.')
    dictionary = pyndri.extract_dictionary(index)

    logging.info('Loading background corpus.')
    background_prob_dist = pyndri_utils.extract_background_prob_dist(index)

    for modifier in configuration.modifier:
        out_base = os.path.join(args.out_base, modifier.identifier)
        assert not os.path.exists(out_base)

        os.makedirs(out_base)

        logging.info('Loading sessions using %s and outputting to %s.',
                     modifier or 'no modifier', out_base)

        with codecs.open(args.session_file, 'r', 'utf8') as f_xml:
            track_edition, _, sessions, session_id_to_topic_id = \
                domain.construct_sessions(
                    f_xml, args.top_sessions, dictionary)

        logging.info('Discovered %d sessions.', len(sessions))

        sessions = domain.alter_sessions(sessions, modifier)
        documents = domain.get_document_set(sessions.values())

        logging.info('Retained %d sessions (%d SERP documents) '
                     'after filtering.',
                     len(sessions), len(documents))

        # Load QRels for debugging and oracle runs.
        qrels_per_session = []

        for qrel_path in args.qrel:
            with open(qrel_path, 'r') as f_qrel:
                qrels_per_session.append(sesh.parse_qrel(f_qrel, None))

        scorer_impls = {}

        for scorer_desc in configuration.scorer:
            assert scorer_desc.type in scorers.SESSION_SCORERS

            identifier = scorer_desc.identifier or scorer_desc.type
            assert identifier not in scorer_impls

            scorer = scorers.create_scorer(scorer_desc, qrels_per_session)
            logging.info('Scoring using %s.', repr(scorer))

            scorer_impls[identifier] = scorer

        anchor_texts = None

        if args.harvested_links_file is not None:
            urls = set(document.url for document in documents)

            logging.info('Loading anchor texts for session SERPs (%d URLs).',
                         len(urls))

            with codecs.open(args.harvested_links_file, 'r', 'latin1') \
                    as f_harvested_links:
                anchor_texts = load_anchor_texts(f_harvested_links, urls)

            logging.info('Discovered anchor texts for %d URLs (%d total).',
                         len(anchor_texts), len(urls))
        else:
            logging.info('No anchor texts loaded.')

        # The following will hold all the rankings.
        document_assessments_per_session_per_scorer = collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.defaultdict(float)))

        assert configuration.candidate_generator in \
            DOCUMENT_CANDIDATE_GENERATORS

        # Document candidate generation.
        candidate_generator = DOCUMENT_CANDIDATE_GENERATORS[
            configuration.candidate_generator](**locals())

        logging.info('Using %s for document candidate generation.',
                     candidate_generator)

        result_queue = multiprocessing.Queue()

        initargs = [
            result_queue,
            args,
            configuration,
            out_base,
            background_prob_dist,
            candidate_generator,
            scorer_impls,
            index,
            dictionary,
            anchor_texts]

        pool = multiprocessing.Pool(
            args.num_workers,
            initializer=score_session_initializer,
            initargs=initargs)

        worker_result = pool.map_async(
            score_session_worker,
            sessions.values())

        # We will not submit any more tasks to the pool.
        pool.close()

        it = multiprocessing_utils.QueueIterator(
            pool, worker_result, result_queue)

        while True:
            try:
                result = next(it)
            except StopIteration:
                break

            scorer_name, session_id, ranking = result

            document_assessments_per_session_per_scorer[
                scorer_name][session_id] = ranking

        for scorer_name in document_assessments_per_session_per_scorer:
            # Switch object asssessments to lists.
            for topic_id, object_assesments in \
                    document_assessments_per_session_per_scorer[
                        scorer_name].items():
                document_assessments_per_session_per_scorer[
                    scorer_name][topic_id] = [
                    (score, document_id)
                    for document_id, score in object_assesments.items()]

        # Write the runs.
        for scorer_name in document_assessments_per_session_per_scorer:
            run_out_path = os.path.join(
                out_base, '{0}.run'.format(scorer_name))

            with io.open(run_out_path, 'w', encoding='utf8') as f_run_out:
                trec_utils.write_run(
                    scorer_name,
                    document_assessments_per_session_per_scorer[scorer_name],
                    f_run_out)

if __name__ == "__main__":
    sys.exit(main())
